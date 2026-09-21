"""One real CPU generation sequence consumes its own HTTP-retrieved KV.

Tiny Llama + local V shards. Fixed draft candidate ids, exact CPU search,
greedy target output; no Scheduler/TP transport/GPU/latency or quality claim.
"""

import asyncio
from types import SimpleNamespace

import torch


def validate_controlled_decode(runner):
    from pvd_controlled_prefetch import ControlledFixture
    from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
        CPUDecodeLifecycle,
        TargetExecutionArbiter,
    )
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
    from sglang.srt.disaggregation.pvd.prediction import (
        CommittedPrefix,
        DraftConfig,
        FakeDraftProvider,
        PredictionPipeline,
        ProbeConfig,
    )
    from sglang.srt.disaggregation.pvd.sparse_cpu_backend import (
        SparseDecodeBinding,
        make_offline_sparse_backend,
    )
    from sglang.srt.disaggregation.pvd.sparse_install import (
        CPUInstalledPromptView,
        InstallProtocolError,
    )
    from sglang.srt.disaggregation.pvd.target_probe import OfflineLlamaTargetProbe
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    pool, req_pool, native = (
        runner.token_to_kv_pool,
        runner.req_to_token_pool,
        runner.attn_backend,
    )
    backend = make_offline_sparse_backend(runner)
    config = runner.model.config
    layers, heads = config.num_hidden_layers, config.num_key_value_heads
    dim = config.hidden_size // config.num_attention_heads
    prompt = (1, 4, 13, 7, 22)
    capacity = (
        len(req_pool.free_slots),
        runner.token_to_kv_pool_allocator.available_size(),
    )

    async def run(inject_failure=False):
        allocator = PrivatePoolAllocator(req_pool, runner.token_to_kv_pool_allocator)
        adapter = DraftForwardAdapter(
            runner,
            architecture="LlamaForCausalLM",
            attention_backend="torch_native",
            bytes_per_token=256,
            device="cpu",
        )
        slot, rows, fixture, task = allocator.alloc_request(), [], None, None
        lifecycle = None
        hooks, errors, seen_generations = [], [], []
        generated, snapshots, epochs = [], [], []
        active = {}
        probe_budget = TransferBudget(2 << 20, 1)

        def check_attention(module, args, output):
            if not active:
                return
            q, _, _, batch = args[:4]
            layer, pos = module.layer_id, int(batch.positions[0])
            with active["view"].read() as groups:
                expected = []
                gen_rows = req_pool.req_to_token[slot, len(prompt) : pos + 1].long()
                for qhead in range(config.num_attention_heads):
                    head = qhead // (config.num_attention_heads // heads)
                    spec, data = groups[layer, head]
                    seen_generations.append(
                        (active["count"], spec.operation_id, spec.target_tokens)
                    )
                    # Independent explicit softmax; Prompt bytes have already
                    # been checked against authoritative V source in pack_source.
                    keys = torch.cat(
                        (data[0], pool.get_key_buffer(layer)[gen_rows, head])
                    )
                    values = torch.cat(
                        (data[1], pool.get_value_buffer(layer)[gen_rows, head])
                    )
                    expected.append(
                        (
                            (keys @ q.reshape(config.num_attention_heads, dim)[qhead])
                            * module.scaling
                        ).softmax(0)
                        @ values
                    )
                expected = torch.stack(expected).reshape_as(output)
                torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-4)
                errors.append(float((output - expected).abs().max()))
            if inject_failure and active["count"] == 5:
                raise RuntimeError("controlled model failure after sparse write")

        def snapshot():
            n = len(generated) - 1  # P first token is output but not a D refresh tick
            result = lifecycle.snapshot()
            assert result.committed_position == n and result.tokens == prompt + tuple(
                generated
            )
            snapshots.append(result)
            return result

        def step():
            n = len(generated) - 1
            assert lifecycle.can_decode()
            assert lifecycle.committed_tokens == n
            view = CPUInstalledPromptView(fixture.group, n)
            position = len(prompt) + n
            old_rows = rows[len(prompt) :]
            previous = [
                (
                    pool.get_key_buffer(l)[old_rows].clone(),
                    pool.get_value_buffer(l)[old_rows].clone(),
                )
                for l in range(layers)
            ]
            new_rows = allocator.alloc_kv(1)
            rows.extend(new_rows)
            allocator.write_mapping(slot, position, new_rows)
            active.update(view=view, count=n)
            runner.attn_backend = backend
            permit = lifecycle.begin_decode()
            assert (
                permit.input_token == generated[-1]
                and permit.query_position == position
            )
            try:
                with backend.consumer.bind(
                    [
                        SparseDecodeBinding(
                            slot,
                            "controlled-decode",
                            fixture.incarnation,
                            position,
                            view,
                        )
                    ]
                ):
                    logits = adapter.forward(
                        DraftForwardInputs(
                            "decode",
                            (generated[-1],),
                            (position,),
                            (position + 1,),
                            (slot,),
                            tuple(new_rows),
                        )
                    )
                assert torch.isfinite(logits).all()
                token = int(logits.argmax())
                assert lifecycle.complete_decode(permit, token)
                generated.append(token)  # ONLY target output commits
            except BaseException:
                if lifecycle._permit is permit:
                    lifecycle.fail_decode(
                        permit, "model forward failed; no retry/rollback"
                    )
                raise
            finally:
                runner.attn_backend = native
                active.clear()
            for l, (k, v) in enumerate(previous):
                torch.testing.assert_close(
                    pool.get_key_buffer(l)[old_rows], k, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    pool.get_value_buffer(l)[old_rows], v, atol=0, rtol=0
                )

        def assert_parked(n):
            # Attempt the actual consumer binding, not only can_decode(). It
            # must fail BEFORE a model forward or a generated KV write.
            before = adapter.forward_count
            view = CPUInstalledPromptView(fixture.group, n)
            try:
                with backend.consumer.bind(
                    [
                        SparseDecodeBinding(
                            slot,
                            "controlled-decode",
                            fixture.incarnation,
                            len(prompt) + n,
                            view,
                        )
                    ]
                ):
                    raise AssertionError("boundary admitted an uninstalled forward")
            except InstallProtocolError:
                pass
            assert adapter.forward_count == before

        try:
            rows = allocator.alloc_kv(len(prompt))
            allocator.write_mapping(slot, 0, rows)
            logits = adapter.forward(
                DraftForwardInputs(
                    "extend",
                    prompt,
                    tuple(range(len(prompt))),
                    (len(prompt),),
                    (slot,),
                    tuple(rows),
                    (0,),
                    (len(prompt),),
                )
            )
            generated.append(int(logits.argmax()))  # P's real first token
            prompt_pool = SimpleNamespace(
                k_buffer=[pool.get_key_buffer(l)[rows].clone() for l in range(layers)],
                v_buffer=[
                    pool.get_value_buffer(l)[rows].clone() for l in range(layers)
                ],
            )
            pc = ProbeConfig(
                "controlled-model",
                tuple(range(layers)),
                head_count=config.num_attention_heads,
            )
            probe = OfflineLlamaTargetProbe(
                runner,
                pc,
                target_model_id=pc.target_model_id,
                max_tokens=32,
                max_predict_tokens=2,
                transient_bytes_bound=1 << 20,
                budget=probe_budget,
            )
            dc = DraftConfig("fixed-draft-fixture", predict_tokens=2)
            provider = FakeDraftProvider(dc, tokens=(19, 27))
            pipeline = PredictionPipeline(provider, probe, dc, pc)
            fixture = ControlledFixture(
                prompt_pool,
                CommittedPrefix("controlled-decode", prompt, 0, "prompt"),
                pipeline,
            )
            lifecycle = CPUDecodeLifecycle(
                "controlled-decode",
                prompt,
                generated[0],
                arbiter=TargetExecutionArbiter(),
            )
            lifecycle.admit(fixture.request)
            for l in range(layers):
                pool.get_key_buffer(l)[rows] = float("nan")
                pool.get_value_buffer(l)[rows] = float("nan")
            req_pool.req_to_token[slot, : len(prompt)] = -1
            hooks = [
                layer.self_attn.attn.register_forward_hook(check_attention)
                for layer in runner.model.model.layers
            ]
            async with fixture.clients() as clients:
                for _ in range(3):
                    step()
                prefix = snapshot()
                entered, release = asyncio.Event(), asyncio.Event()

                class Delayed:
                    async def search(self, *args, **kwargs):
                        entered.set()
                        await release.wait()
                        return await clients[1].search(*args, **kwargs)

                task = lifecycle.launch_refresh(
                    query_positions=(len(prefix.tokens),),
                    clients={0: clients[0], 1: Delayed()},
                    pack_source=fixture.pack_source,
                    timeout_seconds=30,
                )
                await asyncio.wait_for(entered.wait(), 5)
                step()  # n=3->4 while shard HTTP result is delayed
                assert not lifecycle.can_decode()
                assert not lifecycle.try_install({0: 4, 1: 4})
                assert_parked(4)
                release.set()
                epochs.append(await asyncio.wait_for(task, 5))
                assert lifecycle.try_install({0: 4, 1: 4})
                while len(generated) - 1 < 8:
                    step()
                # Intentionally miss the next prefetch window: actual-prefix
                # fallback must not invoke draft or shift the boundary.
                assert_parked(8)
                prefix = snapshot()
                epochs.append(
                    await lifecycle.launch_refresh(
                        query_positions=(len(prefix.tokens) - 1,),
                        clients=clients,
                        pack_source=fixture.pack_source,
                        timeout_seconds=30,
                    )
                )
                assert lifecycle.try_install({0: 8, 1: 8})
                step()
            assert len(provider.calls) == 1
            assert [p.committed_position for p in snapshots] == [3, 8]
            assert all(
                p.tokens == prompt + tuple(generated[: p.committed_position + 1])
                for p in snapshots
            )
            assert len({e.operation_id for e in epochs}) == 2
            for count, operation, boundary in seen_generations:
                assert boundary == (0 if count < 4 else 4 if count < 8 else 8)
                if count >= 4:
                    assert operation == epochs[0 if count < 8 else 1].operation_id
            assert any(len(s.token_ids) < len(prompt) for s in fixture.packed_specs)
            return {
                "committed_d_tokens": len(generated) - 1,
                "installed_boundaries": [e.target_tokens for e in epochs],
                "attention_checks": len(errors),
                "max_attention_error": max(errors),
                "actual_prefixes": True,
                "delayed_http_overlaps_one_cpu_decode": True,
                "poisoned_prompt_pool_not_read": True,
                "lifecycle_dispatch_commit_install": True,
            }
        except RuntimeError as exc:
            if not inject_failure or "controlled model failure" not in str(exc):
                raise
            assert len(generated) - 1 == 5  # failed forward emitted nothing
            assert not fixture.request.can_decode(5)
            assert lifecycle.state == "aborted" and lifecycle.committed_tokens == 5
            assert backend.consumer._bound is None
            return {"failure_aborts_without_committing_token": True}
        finally:
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            for hook in hooks:
                hook.remove()
            runner.attn_backend = native
            if lifecycle is not None:
                await lifecycle.close()
            if fixture is not None:
                fixture.close()
            allocator.clear_mapping(slot)
            allocator.free_kv(rows)
            allocator.free_request(slot)
            assert probe_budget.snapshot()["used_staging_bytes"] == 0
            assert (
                len(req_pool.free_slots),
                runner.token_to_kv_pool_allocator.available_size(),
            ) == capacity

    result = asyncio.run(run())
    failure = asyncio.run(run(inject_failure=True))
    assert failure["failure_aborts_without_committing_token"]
    return {
        "status": "passed",
        **result,
        **failure,
        "gpu_tp_rdma_scheduler_latency_validated": False,
    }
