"""Real CPU multi-request ForwardBatch + HTTP refresh + identity-safe commits.

No production Scheduler/TP/GPU/Mooncake claim. Greedy outputs and fixed draft
candidate ids, bounded tiny-model test only. Hooks are independent test oracles.
"""

import asyncio
from types import SimpleNamespace

import torch


def validate_batch_decode(
    runner,
    *,
    scheduled_results=False,
    draft_provider=None,
    automatic_refresh=False,
    wire_delivery=False,
    rank_runtime=False,
    rank_fault="none",
):
    if rank_fault not in ("none", "lost-resume", "install", "cleanup") or (
        rank_fault != "none"
        and not (rank_runtime and scheduled_results and wire_delivery)
    ):
        raise ValueError("rank faults require the complete scheduled wire/rank loop")
    from pvd_controlled_prefetch import ControlledFixture
    from sglang.srt.disaggregation.pvd.cpu_batch_dispatch import (
        CPUBatchDispatcher,
        batch_results_from_logits,
    )
    from sglang.srt.disaggregation.pvd.cpu_batch_forward import (
        CPUBatchForwardExecutor,
        CPUForwardDestination,
    )
    from sglang.srt.disaggregation.pvd.cpu_decode_lifecycle import (
        CPUDecodeLifecycle,
        LifecycleError,
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
    from sglang.srt.disaggregation.pvd.sparse_install import CPUInstalledPromptView
    from sglang.srt.disaggregation.pvd.target_probe import OfflineLlamaTargetProbe
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    pool, reqpool, native = (
        runner.token_to_kv_pool,
        runner.req_to_token_pool,
        runner.attn_backend,
    )
    arbiter = TargetExecutionArbiter()
    if rank_runtime:
        from sglang.srt.disaggregation.pvd.cpu_rank_batch import CPURankBatchDispatcher

        dispatcher = CPURankBatchDispatcher(arbiter, max_requests=8)
    else:
        dispatcher = CPUBatchDispatcher(arbiter)
    executor = CPUBatchForwardExecutor(runner, dispatcher)
    backend = executor.backend
    builder = DraftForwardAdapter(
        runner,
        architecture="LlamaForCausalLM",
        attention_backend="torch_native",
        bytes_per_token=256,
        device="cpu",
    )
    config = runner.model.config
    layers, heads, qheads = (
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.num_attention_heads,
    )
    dim = config.hidden_size // qheads
    before = (
        len(reqpool.free_slots),
        runner.token_to_kv_pool_allocator.available_size(),
    )
    records, active, errors, sizes = {}, {}, [], []
    budgets, tasks, hooks = [], [], []
    inject_failure = False
    fault_evidence = {"mode": rank_fault}
    reuse_evidence = {}
    held_slots, held_rows = [], []
    pressure_allocator = PrivatePoolAllocator(
        reqpool, runner.token_to_kv_pool_allocator
    )
    processor = None
    driver = None
    if automatic_refresh:
        from sglang.srt.disaggregation.pvd.cpu_refresh_driver import CPURefreshDriver

        driver = CPURefreshDriver(arbiter)
    if scheduled_results:
        from pvd_scheduled_result_smoke import (
            deliver,
            make_batch,
            make_processor,
            make_req,
        )
        from sglang.srt.disaggregation.pvd.cpu_schedule_bridge import CPUScheduleBridge

        processor = make_processor()

    def create(name, tokens):
        allocator = PrivatePoolAllocator(reqpool, runner.token_to_kv_pool_allocator)
        slot = allocator.alloc_request()
        r = SimpleNamespace(
            slot=slot,
            rows=[],
            fixture=None,
            life=None,
            prompt=tokens,
            allocator=allocator,
            retired=False,
            releasing=False,
            refresh_registered=False,
            stale_result=None,
        )
        records[name] = r
        lease = arbiter.acquire()
        try:
            r.rows = allocator.alloc_kv(len(tokens))
            allocator.write_mapping(slot, 0, r.rows)
            logits = builder.forward(
                DraftForwardInputs(
                    "extend",
                    tokens,
                    tuple(range(len(tokens))),
                    (len(tokens),),
                    (slot,),
                    tuple(r.rows),
                    (0,),
                    (len(tokens),),
                )
            )
            stored = SimpleNamespace(
                k_buffer=[
                    pool.get_key_buffer(l)[r.rows].clone() for l in range(layers)
                ],
                v_buffer=[
                    pool.get_value_buffer(l)[r.rows].clone() for l in range(layers)
                ],
            )
            pc = ProbeConfig("batch-target", tuple(range(layers)), head_count=qheads)
            dc = (
                DraftConfig("fixed-draft-fixture", predict_tokens=2)
                if draft_provider is None
                else draft_provider.config
            )
            budget = TransferBudget(2 << 20, 1)
            budgets.append(budget)
            probe = OfflineLlamaTargetProbe(
                runner,
                pc,
                target_model_id=pc.target_model_id,
                max_tokens=32,
                max_predict_tokens=2,
                transient_bytes_bound=1 << 20,
                budget=budget,
            )
            pipeline = PredictionPipeline(
                FakeDraftProvider(dc, tokens=(19, 27))
                if draft_provider is None
                else draft_provider,
                probe,
                dc,
                pc,
            )
            r.fixture = ControlledFixture(
                stored,
                CommittedPrefix(name, tokens, 0, "prompt"),
                pipeline,
                rank_runtime=rank_runtime,
            )
            r.life = CPUDecodeLifecycle(
                name, tokens, int(logits.argmax()), arbiter=arbiter
            )
            executor.register_storage(r.life, r.slot)
            if scheduled_results:
                r.req = make_req(r.life, r.slot, config.vocab_size)
            for l in range(layers):
                pool.get_key_buffer(l)[r.rows] = float("nan")
                pool.get_value_buffer(l)[r.rows] = float("nan")
            reqpool.req_to_token[slot, : len(tokens)] = -1
        finally:
            arbiter.release(lease)
        return r

    async def retire(r):
        """Fixture-owned resources only, after actual execution/Delivery drain.

        A failed allocator release has unknown side effects: never replay it.
        This is not a production cache-release hook or a native RDMA fence.
        """
        if r.retired:
            return
        if r.releasing:
            raise AssertionError("ambiguous pool release is quarantined, not retried")
        if r.life is not None:
            if r.refresh_registered:
                await driver.remove(r.life)
                r.refresh_registered = False
            else:
                await r.life.close()
            executor.unregister_storage(r.life)
        if r.fixture is not None:
            r.fixture.close()
        r.releasing = True
        r.allocator.clear_mapping(r.slot)
        r.allocator.free_kv(r.rows)
        r.allocator.free_request(r.slot)
        if scheduled_results and hasattr(r, "req"):
            r.req.req_pool_idx = None
        r.retired = True

    def release_pressure():
        nonlocal held_rows
        for allocator, slot in held_slots:
            allocator.free_request(slot)
        held_slots.clear()
        pressure_allocator.free_kv(held_rows)
        held_rows = []

    def oracle(module, args, output):
        if not active:
            return
        q, _, _, batch = args[:4]
        expected = []
        for i, slot in enumerate(batch.req_pool_indices.tolist()):
            record, view = active[slot]
            position = int(batch.positions[i])
            gen = reqpool.req_to_token[slot, len(record.prompt) : position + 1].long()
            with view.read() as groups:
                row = []
                for h in range(qheads):
                    head = h // (qheads // heads)
                    _, data = groups[module.layer_id, head]
                    keys = torch.cat(
                        (data[0], pool.get_key_buffer(module.layer_id)[gen, head])
                    )
                    vals = torch.cat(
                        (data[1], pool.get_value_buffer(module.layer_id)[gen, head])
                    )
                    row.append(
                        (
                            (keys @ q.reshape(-1, qheads, dim)[i, h]) * module.scaling
                        ).softmax(0)
                        @ vals
                    )
                expected.append(torch.stack(row))
        expected = torch.stack(expected).reshape_as(output)
        torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-4)
        errors.append(float((output - expected).abs().max()))
        if inject_failure:
            raise RuntimeError("actual batched attention failure")

    def step(names, cancel_after_forward=None):
        ticket = dispatcher.begin([records[n].life for n in names])
        destinations, prior = [], []
        try:
            if scheduled_results:
                scheduled_batch = make_batch([records[n].req for n in names])
                bridge = CPUScheduleBridge(executor, scheduled_batch, ticket)
            for member in ticket.members:
                r, p = records[member.request_id], member.permit
                old_rows = r.rows[len(r.prompt) :]
                prior.append(
                    (
                        r,
                        old_rows,
                        [
                            (
                                pool.get_key_buffer(l)[old_rows].clone(),
                                pool.get_value_buffer(l)[old_rows].clone(),
                            )
                            for l in range(layers)
                        ],
                    )
                )
                new = r.allocator.alloc_kv(1)
                r.rows.extend(new)
                r.allocator.write_mapping(r.slot, p.query_position, new)
                destinations.append(CPUForwardDestination(r.life, r.slot, new[0]))
                view = CPUInstalledPromptView(r.fixture.group, p.committed_tokens)
                active[r.slot] = r, view
            sizes.append(len(ticket.members))
            if len(destinations) == 2:
                from dataclasses import replace

                swapped = [
                    replace(destinations[0], slot=destinations[1].slot),
                    replace(destinations[1], slot=destinations[0].slot),
                ]
                before_forward = executor.forward_count
                try:
                    executor.forward(ticket, swapped)
                except LifecycleError as exc:
                    assert "registered request ownership" in str(exc)
                else:
                    raise AssertionError("cross-request slot swap was admitted")
                assert executor.forward_count == before_forward
            logits = executor.forward(ticket, tuple(reversed(destinations)))
            if executor.forward_count == 1:
                try:
                    executor.forward(ticket, destinations)
                except LifecycleError as exc:
                    assert "same dispatch twice" in str(exc)
                else:
                    raise AssertionError("same real forward executed twice")
                assert executor.forward_count == 1 and runner.attn_backend is native
            results = batch_results_from_logits(
                ticket, logits, finished=(False,) * len(names)
            )
            # Simulate cancellation arriving while execution was in flight:
            # execution has stopped, but no output has yet been committed.
            if cancel_after_forward is not None:
                if rank_fault == "cleanup":
                    from sglang.srt.disaggregation.pvd.sparse_install import (
                        InstallProtocolError,
                    )

                    group = records[cancel_after_forward].fixture.group
                    banks = tuple(group._banks.values())
                    charges = [
                        bank.budget.snapshot()["used_staging_bytes"] for bank in banks
                    ]
                    currents = [bank._current for bank in banks]
                    try:
                        group.close()
                    except InstallProtocolError as exc:
                        assert "drain before close" in str(exc)
                    else:
                        raise AssertionError("close freed an owned result ticket")
                    assert group.runtime._forward is not None and arbiter.busy
                    assert all(
                        bank._current is current
                        for bank, current in zip(banks, currents, strict=True)
                    )
                    assert charges == [
                        bank.budget.snapshot()["used_staging_bytes"] for bank in banks
                    ]
                    fault_evidence["close_refused_without_releasing_bank"] = True
                elif scheduled_results:
                    records[cancel_after_forward].req.is_retracted = True
                else:
                    records[cancel_after_forward].life.terminate("client cancelled")
                assert arbiter.busy
            if scheduled_results:
                deliver(processor, bridge, scheduled_batch, logits)
                if cancel_after_forward is not None:
                    # Keep this one old callback only until the reuse check.
                    records[cancel_after_forward].stale_result = lambda: deliver(
                        processor, bridge, scheduled_batch, logits
                    )
                committed = tuple(
                    r for r in results if r.request_id != cancel_after_forward
                )
            else:
                committed = dispatcher.complete(ticket, tuple(reversed(results)))
            if cancel_after_forward is not None and rank_fault == "cleanup":
                group.close()
                assert all(
                    bank._current is None and bank._next is None for bank in banks
                )
                fault_evidence["close_retry_after_result_drain_succeeded"] = True
                assert (
                    tuple(records[cancel_after_forward].req.output_ids)
                    == records[cancel_after_forward].life.outputs
                )
                fault_evidence["cancelled_output_discarded"] = True
            expected_ids = set(names) - (
                {cancel_after_forward} if cancel_after_forward else set()
            )
            assert {r.request_id for r in committed} == expected_ids
            for result in committed:
                assert records[result.request_id].life.outputs[-1] == result.token
            for r, old_rows, tensors in prior:
                for l, (k, v) in enumerate(tensors):
                    torch.testing.assert_close(
                        pool.get_key_buffer(l)[old_rows], k, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        pool.get_value_buffer(l)[old_rows], v, atol=0, rtol=0
                    )
            return ticket
        except BaseException:
            if dispatcher._ticket is ticket:
                dispatcher.fail(
                    ticket, "batch execution failed; discard all partial output"
                )
            raise
        finally:
            runner.attn_backend = native
            active.clear()

    async def run():
        nonlocal inject_failure
        try:
            old = create("old", (1, 4, 13, 7, 22))
            new = create("new", (1, 6, 17))  # different absolute positions in batch
            hooks.extend(
                layer.self_attn.attn.register_forward_hook(oracle)
                for layer in runner.model.model.layers
            )
            async with (
                old.fixture.clients(wire_delivery=wire_delivery) as clients,
                new.fixture.clients(wire_delivery=wire_delivery) as newclients,
            ):
                old.life.admit(old.fixture.request)
                for _ in range(3):
                    step(("old",))
                entered, release = asyncio.Event(), asyncio.Event()

                class Delayed:
                    async def search(self, *args, **kwargs):
                        entered.set()
                        await release.wait()
                        return await clients[1].search(*args, **kwargs)

                prefix = old.life.snapshot()
                if driver is not None:
                    driver.register(
                        old.life,
                        clients={0: clients[0], 1: Delayed()},
                        pack_source=None if wire_delivery else old.fixture.pack_source,
                        timeout_seconds=30,
                    )
                    old.refresh_registered = True
                    launch = driver.progress().launched[0]
                    assert launch.query_source == "predicted"
                    task = launch.task
                else:
                    task = old.life.launch_refresh(
                        query_positions=(len(prefix.tokens),),
                        clients={0: clients[0], 1: Delayed()},
                        pack_source=None if wire_delivery else old.fixture.pack_source,
                        timeout_seconds=30,
                    )
                tasks.append(task)
                await asyncio.wait_for(entered.wait(), 5)
                state = old.fixture.group.coordinator.snapshot()
                new.life.admit(new.fixture.request)
                if driver is not None:
                    driver.register(
                        new.life,
                        clients=newclients,
                        pack_source=None if wire_delivery else new.fixture.pack_source,
                        timeout_seconds=30,
                    )
                    new.refresh_registered = True
                assert old.fixture.group.coordinator.snapshot() == state
                step(("new", "old"))
                counts = old.life.committed_tokens, new.life.committed_tokens
                assert counts == (4, 1)
                try:
                    dispatcher.begin([new.life, old.life])
                except LifecycleError:
                    pass
                else:
                    raise AssertionError("wait-all policy dispatched a partial batch")
                assert not arbiter.busy and new.life._permit is None
                release.set()
                epoch = await task
                if rank_fault == "lost-resume":
                    from sglang.srt.disaggregation.pvd.rank_install_wire import (
                        RankInstallMessage,
                    )

                    group, dropped = old.fixture.group, []
                    post = group._post

                    def drop(rank, raw):
                        if (
                            rank == 1
                            and RankInstallMessage.decode(raw).kind == "resumed"
                        ):
                            dropped.append(raw)
                        else:
                            post(rank, raw)

                    before_outputs = (
                        tuple(old.req.output_ids),
                        tuple(new.req.output_ids),
                    )
                    before_forward = executor.forward_count
                    group._post = drop
                    try:
                        assert not old.life.try_install({0: 4, 1: 4})
                    finally:
                        group._post = post
                    assert len(dropped) == 1 and not old.life.can_decode()
                    assert (
                        old.fixture.request.delivery.snapshot()["retained_destinations"]
                        == 2
                    )
                    assert old.fixture.request.delivery.snapshot()["ack_tasks"] == 0
                    try:
                        dispatcher.begin([new.life, old.life])
                    except LifecycleError:
                        pass
                    else:
                        raise AssertionError("missing RESUMED admitted model execution")
                    assert executor.forward_count == before_forward
                    assert before_outputs == (
                        tuple(old.req.output_ids),
                        tuple(new.req.output_ids),
                    )
                    post(1, dropped[0])  # explicit delayed reply; no timeout extension
                    fault_evidence.update(
                        wait_all_without_forward=True,
                        delivery_ack_withheld=True,
                        delayed_resume_replayed=True,
                    )
                elif rank_fault == "install":
                    group = old.fixture.group
                    bank = group._banks[1]
                    install = bank.install
                    before_outputs = tuple(old.req.output_ids)

                    def fail_after_swap(*args, **kwargs):
                        install(*args, **kwargs)
                        fault_evidence["injected_after_actual_swap"] = True
                        raise RuntimeError(
                            "injected rank failure after actual bank swap"
                        )

                    bank.install = fail_after_swap
                    try:
                        try:
                            old.life.try_install({0: 4, 1: 4})
                        except RuntimeError as exc:
                            assert "after actual bank swap" in str(exc)
                        else:
                            raise AssertionError("partial installation did not fail")
                    finally:
                        bank.install = install
                    assert old.life.state == "aborted" and not old.life.can_decode()
                    assert (
                        tuple(old.req.output_ids) == before_outputs
                        and old.life.committed_tokens == 4
                    )
                    assert old.fixture.request.delivery.snapshot()["ack_tasks"] == 0
                    try:
                        dispatcher.begin([old.life, new.life])
                    except LifecycleError:
                        pass
                    else:
                        raise AssertionError("partially swapped request admitted")
                    step(("new",))
                    assert new.life.committed_tokens == 2
                    fault_evidence.update(
                        failed_request_not_admitted=True,
                        failed_request_output_unchanged=True,
                        unrelated_req_committed=True,
                        delivery_ack_withheld=True,
                    )
                    return {
                        "status": "passed",
                        "fault_evidence": fault_evidence,
                        "rank_runtime_bound_to_model_banks": True,
                        "real_req_schedule_batch_result_processor": True,
                        "independent_real_draft": draft_provider is not None,
                        "request_local_refresh_driver": driver is not None,
                        "batch_sizes": sizes,
                        "attention_checks": len(errors),
                        "max_attention_error": max(errors),
                        "committed_d_tokens": {
                            name: r.life.committed_tokens for name, r in records.items()
                        },
                        "production_scheduler_gpu_rdma_validated": False,
                    }
                if driver is None:
                    assert old.life.try_install({0: 4, 1: 4})
                else:
                    assert driver.progress().installed == ("old",)
                assert epoch.target_tokens == 4
                step(("old", "new"))  # reorder membership, unlike previous forward
                step(("new", "old"), cancel_after_forward="new")
                assert (old.life.committed_tokens, new.life.committed_tokens) == (6, 2)
                while old.life.committed_tokens < 8:
                    step(("old",))
                prefix = old.life.snapshot()
                if driver is not None:
                    # Deliberately omitted progress during 6->8 to exercise a
                    # missed prefetch window, not a replacement of a late query.
                    launch = driver.progress().launched[0]
                    assert launch.query_source == "committed"
                    task = launch.task
                else:
                    task = old.life.launch_refresh(
                        query_positions=(len(prefix.tokens) - 1,),
                        clients=clients,
                        pack_source=None if wire_delivery else old.fixture.pack_source,
                        timeout_seconds=30,
                    )
                tasks.append(task)
                await task
                if driver is None:
                    assert old.life.try_install({0: 8, 1: 8})
                else:
                    assert driver.progress().installed == ("old",)
                step(("old",))
            delivery_evidence = None
            if wire_delivery:
                deliveries = [
                    d
                    for store in old.fixture.stores.values()
                    for d in store.entries[old.fixture.manifest.key].deliveries.values()
                ]
                assert len(deliveries) == 4  # two boundaries, two V shards
                assert all(d.state.value == "released" for d in deliveries)
                assert old.fixture.packed_specs == new.fixture.packed_specs == []
                assert old.fixture.request.delivery.registry.snapshot() == {}
                transferred = sum(
                    store.transfer_engine.total_put_bytes
                    for store in old.fixture.stores.values()
                )
                assert transferred == sum(d.sparse_manifest.nbytes for d in deliveries)
                delivery_evidence = {
                    "http_shard_deliveries": len(deliveries),
                    "selected_kv_bytes": transferred,
                    "no_local_pack_callback": True,
                    "acknowledged_after_all_rank_install": True,
                    "receive_budget_restored": True,
                    "transport": "fake in-process byte copy; real localhost HTTP control",
                }
            # Force allocator-selected reuse, not a hand-edited free list.
            # Held capacity has no model users and is returned immediately.
            while reqpool.free_slots:
                holder = PrivatePoolAllocator(
                    reqpool, runner.token_to_kv_pool_allocator
                )
                held_slots.append((holder, holder.alloc_request()))
            held_rows.extend(
                pressure_allocator.alloc_kv(
                    runner.token_to_kv_pool_allocator.available_size()
                )
            )
            await retire(new)
            assert new.life not in executor._storage
            assert torch.count_nonzero(reqpool.req_to_token[new.slot]).item() == 0
            assert runner.token_to_kv_pool_allocator.available_size() == len(new.rows)
            third = create("third", (1, 12, 25, 8))
            assert third.slot == new.slot
            assert set(third.rows).issubset(new.rows)
            assert third.life.incarnation != new.life.incarnation
            reuse_evidence.update(
                retired_before_next_admission=True,
                allocator_reused_slot=True,
                allocator_reused_kv_rows=True,
            )
            if scheduled_results:
                mapping = reqpool.req_to_token[third.slot].clone()
                kv = [
                    (
                        pool.get_key_buffer(l)[third.rows].clone(),
                        pool.get_value_buffer(l)[third.rows].clone(),
                    )
                    for l in range(layers)
                ]
                outputs = tuple(third.req.output_ids), tuple(old.req.output_ids)
                available = runner.token_to_kv_pool_allocator.available_size()
                await retire(new)  # replayed cleanup must not clear the reused row
                try:
                    new.stale_result()
                except LifecycleError:
                    pass
                else:
                    raise AssertionError("old result accepted after slot reuse")
                finally:
                    new.stale_result = None
                assert outputs == (
                    tuple(third.req.output_ids),
                    tuple(old.req.output_ids),
                )
                torch.testing.assert_close(reqpool.req_to_token[third.slot], mapping)
                for l, (k, v) in enumerate(kv):
                    torch.testing.assert_close(
                        pool.get_key_buffer(l)[third.rows], k, equal_nan=True
                    )
                    torch.testing.assert_close(
                        pool.get_value_buffer(l)[third.rows], v, equal_nan=True
                    )
                assert runner.token_to_kv_pool_allocator.available_size() == available
                assert executor._storage[third.life] == third.slot
                reuse_evidence["stale_result_refused_without_mutation"] = True
            release_pressure()
            third.life.admit(third.fixture.request)
            outputs_before = old.life.outputs, third.life.outputs
            inject_failure = True
            try:
                step(("third", "old"))
            except RuntimeError as exc:
                assert "actual batched attention failure" in str(exc)
            else:
                raise AssertionError("batch failure was not exercised")
            assert (old.life.outputs, third.life.outputs) == outputs_before
            assert old.life.state == third.life.state == "aborted"
            assert not arbiter.busy and backend.consumer._bound is None
            assert old.life.committed_tokens == 9
            if draft_provider is None:
                assert new.fixture.request.pipeline.provider.calls == []
            if scheduled_results:
                ended = create("length-limit", (1, 14, 23))
                ended.life.admit(ended.fixture.request)
                ended.req.sampling_params.max_new_tokens = 2  # P's token + one D token
                inject_failure = False
                step(("length-limit",))
                assert ended.life.state == "finished" and ended.req.finished()
                assert ended.life.committed_tokens == 1
            return {
                "status": "passed",
                "fault_evidence": fault_evidence,
                "resource_reuse_evidence": reuse_evidence,
                "batch_sizes": sizes,
                "committed_d_tokens": {
                    name: r.life.committed_tokens for name, r in records.items()
                },
                "attention_checks": len(errors),
                "max_attention_error": max(errors),
                "wait_all_at_boundary": True,
                "reordered_results_and_membership": True,
                "cancelled_member_output_discarded": True,
                "real_batch_failure_commits_nothing": True,
                "reusable_batch_executor": True,
                "real_req_schedule_batch_result_processor": scheduled_results,
                "rank_runtime_bound_to_model_banks": rank_runtime,
                "real_req_retraction_and_length_limit": scheduled_results,
                "independent_real_draft": draft_provider is not None,
                "request_local_refresh_driver": driver is not None,
                "sparse_delivery_evidence": delivery_evidence,
                "production_scheduler_gpu_rdma_validated": False,
            }
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for hook in hooks:
                hook.remove()
            runner.attn_backend = native
            for r in records.values():
                await retire(r)
            release_pressure()
            if driver is not None:
                await driver.close()
            assert not arbiter.busy and not executor._storage
            assert all(b.snapshot()["used_staging_bytes"] == 0 for b in budgets)
            assert (
                len(reqpool.free_slots),
                runner.token_to_kv_pool_allocator.available_size(),
            ) == before
            fault_evidence["cleanup_verified"] = True

    return asyncio.run(run())
