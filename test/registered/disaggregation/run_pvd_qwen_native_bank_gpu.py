"""Install real Qwen Prompt KV from two V ranks into one D CUDA bank.

Requires a retained Entry from run_pvd_qwen_native_upload_gpu.py. A real
target Q chooses the sparse refresh; the boundary-four count is a protocol
fixture, not four generated target tokens. No model attention is run here.
"""

import argparse
import asyncio
import copy
import sys
import threading
import uuid
from dataclasses import replace
from types import SimpleNamespace


def _validate(runner, args, *, checkpoint=False):
    if not checkpoint or type(runner.model).__name__ != "Qwen2ForCausalLM":
        raise ValueError("a real Qwen2.5 checkpoint is required")

    import torch
    from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
    from sglang.srt.disaggregation.pvd.control_server import HttpShardClient
    from sglang.srt.disaggregation.pvd.cuda_model_attention import (
        CUDAModelPools,
        make_cuda_sparse_backend,
    )
    from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
    from sglang.srt.disaggregation.pvd.cuda_sparse_attention import (
        CUDASparseAttentionWorkspace,
    )
    from sglang.srt.disaggregation.pvd.cuda_sparse_delivery import (
        CUDAReceiveRoute,
        CUDASparseFanInDelivery,
    )
    from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import (
        CUDASparseReceiveRegistry,
    )
    from sglang.srt.disaggregation.pvd.cuda_target_probe import CUDAQwen2TargetProbe
    from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
    from sglang.srt.disaggregation.pvd.draft_forward_adapter import (
        DraftForwardAdapter,
        PrivatePoolAllocator,
    )
    from sglang.srt.disaggregation.pvd.draft_runner_sglang import DraftForwardInputs
    from sglang.srt.disaggregation.pvd.kv_packer import (
        PVD_TENSOR_LAYOUT,
        describe_kv_layout,
    )
    from sglang.srt.disaggregation.pvd.mooncake_engine import (
        MooncakePVDTransferEngine,
    )
    from sglang.srt.disaggregation.pvd.prediction import (
        CommittedPrefix,
        DraftPrediction,
        ProbeConfig,
    )
    from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity
    from sglang.srt.disaggregation.pvd.protocol import (
        FirstTokenMetadata,
        KVEntryKey,
        KVLayoutSignature,
    )
    from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
    from sglang.srt.disaggregation.pvd.search_routing import RoutedShardSearchClient
    from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
        ResourceGuard,
        TransferBudget,
    )

    config = runner.model.config
    if (
        config.num_hidden_layers != 28
        or config.num_key_value_heads != 4
        or config.num_attention_heads != 28
        or runner.model_config.head_dim != 128
        or args.expected_gpu not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("this bounded bank gate expects Qwen2.5-7B on V100S")
    space = "qwen2.5-7b-real-target"
    key = KVEntryKey(space, "real-qwen-prompt", args.transfer_id)

    async def selected_first_token():
        coordinator = PVDCoordinatorClient(args.coordinator_url)
        try:
            reply = await coordinator.select([key])
            records = reply.get("results")
            if not isinstance(records, list) or len(records) != 1:
                raise RuntimeError("exactly one selected stored Qwen Entry required")
            entry = records[0].get("entry")
            if (
                not isinstance(entry, dict)
                or entry.get("state") != "stored"
                or KVEntryKey.from_dict(entry["manifest"]["key"]) != key
                or not isinstance(entry.get("first_token"), dict)
            ):
                raise RuntimeError("selected Entry lacks its own first-token metadata")
            raw = entry["first_token"].get("output_token_id")
            if type(raw) is not int or not 0 <= raw < config.vocab_size:
                raise RuntimeError("selected Entry first token is not a Qwen token")
            return FirstTokenMetadata.from_dict(entry["first_token"]).output_token_id
        finally:
            await coordinator.close()

    first_token_id = asyncio.run(selected_first_token())
    token_count = 1024
    generator = torch.Generator().manual_seed(20260924)
    tokens = tuple(torch.randint(3, 1000, (token_count,), generator=generator).tolist())
    allocator = PrivatePoolAllocator(
        runner.req_to_token_pool, runner.token_to_kv_pool_allocator
    )
    slot, rows, native_row = allocator.alloc_request(), [], []
    baseline_logits, baseline_generated = None, None
    try:
        rows = allocator.alloc_kv(token_count)
        allocator.write_mapping(slot, 0, rows)
        adapter = DraftForwardAdapter(
            runner,
            architecture="Qwen2ForCausalLM",
            attention_backend="torch_native",
            bytes_per_token=28 * 4 * 128 * 2 * 2,
            device="cuda:0",
        )
        adapter.forward(
            DraftForwardInputs(
                "extend",
                tokens,
                tuple(range(token_count)),
                (token_count,),
                (slot,),
                tuple(rows),
                (0,),
                (token_count,),
            )
        )
        torch.cuda.synchronize("cuda:0")
        pool = SimpleNamespace(
            start_layer=0,
            end_layer=28,
            k_buffer=[
                runner.token_to_kv_pool.get_key_buffer(layer)[rows].clone()
                for layer in range(28)
            ],
            v_buffer=[
                runner.token_to_kv_pool.get_value_buffer(layer)[rows].clone()
                for layer in range(28)
            ],
        )
        if args.model_forward:
            native_row = allocator.alloc_kv(1)
            allocator.write_mapping(slot, token_count, native_row)
            baseline_logits = adapter.forward(
                DraftForwardInputs(
                    "decode",
                    (first_token_id,),
                    (token_count,),
                    (token_count + 1,),
                    (slot,),
                    tuple(native_row),
                )
            ).detach().clone()
            torch.cuda.synchronize("cuda:0")
            baseline_generated = tuple(
                (
                    runner.token_to_kv_pool.get_key_buffer(layer)[native_row].clone(),
                    runner.token_to_kv_pool.get_value_buffer(layer)[native_row].clone(),
                )
                for layer in range(28)
            )
    finally:
        if rows:
            torch.cuda.synchronize("cuda:0")
            allocator.clear_mapping(slot)
            allocator.free_kv(native_row)
            allocator.free_kv(rows)
        allocator.free_request(slot)

    compute_extra = describe_kv_layout(pool)
    storage_extra = copy.deepcopy(compute_extra)
    storage_extra["component_token_shapes"] = [
        [2, shape[1]] for shape in compute_extra["component_token_shapes"]
    ]
    storage_extra["component_bytes_per_token"] = [
        value // 2 for value in compute_extra["component_bytes_per_token"]
    ]
    storage = KVLayoutSignature(
        model_id=space,
        model_revision="cloudlab-local-checkpoint",
        kv_dtype=compute_extra["component_dtypes"][0],
        page_size=4,
        num_layers=28,
        total_kv_heads=4,
        kv_heads_per_rank=2,
        head_dim=128,
        tp_size=2,
        pp_size=1,
        tensor_layout=PVD_TENSOR_LAYOUT,
        extra=storage_extra,
    )
    if storage.fingerprint != args.layout_fingerprint:
        raise RuntimeError("D model layout differs from retained P/V Entry")
    compute = replace(
        storage,
        tp_size=1,
        kv_heads_per_rank=4,
        extra=compute_extra,
    )

    probe_budget = TransferBudget(256 << 20, 1)
    lock = threading.Lock()
    probe = CUDAQwen2TargetProbe(
        runner,
        ProbeConfig(space, tuple(range(28)), head_start=0, head_count=28),
        device="cuda:0",
        execution_lock=lock,
        target_model_id=space,
        max_tokens=token_count + (8 if args.generated_refresh else 2),
        max_predict_tokens=1,
        transient_bytes_bound=64 << 20,
        budget=probe_budget,
    )
    prefix = CommittedPrefix("real-qwen-prompt", tokens, 0, "prompt")
    with probe.branch():
        queries = probe.capture(
            prefix,
            DraftPrediction(prefix.request_id, prefix.version, (first_token_id,)),
        )
        q = {
            (query.layer, head): [
                query.vectors[0, head * 7 + member].float().tolist()
                for member in range(7)
            ]
            for query in queries
            for head in range(4)
        }
        if any(
            query.positional_encoding != "rope_applied"
            or query.vector_space != space
            or query.positions != (token_count,)
            for query in queries
        ):
            raise AssertionError("D target probe produced foreign Q")
    if probe_budget.snapshot()["used_staging_bytes"] or lock.locked():
        raise AssertionError("D target probe retained resources")

    def verify_model_forward(group):
        """One real target-model Decode forward against the installed Prompt bank."""
        workspace_budget = TransferBudget(1 << 20, 1)
        output_budget = TransferBudget(1 << 20, 1)
        workspace = CUDASparseAttentionWorkspace(
            device="cuda:0",
            dtype=torch.float16,
            head_dim=128,
            chunk_tokens=64,
            budget=workspace_budget,
        )
        original_backend = runner.attn_backend
        decoder = PrivatePoolAllocator(
            runner.req_to_token_pool, runner.token_to_kv_pool_allocator
        )
        slot = decoder.alloc_request()
        row = []
        guard = backend = None
        try:
            row = decoder.alloc_kv(1)
            decoder.write_mapping(slot, token_count, row)

            def retire_rows():
                torch.cuda.synchronize("cuda:0")
                decoder.clear_mapping(slot)
                decoder.free_kv(row)
                decoder.free_request(slot)

            guard = ResourceGuard(
                CUDAModelPools(runner.req_to_token_pool, runner.token_to_kv_pool),
                retire_rows,
            )
            backend = make_cuda_sparse_backend(
                runner,
                workspace=workspace,
                execution_lock=lock,
                output_budget=output_budget,
                max_batch_size=1,
            )
            runner.attn_backend = backend
            sparse_adapter = DraftForwardAdapter(
                runner,
                architecture="Qwen2ForCausalLM",
                attention_backend="torch_native",
                bytes_per_token=28 * 4 * 128 * 2 * 2,
                device="cuda:0",
            )
            with group.model_forward(
                backend.consumer, slot=slot, decode_tokens=0, pool_owner=guard
            ):
                sparse_logits = sparse_adapter.forward(
                    DraftForwardInputs(
                        "decode",
                        (first_token_id,),
                        (token_count,),
                        (token_count + 1,),
                        (slot,),
                        tuple(row),
                    )
                ).detach().clone()
            torch.cuda.synchronize("cuda:0")
            kv_max_error = 0.0
            for layer, (native_k, native_v) in enumerate(baseline_generated):
                for actual, expected in (
                    (runner.token_to_kv_pool.get_key_buffer(layer)[row], native_k),
                    (runner.token_to_kv_pool.get_value_buffer(layer)[row], native_v),
                ):
                    delta = (actual.float() - expected.float()).abs()
                    if not torch.isfinite(delta).all():
                        raise AssertionError(
                            "Qwen generated K/V contain nonfinite error"
                        )
                    kv_max_error = max(kv_max_error, float(delta.max()))
            max_error = float((sparse_logits - baseline_logits).abs().max())
            same_top_token = bool(
                torch.equal(sparse_logits.argmax(-1), baseline_logits.argmax(-1))
            )
            if (
                not torch.isfinite(sparse_logits).all()
                or not same_top_token
                or max_error > 0.15
                or kv_max_error > 0.15
            ):
                raise AssertionError(
                    "sparse Qwen forward differs from dense baseline: "
                    f"logits={max_error}, generated_kv={kv_max_error}, "
                    f"same_top_token={same_top_token}"
                )
            guard.request_release()
            if guard.value is not None:
                raise AssertionError("sparse model pool rows were not retired")
            return {
                "max_logit_abs_error": max_error,
                "max_generated_kv_abs_error": kv_max_error,
                "same_top_token": True,
            }
        finally:
            runner.attn_backend = original_backend
            if guard is None or backend is None:
                # No native forward started; this row can be retired locally.
                if guard is not None:
                    guard.request_release()
                elif row:
                    decoder.clear_mapping(slot)
                    decoder.free_kv(row)
                if guard is None:
                    decoder.free_request(slot)
            if backend is None or backend.consumer.snapshot()["quarantine"] is None:
                workspace.close()
            if (
                workspace_budget.snapshot()["used_staging_bytes"]
                or output_budget.snapshot()["used_staging_bytes"]
            ):
                raise AssertionError("sparse attention budgets were not refunded")

    class GeneratedDecode:
        """Keep actual generated rows on D across a full-to-sparse refresh."""

        def __init__(self, group):
            self.group = group
            self.workspace_budget = TransferBudget(1 << 20, 1)
            self.output_budget = TransferBudget(1 << 20, 1)
            self.workspace = CUDASparseAttentionWorkspace(
                device="cuda:0",
                dtype=torch.float16,
                head_dim=128,
                chunk_tokens=64,
                budget=self.workspace_budget,
            )
            self.decoder = PrivatePoolAllocator(
                runner.req_to_token_pool, runner.token_to_kv_pool_allocator
            )
            self.slot = self.decoder.alloc_request()
            self.rows = []
            self.inputs = []
            self.original_backend = runner.attn_backend

            def retire_rows():
                torch.cuda.synchronize("cuda:0")
                self.decoder.clear_mapping(self.slot)
                self.decoder.free_kv(self.rows)
                self.decoder.free_request(self.slot)

            self.guard = ResourceGuard(
                CUDAModelPools(runner.req_to_token_pool, runner.token_to_kv_pool),
                retire_rows,
            )
            self.backend = make_cuda_sparse_backend(
                runner,
                workspace=self.workspace,
                execution_lock=lock,
                output_budget=self.output_budget,
                max_batch_size=1,
            )
            self.adapter = DraftForwardAdapter(
                runner,
                architecture="Qwen2ForCausalLM",
                attention_backend="torch_native",
                bytes_per_token=28 * 4 * 128 * 2 * 2,
                device="cuda:0",
            )
            self.closed = False

        def forward(self, count, input_token):
            if self.closed or count != len(self.inputs):
                raise AssertionError("generated Decode count is not committed")
            row = self.decoder.alloc_kv(1)
            self.rows.extend(row)
            self.decoder.write_mapping(self.slot, token_count + count, row)
            runner.attn_backend = self.backend
            try:
                with self.group.model_forward(
                    self.backend.consumer,
                    slot=self.slot,
                    decode_tokens=count,
                    pool_owner=self.guard,
                ):
                    logits = self.adapter.forward(
                        DraftForwardInputs(
                            "decode",
                            (input_token,),
                            (token_count + count,),
                            (token_count + count + 1,),
                            (self.slot,),
                            tuple(row),
                        )
                    )
                if not torch.isfinite(logits).all():
                    raise AssertionError("generated Decode logits are nonfinite")
                self.inputs.append(input_token)
                return int(logits.argmax(-1).item())
            finally:
                runner.attn_backend = self.original_backend

        def close(self):
            if self.closed:
                return
            runner.attn_backend = self.original_backend
            if self.backend.consumer.snapshot()["quarantine"] is not None:
                raise RuntimeError("unknown model completion retains generated rows")
            self.guard.request_release()
            if self.guard.value is not None:
                raise AssertionError("generated row owner was not retired")
            self.workspace.close()
            if any(
                budget.snapshot()["used_staging_bytes"]
                for budget in (self.workspace_budget, self.output_budget)
            ):
                raise AssertionError("generated Decode budgets were not refunded")
            self.closed = True

    async def install():
        engine = MooncakePVDTransferEngine(
            hostname=args.decode_host,
            gpu_id=0,
            rail=args.rail,
            budget=TransferBudget(256 << 20, 8),
        )
        receive_budget = TransferBudget(256 << 20, 8)
        aggregate_budget = TransferBudget(256 << 20, 2)
        bank_budget = TransferBudget(256 << 20, 2)
        registry = CUDASparseReceiveRegistry(
            engine,
            receive_budget,
            receiver_epoch="real-qwen-bank:" + uuid.uuid4().hex,
            device="cuda:0",
        )
        endpoints = {
            rank: f"{args.vector_base_url}:{args.shard_port_base + rank}"
            for rank in (0, 1)
        }
        controls = {
            rank: HttpShardClient(rank, url, timeout_seconds=180)
            for rank, url in endpoints.items()
        }
        searches = {
            rank: PVDShardSearchClient(url) for rank, url in endpoints.items()
        }
        routing = RoutedShardSearchClient(
            storage_layout=storage,
            compute_layout=compute,
            compute_rank=0,
            entry_transfer_id=key.transfer_id,
            prompt_tokens=token_count,
            vector_space=space,
            metric="ip",
            clients=searches,
        )
        bank = CUDASparseWorkingSet(
            device="cuda:0",
            dtype=torch.float16,
            budget=bank_budget,
            request_id=key.req_id,
            incarnation="real-qwen-bank",
            entry_transfer_id=key.transfer_id,
            layout_fingerprint=compute.fingerprint,
            expected_groups=tuple(routing.groups),
            prompt_tokens=token_count,
            head_dim=128,
            max_union_tokens=70,
        )
        group = CUDARuntimeInstallGroup(
            {0: bank},
            interval=4,
            lead_tokens=1,
            peer_epochs={0: "real-qwen-d-peer"},
            timeout_seconds=180,
            max_pending_events=16,
            max_pending_bytes=131072,
        )
        coordinator = PVDCoordinatorClient(args.coordinator_url)
        delivery = generated = None
        report = {
            "installed_boundaries": [],
            "groups_per_round": 112,
            "q_heads_per_kv_head": 7,
            "union_limit_per_group": 70,
            "p_first_token_id": first_token_id,
        }
        try:
            routes, results = {}, {}
            for rank in (0, 1):
                health = await controls[rank].health()
                if (
                    health["rank"] != rank
                    or health["rail"] != args.rail
                    or not health["ready"]
                    or health["sparse_packing_mode"]
                    != "cuda_synchronous_experimental"
                ):
                    raise RuntimeError(f"V rank {rank} cannot serve real sparse KV")
                routes[rank] = CUDAReceiveRoute(
                    controls[rank], health["worker_epoch"], "real-qwen-bank", args.rail
                )
            for layer in range(28):
                for head in range(4):
                    rank = head // 2
                    results[layer, head] = await searches[rank].search(
                        SearchRequestIdentity(
                            space, "rope_applied", key.transfer_id, layer, head
                        ),
                        queries=q[layer, head],
                        top_k=10,
                        scope=routing.scope,
                    )
                    if not 0 < len(results[layer, head].token_ids) <= 70:
                        raise AssertionError("GQA token union exceeded its bound")
            sizes = [len(result.token_ids) for result in results.values()]
            report["gqa_union_size_range"] = [min(sizes), max(sizes)]
            delivery = CUDASparseFanInDelivery(
                group,
                registry,
                routing,
                key=key,
                routes=routes,
                aggregate_budget=aggregate_budget,
                poll_interval_seconds=0.1,
            )
            for decode_tokens in (0, 3):
                if decode_tokens == 3 and args.generated_refresh:
                    if generated is None or len(generated.inputs) != 3:
                        raise AssertionError("three real Decode steps required")
                    official = CommittedPrefix(
                        key.req_id,
                        tokens + tuple(generated.inputs) + (next_token,),
                        3,
                        "actual-decode-three",
                    )
                    with probe.branch():
                        actual_queries = probe.capture_committed(
                            official, (token_count + 3,)
                        )
                    if any(
                        item.positions != (token_count + 3,)
                        or item.positional_encoding != "rope_applied"
                        for item in actual_queries
                    ):
                        raise AssertionError("refresh Q is not from actual prefix")
                    actual_q = {
                        (item.layer, head): [
                            item.vectors[0, head * 7 + member].float().tolist()
                            for member in range(7)
                        ]
                        for item in actual_queries
                        for head in range(4)
                    }
                    for layer in range(28):
                        for head in range(4):
                            results[layer, head] = await routing.search(
                                SearchRequestIdentity(
                                    space,
                                    "rope_applied",
                                    key.transfer_id,
                                    layer,
                                    head,
                                ),
                                queries=actual_q[layer, head],
                                top_k=10,
                                scope=routing.scope,
                            )
                    report["refresh_q_position"] = token_count + 3
                epoch = group.begin(decode_tokens)
                specs = tuple(
                    SparseKVSpec(
                        request_id=key.req_id,
                        incarnation="real-qwen-bank",
                        operation_id=epoch.operation_id,
                        target_tokens=epoch.target_tokens,
                        entry_transfer_id=key.transfer_id,
                        index_version=results[layer, head].index_version,
                        id_mapping_version=results[layer, head].id_mapping_version,
                        layout_fingerprint=compute.fingerprint,
                        layer=layer,
                        kv_head=head,
                        token_ids=(
                            tuple(range(token_count))
                            if decode_tokens == 0
                            else results[layer, head].token_ids
                        ),
                    )
                    for layer in range(28)
                    for head in range(4)
                )
                receipt = await delivery.stage(epoch, 0, specs)
                delivery.require_installable(epoch)
                if decode_tokens == 3 and args.generated_refresh:
                    next_token = generated.forward(3, next_token)
                if not group.try_install(epoch, {0: epoch.target_tokens}):
                    raise RuntimeError("D CUDA bank did not complete install")
                if receipt.epoch != epoch or not group.can_decode(epoch.target_tokens):
                    raise AssertionError("D installation lacks current RESUMED proof")
                delivery.installed(epoch)
                if errors := await delivery.retry_acks(epoch):
                    raise RuntimeError(f"V Delivery ACK failed: {errors}")
                with bank.read() as groups:
                    for spec in specs:
                        installed, actual = groups[spec.layer, spec.kv_head]
                        expected = torch.stack(
                            (
                                pool.k_buffer[spec.layer][
                                    list(spec.token_ids), spec.kv_head
                                ],
                                pool.v_buffer[spec.layer][
                                    list(spec.token_ids), spec.kv_head
                                ],
                            )
                        )
                        if installed != spec:
                            raise AssertionError("D bank installed a foreign spec")
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                report["installed_boundaries"].append(epoch.target_tokens)
                if decode_tokens == 0 and args.model_forward:
                    report["real_model_forward"] = verify_model_forward(group)
                if decode_tokens == 0 and args.generated_refresh:
                    generated = GeneratedDecode(group)
                    next_token = first_token_id
                    for count in range(3):
                        next_token = generated.forward(count, next_token)
                if decode_tokens == 3 and args.generated_refresh:
                    next_token = generated.forward(4, next_token)
                    report["generated_refresh"] = {
                        "actual_steps_before_refresh": 4,
                        "sparse_bank_consumed_at_count": 4,
                        "next_target_token": next_token,
                    }
            if generated is not None:
                generated.close()
            if await delivery.close() or registry.snapshot():
                raise RuntimeError("D retained sparse receive registrations")
            group.close()
            if any(
                budget.snapshot()["used_staging_bytes"]
                for budget in (receive_budget, aggregate_budget, bank_budget)
            ):
                raise AssertionError("D bank/aggregate/receive budget not refunded")
            await coordinator.release_entry(key)
            report["released_entry"] = True
            return report
        finally:
            if generated is not None:
                generated.close()
            if delivery is not None:
                errors = await delivery.close()
            else:
                errors = await registry.close()
            for client in controls.values():
                await client.close()
            for client in searches.values():
                await client.close()
            await routing.close()
            await coordinator.close()
            if errors:
                raise RuntimeError(f"D retained unsafe sparse resources: {errors}")

    return asyncio.run(install())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-host", required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--vector-base-url", required=True)
    parser.add_argument("--shard-port-base", type=int, required=True)
    parser.add_argument("--transfer-id", required=True)
    parser.add_argument("--layout-fingerprint", required=True)
    parser.add_argument("--rail", required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--model-forward", action="store_true")
    parser.add_argument("--generated-refresh", action="store_true")
    args, model_args = parser.parse_known_args(argv)
    if args.generated_refresh and not args.model_forward:
        parser.error("--generated-refresh requires --model-forward")
    from run_pvd_cuda_probe_smoke import main as run_model

    return run_model(
        model_args,
        validator=lambda runner, checkpoint=False: _validate(
            runner, args, checkpoint=checkpoint
        ),
        schema="pvd-qwen2.5-native-full-bank-v1",
    )


if __name__ == "__main__":
    sys.exit(main())
