"""Three-node native V-to-D fan-in, GPU bank install and Delivery ACK gate.

Requires a retained synthetic Entry from run_pvd_native_upload_index_gpu.py.
This exercises full-Prompt bootstrap and one sparse refresh with synthetic KV;
no target-model Decode forward, real Q quality or latency overlap is claimed.
"""

import argparse
import asyncio
import copy
import json
import sys
import uuid
from dataclasses import replace


async def run(args):
    import torch
    from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
    from sglang.srt.disaggregation.pvd.control_server import HttpShardClient
    from sglang.srt.disaggregation.pvd.cuda_runtime_group import CUDARuntimeInstallGroup
    from sglang.srt.disaggregation.pvd.cuda_sparse_delivery import (
        CUDAReceiveRoute,
        CUDASparseFanInDelivery,
    )
    from sglang.srt.disaggregation.pvd.cuda_sparse_receiver import (
        CUDASparseReceiveRegistry,
    )
    from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
    from sglang.srt.disaggregation.pvd.kv_packer import (
        PVD_TENSOR_LAYOUT,
        describe_kv_layout,
    )
    from sglang.srt.disaggregation.pvd.mooncake_engine import (
        MooncakePVDTransferEngine,
    )
    from sglang.srt.disaggregation.pvd.prompt_index import SearchRequestIdentity
    from sglang.srt.disaggregation.pvd.protocol import KVEntryKey, KVLayoutSignature
    from sglang.srt.disaggregation.pvd.search_client import PVDShardSearchClient
    from sglang.srt.disaggregation.pvd.search_routing import RoutedShardSearchClient
    from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVSpec
    from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

    if (
        not torch.cuda.is_available()
        or args.expected_gpu not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("requested D GPU is not the expected device")
    key = KVEntryKey("synthetic-pv-gate", "synthetic-req", args.transfer_id)
    generator = torch.Generator().manual_seed(20260924)
    shape = (1024, 2, 32)
    keys = [
        torch.nn.functional.normalize(
            torch.randn(shape, generator=generator), dim=-1
        ).half()
        for _ in range(2)
    ]
    values = [torch.randn(shape, generator=generator).half() for _ in range(2)]

    class Pool:
        start_layer = 0
        end_layer = 2
        k_buffer = keys
        v_buffer = values

    extra = copy.deepcopy(describe_kv_layout(Pool()))
    extra["component_token_shapes"] = [
        [1, item[1]] for item in extra["component_token_shapes"]
    ]
    extra["component_bytes_per_token"] = [
        item // 2 for item in extra["component_bytes_per_token"]
    ]
    storage = KVLayoutSignature(
        model_id="synthetic-target",
        model_revision="native-pv-gate-v1",
        kv_dtype=extra["component_dtypes"][0],
        page_size=4,
        num_layers=2,
        total_kv_heads=2,
        kv_heads_per_rank=1,
        head_dim=32,
        tp_size=2,
        pp_size=1,
        tensor_layout=PVD_TENSOR_LAYOUT,
        extra=extra,
    )
    if storage.fingerprint != args.layout_fingerprint:
        raise RuntimeError("retained P Entry layout does not match this D gate")
    compute = replace(
        storage,
        tp_size=1,
        kv_heads_per_rank=2,
        extra={
            **extra,
            "component_token_shapes": [
                [2, item[1]] for item in extra["component_token_shapes"]
            ],
            "component_bytes_per_token": [
                item * 2 for item in extra["component_bytes_per_token"]
            ],
        },
    )
    engine = MooncakePVDTransferEngine(
        hostname=args.decode_host,
        gpu_id=0,
        rail=args.rail,
        budget=TransferBudget(8 << 20, 8),
    )
    receive_budget = TransferBudget(8 << 20, 8)
    aggregate_budget = TransferBudget(8 << 20, 2)
    bank_budget = TransferBudget(8 << 20, 2)
    registry = CUDASparseReceiveRegistry(
        engine,
        receive_budget,
        receiver_epoch="synthetic-d-install:" + uuid.uuid4().hex,
        device="cuda:0",
    )
    clients = {
        rank: HttpShardClient(
            rank, f"{args.vector_base_url}:{args.shard_port_base + rank}"
        )
        for rank in (0, 1)
    }
    searches = {
        rank: PVDShardSearchClient(
            f"{args.vector_base_url}:{args.shard_port_base + rank}"
        )
        for rank in (0, 1)
    }
    routing = RoutedShardSearchClient(
        storage_layout=storage,
        compute_layout=compute,
        compute_rank=0,
        entry_transfer_id=key.transfer_id,
        prompt_tokens=1024,
        vector_space="synthetic-target",
        metric="ip",
        clients=searches,
    )
    bank = CUDASparseWorkingSet(
        device="cuda:0",
        dtype=torch.float16,
        budget=bank_budget,
        request_id=key.req_id,
        incarnation="synthetic-d-install",
        entry_transfer_id=key.transfer_id,
        layout_fingerprint=compute.fingerprint,
        expected_groups=tuple(routing.groups),
        prompt_tokens=1024,
        head_dim=32,
        max_union_tokens=1024,
    )
    group = CUDARuntimeInstallGroup(
        {0: bank},
        interval=4,
        lead_tokens=1,
        peer_epochs={0: "synthetic-d-peer"},
        timeout_seconds=30,
        max_pending_events=16,
        max_pending_bytes=131072,
    )
    coordinator = PVDCoordinatorClient(args.coordinator_url)
    delivery = None
    report = {
        "schema": "pvd-native-sparse-install-gpu-v1",
        "status": "failed",
        "transfer_id": key.transfer_id,
        "rail": args.rail,
        "installed_boundaries": [],
    }
    try:
        routes, versions, selected = {}, {}, {}
        for rank in (0, 1):
            health = await clients[rank].health()
            if (
                health["rank"] != rank
                or health["rail"] != args.rail
                or health["sparse_packing_mode"]
                != "cuda_synchronous_experimental"
                or not health["ready"]
            ):
                raise RuntimeError(f"V rank {rank} is not ready for sparse fan-in")
            routes[rank] = CUDAReceiveRoute(
                clients[rank], health["worker_epoch"], "synthetic-d-install", args.rail
            )
            answer = await searches[rank].search(
                # Use a synthetic stored K row only to obtain one real index
                # version and a bounded logical selection for the refresh.
                SearchRequestIdentity(
                    "synthetic-target", "rope_applied", key.transfer_id, 0, rank
                ),
                queries=[keys[0][1, rank].float().tolist()],
                top_k=2,
                scope=routing.scope,
            )
            versions[rank] = (answer.index_version, answer.id_mapping_version)
            selected[rank] = tuple(answer.token_ids)
            if not selected[rank]:
                raise AssertionError("V search returned an empty sparse selection")
        delivery = CUDASparseFanInDelivery(
            group,
            registry,
            routing,
            key=key,
            routes=routes,
            aggregate_budget=aggregate_budget,
            poll_interval_seconds=0.05,
        )

        for decode_tokens in (0, 3):
            epoch = group.begin(decode_tokens)
            specs = tuple(
                SparseKVSpec(
                    request_id=key.req_id,
                    incarnation="synthetic-d-install",
                    operation_id=epoch.operation_id,
                    target_tokens=epoch.target_tokens,
                    entry_transfer_id=key.transfer_id,
                    index_version=versions[head][0],
                    id_mapping_version=versions[head][1],
                    layout_fingerprint=compute.fingerprint,
                    layer=layer,
                    kv_head=head,
                    token_ids=(
                        tuple(range(1024))
                        if decode_tokens == 0
                        else selected[head]
                    ),
                )
                for layer in range(2)
                for head in range(2)
            )
            receipt = await delivery.stage(epoch, 0, specs)
            delivery.require_installable(epoch)
            if not group.try_install(epoch, {0: epoch.target_tokens}):
                raise RuntimeError("D CUDA bank did not complete the install exchange")
            if receipt.epoch != epoch or not group.can_decode(epoch.target_tokens):
                raise AssertionError("D install receipt or resume gate is not current")
            delivery.installed(epoch)
            errors = await delivery.retry_acks(epoch)
            if errors:
                raise RuntimeError(f"V Delivery ACK/close failed: {errors}")
            with bank.read() as groups:
                for spec in specs:
                    installed_spec, actual = groups[spec.layer, spec.kv_head]
                    expected = torch.stack(
                        (
                            keys[spec.layer][list(spec.token_ids), spec.kv_head],
                            values[spec.layer][list(spec.token_ids), spec.kv_head],
                        )
                    ).to("cuda:0")
                    if installed_spec != spec:
                        raise AssertionError("D bank installed a foreign sparse spec")
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            report["installed_boundaries"].append(epoch.target_tokens)
        if await delivery.close() or registry.snapshot():
            raise RuntimeError("D retained a sparse destination after ACK")
        group.close()
        if any(
            budget.snapshot()["used_staging_bytes"] != 0
            for budget in (receive_budget, aggregate_budget, bank_budget)
        ):
            raise AssertionError("D sparse receive/aggregate/bank budget did not refund")
        await coordinator.release_entry(key)
        report["released_entry"] = True
        report["status"] = "passed"
    finally:
        if delivery is not None:
            report["cleanup_errors"] = await delivery.close()
        else:
            report["cleanup_errors"] = await registry.close()
        if report["cleanup_errors"]:
            report["status"] = "failed"
        for client in clients.values():
            await client.close()
        for client in searches.values():
            await client.close()
        await routing.close()
        await coordinator.close()
    return report


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
    report = asyncio.run(run(parser.parse_args(argv)))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
