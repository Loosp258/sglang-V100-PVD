"""Measure the pure-Python cost of a Qwen-shaped PVD fan-in manifest.

This is a CPU planning probe, not a GPU, RDMA or inference benchmark. It uses
the real protocol objects with 28 layers, 56 K/V components and V TP2 -> D
TP1, while keeping each component tiny so tensor allocation is not measured.
"""

import argparse
import copy
import json
import time
from dataclasses import replace

import torch
from sglang.srt.disaggregation.pvd.full_kv_fanin import FullKVFanInReceiver
from sglang.srt.disaggregation.pvd.full_kv_fanin_plan import validate_fanin_plan
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey
from sglang.srt.disaggregation.pvd.transfer_engine import FakeTransferEngine
from sglang.srt.disaggregation.pvd.transfer_lifecycle import ResourceGuard
from test_pvd_fanin_mapping import layout


def qwen_shaped_layout(tp):
    base = layout(tp, page_size=1)
    extra = copy.deepcopy(base.extra)
    for key in (
        "component_bytes_per_token",
        "component_dtypes",
        "component_token_shapes",
    ):
        extra[key] *= 14  # Two layers in the fixture -> 28 layers, K and V.
    extra["component_count"] = 56
    return replace(base, num_layers=28, extra=extra)


def measure(tokens):
    storage, compute = qwen_shaped_layout(2), qwen_shaped_layout(1)
    expected_slices = tokens * 56 * 2
    engine = FakeTransferEngine()
    size = tokens * sum(compute.extra["component_bytes_per_token"])
    registration = engine.register_memory(
        torch.empty(size, dtype=torch.uint8),
        endpoint="D",
        rank=0,
        rail="mlx5_0",
        metadata={"pvd_receiver_epoch": "decode", "pvd_generation": "generation"},
    )
    guard = ResourceGuard(registration, lambda: engine.release_memory(registration))
    start = time.perf_counter()
    receiver = FullKVFanInReceiver(
        key=KVEntryKey("model", "request", "transfer"),
        delivery_id="delivery",
        registration=registration,
        guard=guard,
        storage=storage,
        compute=compute,
        token_count=tokens,
        max_slices=expected_slices,
    )
    built = time.perf_counter()
    manifest, identities = receiver.publish_for_sources({"0": "v0", "1": "v1"})
    published = time.perf_counter()
    encoded = json.dumps(manifest, separators=(",", ":")).encode()
    serialized = time.perf_counter()
    validate_fanin_plan(manifest, max_slices=expected_slices)
    validated = time.perf_counter()
    # No RPC was sent. Synthetic NOT_SUBMITTED proofs close the fake-MR pin.
    for rank, identity in identities.items():
        receiver.observe(
            {
                "protocol": manifest["protocol"],
                "plan_fingerprint": manifest["plan_fingerprint"],
                "source_rank": rank,
                "identity": identity.to_dict(),
                "fenced": True,
                "transport_state": "not_submitted",
                "transferred_bytes": 0,
            }
        )
    receiver.close()
    guard.request_release()
    return {
        "tokens": tokens,
        "slices": expected_slices,
        "manifest_bytes": len(encoded),
        "build_seconds": built - start,
        "publish_seconds": published - built,
        "serialize_seconds": serialized - published,
        "validate_seconds": validated - serialized,
        "total_seconds": validated - start,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, choices=(236, 517, 1010, 2044), required=True
    )
    args = parser.parse_args()
    print(json.dumps(measure(args.tokens), indent=2))
