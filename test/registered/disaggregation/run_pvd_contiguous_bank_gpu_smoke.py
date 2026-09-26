"""Standalone real-CUDA check for D's opt-in contiguous sparse-bank copy.

Compares the copied banks with the unchanged per-group path, including a
refresh boundary. No model weights, RDMA or running service are touched.
"""

import argparse
import json

import torch
from sglang.srt.disaggregation.pvd.cuda_working_set import CUDASparseWorkingSet
from sglang.srt.disaggregation.pvd.sparse_payload import SparseKVPayload, SparseKVSpec
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransferBudget,
)


def _source(device, dtype, tokens, boundary):
    head_dim = 8
    groups = ((0, 0), (0, 1))
    values_per_group = 2 * len(tokens) * head_dim
    values = torch.arange(
        values_per_group * len(groups), device=device, dtype=torch.float32
    ).to(dtype)
    # The first full-Prompt importer supplies this four-dimensional shape.
    # A flat-only fixture misses element offsets sliced as a group dimension.
    backing = values.contiguous().view(len(groups), 2, len(tokens), head_dim)
    payloads = []
    for index, (layer, head) in enumerate(groups):
        spec = SparseKVSpec(
            "req",
            "inc",
            f"op-{boundary}",
            boundary,
            "entry",
            "index",
            "mapping",
            "layout",
            layer,
            head,
            tokens,
        )
        tensor = backing[index]
        payloads.append(SparseKVPayload(spec, tensor))
    return tuple(payloads), ResourceGuard(backing, lambda: None)


def _bank(device, dtype, contiguous):
    budget = TransferBudget(4096, 2)
    bank = CUDASparseWorkingSet(
        device=device,
        dtype=dtype,
        budget=budget,
        contiguous_stage_copy=contiguous,
        request_id="req",
        incarnation="inc",
        entry_transfer_id="entry",
        layout_fingerprint="layout",
        expected_groups=((0, 0), (0, 1)),
        prompt_tokens=4,
        head_dim=8,
        max_union_tokens=2,
    )
    return bank, budget


def check(device, dtype):
    observed = {}
    for contiguous in (False, True):
        bank, budget = _bank(device, dtype, contiguous)
        try:
            for boundary, tokens in ((0, (0, 1, 2, 3)), (4, (3, 1))):
                payloads, guard = _source(device, dtype, tokens, boundary)
                bank.stage(payloads, source_guard=guard)
                guard.request_release()
                if guard.value is not None:
                    raise AssertionError("completed staging retained its source")
                bank.install(boundary)
                with bank.read() as groups:
                    observed[(contiguous, boundary)] = {
                        key: value.detach().clone().cpu()
                        for key, (_, value) in groups.items()
                    }
                    if contiguous:
                        addresses = {
                            value.untyped_storage().data_ptr()
                            for _, value in groups.values()
                        }
                        if len(addresses) != 1:
                            raise AssertionError("contiguous bank has split storage")
        finally:
            bank.close()
        if budget.snapshot()["used_staging_bytes"] != 0:
            raise AssertionError("D bank budget did not refund after close")
    for boundary in (0, 4):
        for key in ((0, 0), (0, 1)):
            if not torch.equal(
                observed[(False, boundary)][key],
                observed[(True, boundary)][key],
            ):
                raise AssertionError(
                    f"bank bytes disagree at {dtype}, {boundary}, {key}"
                )
    return {"dtype": str(dtype), "boundaries": [0, 4], "groups": 2}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        parser.error("a real indexed CUDA device is required")
    results = [
        check(device, dtype) for dtype in (torch.float16, torch.bfloat16, torch.float32)
    ]
    print(json.dumps({"device": str(device), "checks": results}, indent=2))


if __name__ == "__main__":
    main()
