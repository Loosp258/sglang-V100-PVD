"""Native V100S acceptance for bounded CAGRA build, extend, search, dispose."""

# ruff: noqa: I001 -- Load cuVS before Torch in this CloudLab wheel.
from cuvs.neighbors import cagra as _cagra  # noqa: F401

import torch

from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend


def main():
    backend = CagraIndexBackend(
        device="cuda:0",
        native_bytes_per_index=536870912,
        global_native_cap_bytes=671088640,
        graph_degree=8,
        intermediate_degree=16,
        itopk_size=64,
    )
    assert backend.supports_extend
    source = torch.randn(
        (2304, 128), device="cuda:0", dtype=torch.float32
    ).contiguous()
    first = source[:1024].contiguous()
    middle = source[1024:1664].contiguous()
    tail = source[1664:].contiguous()
    original = backend.build(first, vector_space="prompt", metric="ip")
    intermediate = backend.extend(original, middle)
    assert intermediate.count == 1664
    extended = backend.extend(intermediate, tail)
    assert extended.count == 2304
    rows, scores = backend.search(extended, source[100:101].contiguous(), top_k=4)
    assert rows.shape == scores.shape == (1, 4)
    backend.dispose(extended)
    assert backend.runtime.global_allocated_bytes() == 0
    print(
        {
            "supports_extend": backend.supports_extend,
            "rows": extended.count,
            "extension_counts": [intermediate.count, extended.count],
            "top4": rows[0].tolist(),
            "native_retained_bytes_after_dispose": 0,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
