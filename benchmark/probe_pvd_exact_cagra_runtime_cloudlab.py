"""CloudLab V100S smoke for the opt-in serving exact-head CAGRA path."""

import torch

from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend


def main():
    torch.cuda.set_device(0)
    backend = CagraIndexBackend(
        device="cuda:0", native_bytes_per_index=536870912,
        global_native_cap_bytes=671088640, graph_degree=16,
        intermediate_degree=16, itopk_size=256, exact_head_groups=4,
    )
    generator = torch.Generator(device="cuda:0").manual_seed(20260929)
    initial = torch.randn((4 * 512, 128), device="cuda:0", generator=generator)
    tail = torch.randn((4 * 128, 128), device="cuda:0", generator=generator)
    index = backend.build(initial, vector_space="qwen25-7b-pvd", metric="ip")
    assert index.count == 2048
    index = backend.extend(index, tail)
    assert index.count == 2560
    found, scores = backend.search(index, initial[:2].contiguous(), top_k=10)
    assert found.shape == scores.shape == (2, 10)
    backend.dispose(index)
    assert backend.runtime.global_allocated_bytes() == 0
    print("exact per-head CAGRA build, extend, search and dispose passed")


if __name__ == "__main__":
    main()
