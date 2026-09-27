"""A private PVD forward must not replace a simultaneous formal context."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
    has_forward_context,
)


def test_forward_context_isolated_between_simultaneous_threads():
    barrier = Barrier(2)
    formal_backend = object()
    private_backend = object()

    def run(backend, other_backend):
        assert not has_forward_context()
        with forward_context(ForwardContext(attn_backend=backend)):
            barrier.wait(timeout=5)
            assert get_attn_backend() is backend
            assert get_attn_backend() is not other_backend
            with forward_context(ForwardContext(attn_backend=other_backend)):
                assert get_attn_backend() is other_backend
            assert get_attn_backend() is backend
            barrier.wait(timeout=5)
        assert not has_forward_context()

    with ThreadPoolExecutor(max_workers=2) as pool:
        formal = pool.submit(run, formal_backend, private_backend)
        private = pool.submit(run, private_backend, formal_backend)
        formal.result(timeout=10)
        private.result(timeout=10)

    assert not has_forward_context()
