"""Real fake-engine bytes and controlled native-handle races, CPU only."""

import copy
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import test_pvd_fanin_lifecycle as lifecycle
import torch
from sglang.srt.disaggregation.pvd.full_kv_fanin_plan import (
    plan_fingerprint,
    validate_fanin_plan,
)
from sglang.srt.disaggregation.pvd.full_kv_fanin_writer import FullKVFanInWriter
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey, ProtocolValidationError
from sglang.srt.disaggregation.pvd.transfer_engine import (
    FakeTransferEngine,
    MemorySlice,
    TransferHandle,
    TransferStatus,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransportState,
)


@pytest.fixture
def case():
    yield from lifecycle.case.__wrapped__()


@contextmanager
def writer(c, rank=0, *, engine=None, **changes):
    engine = engine or c.engine
    raw = (torch.arange(c.size // 2, dtype=torch.int64) + rank * 29).to(torch.uint8)
    # Entry allocation begins inside a larger registered V pool.
    data = torch.cat(
        (torch.zeros(11, dtype=torch.uint8), raw, torch.zeros(9, dtype=torch.uint8))
    )
    registration = engine.register_memory(
        data, endpoint=f"V{rank}", rank=rank, rail="mlx5_7"
    )
    releases = []

    def release():
        releases.append(rank)
        engine.release_memory(registration)

    guard = ResourceGuard(object(), release)  # Store's allocation guard.
    args = dict(
        engine=engine,
        source=MemorySlice(registration, 11, len(raw)),
        source_guard=guard,
        source_key=c.key,
        source_layout=c.storage,
        source_rank=rank,
        sender_epoch=f"V-{rank}",
        max_slices=64,
        max_inflight=2,
    )
    args.update(changes)
    try:
        result = FullKVFanInWriter(c.manifest, **args)
        guard.request_release()
        yield NS(**locals())
    finally:
        # Native operations here are CPU doubles; this is test-only disposal.
        engine.release_memory(registration)


class DelayedEngine(FakeTransferEngine):
    def __init__(self):
        super().__init__()
        self.submitted = []

    def submit_put(self, local, remote, *, remote_offset=0):
        handle = TransferHandle(
            uuid.uuid4().hex, transport_state=TransportState.IN_FLIGHT
        )
        self.submitted.append((handle, local, remote, remote_offset))
        return handle

    def complete(self, handle, *, success=True):
        _, local, remote, offset = next(r for r in self.submitted if r[0] is handle)
        if success:
            done = super().submit_put(local, remote, remote_offset=offset)
            handle.transferred_bytes = done.transferred_bytes
            handle.status = done.status
            handle.transport_state = done.transport_state
        else:
            handle.status = TransferStatus.FAILED
            handle.transport_state = TransportState.TERMINAL_FAILED


def test_two_writers_reconstruct_bytes_and_feed_receiver_proofs(case):
    c = case
    c.manifest = c.receiver.publish()
    with writer(c, 0) as a, writer(c, 1) as b:
        c.receiver.adopt({0: a.result.identity, 1: b.result.identity})
        c.guard.request_release()
        expected = torch.zeros_like(c.registration.buffer)
        for w in (b, a):
            result = w.result.start()
            for _ in range(20):
                if result["fenced"]:
                    break
                result = w.result.poll()
            assert result["transport_state"] == "terminal_success" and result["fenced"]
            c.receiver.observe(result)
            assert c.receiver.ready is (w is a)
            for part in c.manifest["writers"][str(w.rank)]:
                lo, ro, n = part["local_offset"], part["remote_offset"], part["length"]
                expected[ro : ro + n] = w.raw[lo : lo + n]
            count = c.engine.total_put_bytes
            assert w.result.start() == result
            assert c.engine.total_put_bytes == count and w.releases == [w.rank]
        assert torch.equal(c.registration.buffer, expected)
        c.receiver.close()
        assert c.released == ["MR released"]


def test_cancelled_handles_stay_pinned_until_native_completion(case):
    c = case
    c.manifest = c.receiver.publish()
    engine = DelayedEngine()
    with writer(c, engine=engine) as w:
        w.result.start()
        assert len(engine.submitted) == 2
        result = w.result.cancel()
        assert not result["fenced"] and result["transport_state"] == "draining"
        assert not w.releases
        assert all(h.status == TransferStatus.CANCELLED for h, *_ in engine.submitted)
        for h, *_ in engine.submitted:
            engine.complete(h)
        result = w.result.poll()
        assert result["fenced"] and result["transport_state"] == "terminal_failed"
        assert len(engine.submitted) == 2 and w.releases == [0]


def test_cancel_before_start_is_closed_not_submitted(case):
    c = case
    c.manifest = c.receiver.publish()
    with writer(c) as w:
        result = w.result.cancel()
        assert result["fenced"] and result["transport_state"] == "not_submitted"
        assert w.result.start() == result and c.engine.total_put_bytes == 0


def test_submit_exception_is_sticky_unknown_even_after_known_peer_drains(case):
    c = case
    c.manifest = c.receiver.publish()

    class Raises(DelayedEngine):
        def submit_put(self, *args, **kw):
            if self.submitted:
                raise RuntimeError("posted but no returned handle")
            return super().submit_put(*args, **kw)

    engine = Raises()
    with writer(c, engine=engine) as w:
        result = w.result.start()
        assert result["transport_state"] == "unknown" and not result["fenced"]
        engine.complete(engine.submitted[0][0])
        result = w.result.cancel()
        assert not result["fenced"] and not w.releases
        assert w.result._source is not None


def test_cancel_racing_submit_cannot_fence_the_unreturned_handle(case):
    c = case
    c.manifest = c.receiver.publish()
    entered, resume = threading.Event(), threading.Event()

    class Blocked(DelayedEngine):
        def submit_put(self, *args, **kw):
            entered.set()
            assert resume.wait(5)
            return super().submit_put(*args, **kw)

    engine = Blocked()
    with writer(c, engine=engine) as w, ThreadPoolExecutor(1) as pool:
        future = pool.submit(w.result.start)
        try:
            assert entered.wait(5)
            assert not w.result.cancel()["fenced"] and not w.releases
        finally:
            resume.set()
        assert not future.result()["fenced"]
        assert len(engine.submitted) == 1
        engine.complete(engine.submitted[0][0])
        assert w.result.poll()["fenced"] and w.releases == [0]


@pytest.mark.parametrize(
    "fault",
    [
        "ranges",
        "omit-writer",
        "extra-field",
        "float-rank",
        "bool-tokens",
        "unknown-protocol",
        "oversized",
    ],
)
def test_sender_recomputes_plan_even_when_hash_is_recomputed(case, fault):
    raw = case.receiver.publish()
    raw.pop("plan_fingerprint")
    if fault == "ranges":
        raw["writers"]["0"][0]["remote_offset"] += 1
    elif fault == "omit-writer":
        raw["writers"].pop("1")
    elif fault == "extra-field":
        raw["destination"]["ignored"] = True
    elif fault == "float-rank":
        raw["destination"]["rank"] = 0.0
    elif fault == "bool-tokens":
        raw["token_count"] = True
    elif fault == "unknown-protocol":
        raw["protocol"] = "legacy"
    else:
        raw["token_count"] = 10**9
    raw["plan_fingerprint"] = plan_fingerprint(raw)
    with pytest.raises(ProtocolValidationError):
        validate_fanin_plan(raw, max_slices=64)


@pytest.mark.parametrize("fault", ["key", "layout", "rank", "rail"])
def test_writer_refuses_foreign_entry_or_source(case, fault):
    c = case
    c.manifest = c.receiver.publish()
    if fault == "key":
        changes = {"source_key": KVEntryKey("model", "other", "upload")}
    elif fault == "layout":
        changes = {"source_layout": replace(c.storage, model_revision="other")}
    elif fault == "rank":
        changes = {"source_rank": 1}
    else:
        c.manifest["destination"]["rail"] = "mlx5_8"
        raw = copy.deepcopy(c.manifest)
        raw.pop("plan_fingerprint")
        c.manifest["plan_fingerprint"] = plan_fingerprint(raw)
        changes = {}
    with pytest.raises(ProtocolValidationError), writer(c, **changes):
        pytest.fail("invalid source admitted")


def test_inflight_bound_prevents_submitting_entire_prompt(case):
    c = case
    c.manifest = c.receiver.publish()
    engine = DelayedEngine()
    with writer(c, engine=engine) as w:
        w.result.start()
        for _ in range(5):
            w.result.poll()
        assert len(engine.submitted) == 2 and len(w.result._pending) == 2
        assert not w.releases
        engine.complete(engine.submitted[0][0])
        w.result.poll()
        assert len(engine.submitted) == 3 and len(w.result._pending) == 2


@pytest.mark.parametrize(
    "fault", ["base-exception", "duplicate-handle", "short-success", "poll-error"]
)
def test_ambiguous_native_results_never_manufacture_a_fence(case, fault):
    c = case
    c.manifest = c.receiver.publish()

    class Interrupted(BaseException):
        pass

    class BadEngine(DelayedEngine):
        def submit_put(self, *args, **kw):
            if fault == "base-exception":
                raise Interrupted()
            if fault == "duplicate-handle" and self.submitted:
                return self.submitted[0][0]
            h = super().submit_put(*args, **kw)
            if fault == "short-success":
                h.transport_state = TransportState.TERMINAL_SUCCESS
            return h

        def poll(self, handle):
            if fault == "poll-error":
                raise RuntimeError("cannot inspect native handle")
            return super().poll(handle)

    with writer(c, engine=BadEngine()) as w:
        if fault == "base-exception":
            with pytest.raises(Interrupted):
                w.result.start()
        else:
            w.result.start()
        result = w.result.cancel()
        assert result["transport_state"] == "unknown" and not result["fenced"]
        assert not w.releases and w.result._source is not None
