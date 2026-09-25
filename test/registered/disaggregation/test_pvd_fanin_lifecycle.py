"""Shared real CPU MR guard, synthetic sender proofs; no network fence evidence."""

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch
from sglang.srt.disaggregation.pvd.full_kv_fanin import FullKVFanInReceiver
from sglang.srt.disaggregation.pvd.full_kv_fanin_plan import (
    FULL_KV_FANIN_PROTOCOL,
    RANK_PACKED_FULL_KV_FANIN_PROTOCOL,
    plan_fingerprint,
    validate_fanin_plan,
)
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    KVEntryKey,
    ProtocolValidationError,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.transfer_authorization import WriteAuthorization
from sglang.srt.disaggregation.pvd.transfer_engine import (
    FakeTransferEngine,
    MemorySlice,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransportState,
)
from test_pvd_fanin_mapping import layout


@pytest.fixture(params=[FULL_KV_FANIN_PROTOCOL, RANK_PACKED_FULL_KV_FANIN_PROTOCOL])
def case(request=None):
    engine, released = FakeTransferEngine(), []
    compute, storage = layout(1), layout(2)
    size = sum(compute.extra["component_bytes_per_token"]) * 3
    registration = engine.register_memory(
        torch.zeros(size, dtype=torch.uint8),
        endpoint="D",
        rank=0,
        rail="mlx5_7",
        metadata={
            "pvd_receiver_epoch": "decode-process",
            "pvd_generation": "allocation-1",
        },
    )

    def release():
        released.append("MR released")
        engine.release_memory(registration)

    guard = ResourceGuard(registration, release)
    key = KVEntryKey("model", "entry", "upload")
    receiver = FullKVFanInReceiver(
        key=key,
        delivery_id="delivery",
        registration=registration,
        guard=guard,
        storage=storage,
        compute=compute,
        token_count=3,
        max_slices=64,
        protocol=request.param if request is not None else FULL_KV_FANIN_PROTOCOL,
    )
    identities = {
        rank: WriteIdentity(
            protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
            sender_epoch=f"V-{rank}",
            receiver_epoch="decode-process",
            transfer_id=f"delivery:d0:v{rank}",
            region_id=registration.descriptor.region_id,
            generation="allocation-1",
            shard_rank=0,
            key=key,
        )
        for rank in (0, 1)
    }
    c = NS(**locals())
    yield c
    # Test-only disposal after a case deliberately withholds synthetic proof.
    # Do not imply production can recover an unconfirmed remote write this way.
    engine.release_memory(registration)


def publish(c):
    c.manifest = c.receiver.publish()
    c.receiver.adopt(c.identities)
    c.guard.request_release()
    assert c.guard.value is c.registration and not c.released


def reply(c, rank, **changes):
    result = dict(
        protocol=c.manifest["protocol"],
        plan_fingerprint=c.manifest["plan_fingerprint"],
        source_rank=rank,
        identity=c.identities[rank].to_dict(),
        fenced=True,
        transport_state="terminal_success",
        transferred_bytes=c.size // 2,
    )
    result.update(changes)
    return result


def test_shared_mr_survives_one_writer_and_waits_for_complete_success(case):
    c = case
    publish(c)
    assert c.receiver.observe(reply(c, 1))
    assert not c.receiver.ready
    with pytest.raises(ProtocolValidationError, match="all possible"):
        c.receiver.close()
    assert not c.released
    assert c.receiver.observe(reply(c, 0)) and c.receiver.ready
    assert not c.released, "completion must not dispose active local consumers"
    c.receiver.close()
    assert c.released == ["MR released"] and c.guard.value is None
    assert c.receiver._guard is None
    with pytest.raises(ProtocolValidationError, match="closed"):
        c.receiver.publish()


def test_protocol_and_writer_ranges_cannot_be_reinterpreted(case):
    wire = case.receiver.publish()
    plan = validate_fanin_plan(wire, max_slices=64)
    assert plan.protocol == wire["protocol"]
    altered = copy.deepcopy(wire)
    altered["protocol"] = (
        RANK_PACKED_FULL_KV_FANIN_PROTOCOL
        if wire["protocol"] == FULL_KV_FANIN_PROTOCOL
        else FULL_KV_FANIN_PROTOCOL
    )
    altered.pop("plan_fingerprint")
    altered["plan_fingerprint"] = plan_fingerprint(altered)
    with pytest.raises(ProtocolValidationError, match="ranges"):
        validate_fanin_plan(altered, max_slices=64)
    altered = copy.deepcopy(wire)
    altered["writers"]["0"][0]["remote_offset"] += 1
    altered.pop("plan_fingerprint")
    altered["plan_fingerprint"] = plan_fingerprint(altered)
    with pytest.raises(ProtocolValidationError, match="ranges"):
        validate_fanin_plan(altered, max_slices=64)


@pytest.mark.parametrize("state", ["in_flight", "draining", "unknown"])
def test_cancel_and_nonterminal_reply_never_release(case, state):
    c = case
    publish(c)
    c.receiver.observe(reply(c, 0))
    c.receiver.cancel()
    assert not c.receiver.observe(
        reply(c, 1, fenced=False, transport_state=state, transferred_bytes=0)
    )
    with pytest.raises(ProtocolValidationError, match="terminal transport"):
        c.receiver.observe(reply(c, 1, transport_state=state, transferred_bytes=0))
    with pytest.raises(ProtocolValidationError, match="all possible"):
        c.receiver.close()
    assert not c.released and not c.receiver.ready
    c.receiver.observe(
        reply(c, 1, transport_state="terminal_failed", transferred_bytes=0)
    )
    assert not c.receiver.ready
    c.receiver.close()
    assert len(c.released) == 1


def test_publish_failure_before_adoption_still_pins_all_writers(case):
    c = case
    c.manifest = c.receiver.publish()
    c.guard.request_release()
    c.receiver.cancel()
    with pytest.raises(ProtocolValidationError, match="all possible"):
        c.receiver.close()
    # Recovery must recover ALL identities, not assume failed RPC == no writes.
    c.receiver.adopt(c.identities)
    for rank in (0, 1):
        c.receiver.observe(
            reply(c, rank, transport_state="not_submitted", transferred_bytes=0)
        )
    c.receiver.close()
    assert len(c.released) == 1


def test_unpublished_cancellation_can_close_without_remote_proof(case):
    c = case
    c.guard.request_release()
    c.receiver.cancel()
    with pytest.raises(ProtocolValidationError, match="one-shot"):
        c.receiver.publish()
    c.receiver.close()
    assert len(c.released) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "foreign-source",
        "same-D-collapse",
        "region",
        "generation",
        "transfer",
        "entry",
        "receiver",
        "bool-rank",
    ],
)
def test_complete_identity_set_and_source_qualified_ids_are_required(case, fault):
    c = case
    c.manifest = c.receiver.publish()
    identities = dict(c.identities)
    if fault in ("missing", "same-D-collapse"):
        identities = {0: c.identities[0]}
    elif fault == "foreign-source":
        identities[2] = identities.pop(1)
    elif fault == "bool-rank":
        identities = {False: identities[0], 1: identities[1]}
    else:
        field, value = {
            "region": ("region_id", "other"),
            "generation": ("generation", "old"),
            "transfer": ("transfer_id", "delivery:d0:v1"),
            "entry": ("key", KVEntryKey("model", "other", "upload")),
            "receiver": ("receiver_epoch", "old-process"),
        }[fault]
        identities[0] = replace(identities[0], **{field: value})
    with pytest.raises(ProtocolValidationError):
        c.receiver.adopt(identities)
    assert c.receiver._identities == {}


def test_adoption_is_atomic_repeatable_but_not_replaceable(case):
    c = case
    publish(c)
    c.receiver.adopt(dict(c.identities))
    changed = {**c.identities, 0: replace(c.identities[0], sender_epoch="restarted-V")}
    with pytest.raises(ProtocolValidationError, match="replace"):
        c.receiver.adopt(changed)
    c.receiver.observe(reply(c, 0))
    assert not c.receiver.ready


@pytest.mark.parametrize(
    "fault",
    [
        "protocol",
        "plan",
        "identity",
        "rank",
        "fenced",
        "short",
        "float-bytes",
        "not-submitted-bytes",
        "missing-field",
    ],
)
def test_malformed_or_stale_proof_does_not_count(case, fault):
    c = case
    publish(c)
    value = reply(c, 0)
    if fault == "protocol":
        value["protocol"] = "legacy"
    elif fault == "plan":
        value["plan_fingerprint"] = "old-plan"
    elif fault == "identity":
        value["identity"] = replace(c.identities[0], sender_epoch="old").to_dict()
    elif fault == "rank":
        value["source_rank"] = True
    elif fault == "fenced":
        value["fenced"] = 1
    elif fault == "short":
        value["transferred_bytes"] -= 1
    elif fault == "float-bytes":
        value["transferred_bytes"] = float(value["transferred_bytes"])
    elif fault == "not-submitted-bytes":
        value["transport_state"] = "not_submitted"
    else:
        value.pop("plan_fingerprint")
    with pytest.raises(ProtocolValidationError):
        c.receiver.observe(value)
    assert not c.receiver._proofs and not c.receiver.ready and not c.released


def test_duplicate_terminal_proof_is_idempotent_but_cannot_change(case):
    c = case
    publish(c)
    c.receiver.observe(reply(c, 0))
    c.receiver.observe(reply(c, 0))
    assert len(c.receiver._proofs) == 1
    with pytest.raises(ProtocolValidationError, match="proof changed"):
        c.receiver.observe(
            reply(c, 0, transport_state="terminal_failed", transferred_bytes=0)
        )


def test_published_manifest_is_a_copy_and_calls_are_owner_thread_only(case):
    c = case
    publish(c)
    c.manifest["writers"].clear()
    assert set(c.receiver._plans) == {0, 1}
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(ProtocolValidationError, match="another thread"):
            pool.submit(c.receiver.cancel).result()
    assert not c.receiver._cancelled


def test_local_reader_pin_outlives_network_close(case):
    c = case
    c.guard.pin("local-import")
    publish(c)
    for rank in (0, 1):
        c.receiver.observe(reply(c, rank))
    c.receiver.close()
    assert not c.released
    c.guard.unpin("local-import")
    assert len(c.released) == 1


def test_capacity_refusal_precedes_plan_materialization_and_pin(case, monkeypatch):
    c = case
    import sglang.srt.disaggregation.pvd.full_kv_fanin as module

    monkeypatch.setattr(
        module,
        "packed_fanin_transfer_slices",
        lambda *a, **kw: pytest.fail("materialized oversized plan"),
    )
    before = set(c.guard._owners)
    with pytest.raises(ProtocolValidationError, match="slice count"):
        FullKVFanInReceiver(
            key=c.key,
            delivery_id="new",
            registration=c.registration,
            guard=c.guard,
            storage=c.storage,
            compute=c.compute,
            token_count=3,
            max_slices=1,
        )
    assert c.guard._owners == before


def test_replaced_backing_bytes_are_rejected(case):
    c = case
    publish(c)
    old = c.registration.buffer
    c.registration.buffer = torch.zeros_like(old)
    with pytest.raises(ProtocolValidationError, match="destination changed"):
        c.receiver.observe(reply(c, 0))
    assert not c.released
    c.registration.buffer = old


def test_local_release_error_leaves_closed_marker_and_retains_guard(case):
    c = case
    publish(c)
    for rank in (0, 1):
        c.receiver.observe(reply(c, rank))

    def fail():
        raise RuntimeError("unregister uncertain")

    c.guard._release = fail
    with pytest.raises(RuntimeError, match="unregister uncertain"):
        c.receiver.close()
    assert c.guard.value is c.registration and c.receiver._guard is c.guard
    with pytest.raises(ProtocolValidationError, match="closed"):
        c.receiver.publish()


def test_two_fake_writers_supply_real_authorization_closures(case):
    c = case
    publish(c)
    expected = torch.zeros_like(c.registration.buffer)
    for rank in (1, 0):
        raw = (torch.arange(c.size // 2, dtype=torch.int64) + 37 * rank).to(torch.uint8)
        registration = c.engine.register_memory(
            raw, endpoint=f"V{rank}", rank=rank, rail="mlx5_7"
        )
        guard = ResourceGuard(
            registration, lambda reg=registration: c.engine.release_memory(reg)
        )
        auth = WriteAuthorization(c.identities[rank], guard)
        guard.request_release()
        auth.begin(c.identities[rank])
        written = 0
        for part in c.manifest["writers"][str(rank)]:
            lo, ro, length = part["local_offset"], part["remote_offset"], part["length"]
            handle = c.engine.submit_put(
                MemorySlice(registration, lo, length),
                c.registration.descriptor,
                remote_offset=ro,
            )
            assert handle.transport_state == TransportState.TERMINAL_SUCCESS
            written += handle.transferred_bytes
            expected[ro : ro + length] = raw[lo : lo + length]
        auth.close()
        auth.observe_terminal(c.identities[rank], TransportState.TERMINAL_SUCCESS)
        proof = reply(
            c,
            rank,
            fenced=auth.fence(c.identities[rank])["fenced"],
            transferred_bytes=written,
        )
        c.receiver.observe(proof)
        assert guard.value is None
        assert c.receiver.ready is (rank == 0)
        assert not c.released
    assert torch.equal(c.registration.buffer, expected)
    c.receiver.close()
    assert len(c.released) == 1
