"""V-side index lifecycle: ordering, independence from delivery, identity.

CPU/logic only. No vectors are built, no cuVS or CAGRA is involved, and no GPU
is touched. These tests establish that the state machine refuses the unsafe
orderings; they say nothing about recall, build time, or V100S support.
"""

import pytest
from sglang.srt.disaggregation.pvd.index_lifecycle import (
    IndexDescriptor,
    IndexGate,
    IndexState,
)

ENTRY = "entry-1"
SPACE = "target/model-8b"
MAPPING = "map-v1"


def make_descriptor(**kwargs):
    base = dict(
        index_version="idx-1",
        entry_transfer_id=ENTRY,
        vector_space=SPACE,
        id_mapping_version=MAPPING,
        vector_count=128,
        metric="ip",
    )
    base.update(kwargs)
    return IndexDescriptor(**base)


def ready_gate():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    gate.begin_build()
    gate.mark_ready(make_descriptor())
    return gate


# --------------------------------------------------------------------------
# Construction and descriptor validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", None, 7])
def test_an_entry_id_is_required(bad):
    with pytest.raises(ValueError):
        IndexGate(bad)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_the_attempt_limit_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError):
        IndexGate(ENTRY, max_build_attempts=bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index_version": ""},
        {"entry_transfer_id": "  "},
        {"vector_space": ""},
        {"id_mapping_version": ""},
        {"metric": ""},
        {"vector_count": 0},
        {"vector_count": -1},
        {"vector_count": True},
        {"vector_count": 1.0},
    ],
)
def test_an_incoherent_descriptor_is_refused(kwargs):
    with pytest.raises(ValueError):
        make_descriptor(**kwargs)


# --------------------------------------------------------------------------
# An index may only be built over complete, visible KV
# --------------------------------------------------------------------------


def test_a_fresh_gate_is_absent_and_neither_deliverable_nor_searchable():
    gate = IndexGate(ENTRY)
    assert gate.state is IndexState.ABSENT
    assert not gate.deliverable
    assert not gate.searchable


def test_building_before_the_kv_is_readable_is_refused():
    gate = IndexGate(ENTRY)
    with pytest.raises(ValueError, match="before the complete Prompt KV"):
        gate.begin_build()
    assert gate.state is IndexState.ABSENT
    assert gate.attempts == 0


def test_readable_kv_unblocks_building():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    gate.begin_build()
    assert gate.state is IndexState.BUILDING
    assert gate.attempts == 1


def test_two_concurrent_builds_are_refused():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    gate.begin_build()
    with pytest.raises(ValueError, match="already in progress"):
        gate.begin_build()
    assert gate.attempts == 1


# --------------------------------------------------------------------------
# Delivery must never depend on the index
# --------------------------------------------------------------------------


def test_stored_kv_is_deliverable_with_no_index_at_all():
    """Bootstrap pulls the whole prompt; it must not wait for INDEX_READY."""
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    assert gate.deliverable
    assert gate.state is IndexState.ABSENT
    assert not gate.searchable


@pytest.mark.parametrize("phase", ["building", "ready", "failed"])
def test_delivery_stays_available_through_every_index_phase(phase):
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    gate.begin_build()
    if phase == "ready":
        gate.mark_ready(make_descriptor())
    elif phase == "failed":
        gate.mark_failed("cuvs build error")
    assert gate.deliverable


def test_a_failed_index_does_not_make_the_entry_undeliverable():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    gate.begin_build()
    gate.mark_failed("out of memory")
    assert gate.state is IndexState.FAILED
    assert gate.deliverable
    assert not gate.searchable


# --------------------------------------------------------------------------
# Readiness, immutability and reuse
# --------------------------------------------------------------------------


def test_a_ready_index_is_searchable():
    gate = ready_gate()
    assert gate.state is IndexState.READY
    assert gate.searchable
    assert gate.descriptor.index_version == "idx-1"


def test_readiness_is_not_consumed_by_a_search():
    """One immutable Prompt index serves many Delivery rounds."""
    gate = ready_gate()
    for _ in range(5):
        descriptor, _ = gate.authorize_search(
            SPACE, expected_id_mapping_version=MAPPING
        )
        assert descriptor.index_version == "idx-1"
    assert gate.searchable


def test_a_built_index_is_immutable():
    gate = ready_gate()
    with pytest.raises(ValueError, match="immutable"):
        gate.begin_build()


def test_readiness_without_a_build_is_refused():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    with pytest.raises(ValueError, match="no index build"):
        gate.mark_ready(make_descriptor())


def test_a_descriptor_for_another_entry_is_refused():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    gate.begin_build()
    with pytest.raises(ValueError, match="different Entry"):
        gate.mark_ready(make_descriptor(entry_transfer_id="entry-2"))
    assert not gate.searchable


def test_a_non_descriptor_is_refused():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    gate.begin_build()
    with pytest.raises(ValueError, match="descriptor is required"):
        gate.mark_ready({"index_version": "idx-1"})


# --------------------------------------------------------------------------
# Failure is distinguishable from absence, and retries are bounded
# --------------------------------------------------------------------------


def test_failure_records_a_reason_and_is_not_absence():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    gate.begin_build()
    gate.mark_failed("cuvs raised")
    assert gate.state is IndexState.FAILED
    assert gate.error == "cuvs raised"
    assert gate.state is not IndexState.ABSENT


def test_failing_without_a_build_is_refused():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    with pytest.raises(ValueError, match="no index build"):
        gate.mark_failed("nothing to fail")


def test_an_empty_failure_reason_is_refused():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    gate.begin_build()
    with pytest.raises(ValueError, match="reason"):
        gate.mark_failed("   ")


def test_a_failed_build_may_be_retried_up_to_the_limit():
    gate = IndexGate(ENTRY, max_build_attempts=2)
    gate.mark_kv_readable()
    for attempt in range(2):
        gate.begin_build()
        assert gate.attempts == attempt + 1
        gate.mark_failed("transient")
    assert gate.exhausted
    with pytest.raises(ValueError, match="refusing another attempt"):
        gate.begin_build()


def test_a_successful_retry_clears_the_error():
    gate = IndexGate(ENTRY, max_build_attempts=2)
    gate.mark_kv_readable()
    gate.begin_build()
    gate.mark_failed("transient")
    gate.begin_build()
    gate.mark_ready(make_descriptor())
    assert gate.searchable
    assert gate.error is None
    assert not gate.exhausted


# --------------------------------------------------------------------------
# Identities are not conflated
# --------------------------------------------------------------------------


def test_searching_a_not_ready_index_is_refused_with_its_state():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    with pytest.raises(ValueError, match="absent"):
        gate.authorize_search(SPACE, expected_id_mapping_version=MAPPING)
    gate.begin_build()
    with pytest.raises(ValueError, match="building"):
        gate.authorize_search(SPACE, expected_id_mapping_version=MAPPING)


def test_a_query_from_another_vector_space_is_refused():
    gate = ready_gate()
    with pytest.raises(ValueError, match="draft/model-1b"):
        gate.authorize_search("draft/model-1b", expected_id_mapping_version=MAPPING)


def test_a_stale_id_mapping_is_refused():
    gate = ready_gate()
    with pytest.raises(ValueError, match="id mapping has been rebuilt"):
        gate.authorize_search(SPACE, expected_id_mapping_version="map-v2")


def test_index_version_and_mapping_version_are_separate_identities():
    gate = IndexGate(ENTRY)
    gate.mark_kv_readable()
    gate.begin_build()
    gate.mark_ready(make_descriptor(index_version="idx-9", id_mapping_version="map-v7"))
    descriptor, validated = gate.authorize_search(
        SPACE, expected_id_mapping_version="map-v7"
    )
    assert descriptor.index_version == "idx-9"
    assert descriptor.id_mapping_version == "map-v7"
    # The caller pinned the mapping and not the index build, and the gate
    # says so rather than reporting a blanket "validated".
    assert validated == ("vector_space", "id_mapping_version")
    with pytest.raises(ValueError, match="index has been rebuilt"):
        gate.authorize_search(SPACE, expected_index_version="idx-8")


# --------------------------------------------------------------------------
# Closing
# --------------------------------------------------------------------------


def test_closing_refuses_further_use_and_stops_delivery():
    gate = ready_gate()
    gate.close()
    assert gate.state is IndexState.CLOSED
    assert not gate.searchable
    assert not gate.deliverable


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.mark_kv_readable(),
        lambda g: g.begin_build(),
        lambda g: g.mark_ready(make_descriptor()),
        lambda g: g.mark_failed("late"),
        lambda g: g.authorize_search(SPACE, expected_id_mapping_version=MAPPING),
    ],
)
def test_a_closed_gate_refuses_every_transition(call):
    gate = ready_gate()
    gate.close()
    with pytest.raises(ValueError, match="closed"):
        call(gate)


def test_closing_retains_the_descriptor_for_the_owner_to_free():
    gate = ready_gate()
    descriptor = gate.descriptor
    gate.close()
    assert gate.descriptor is descriptor


def test_the_snapshot_reports_every_distinguishable_state():
    gate = IndexGate(ENTRY)
    assert gate.snapshot()["state"] == "absent"
    gate.mark_kv_readable()
    gate.begin_build()
    assert gate.snapshot()["state"] == "building"
    gate.mark_failed("boom")
    failed = gate.snapshot()
    assert failed["state"] == "failed"
    assert failed["error"] == "boom"
    assert failed["deliverable"] is True
    gate.begin_build()
    gate.mark_ready(make_descriptor())
    ready = gate.snapshot()
    assert ready["state"] == "ready"
    assert ready["searchable"] is True
    assert ready["index_version"] == "idx-1"
