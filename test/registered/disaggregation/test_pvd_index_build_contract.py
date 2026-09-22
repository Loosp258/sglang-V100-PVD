"""An opaque backend handle cannot authorize inconsistent index metadata."""

from dataclasses import replace

import pytest
from sglang.srt.disaggregation.pvd.index_lifecycle import IndexState
from sglang.srt.disaggregation.pvd.index_search import BruteForceIndexBackend
from sglang.srt.disaggregation.pvd.request_state import EntryShardState
from test_pvd_prompt_index import budgeted, stored_entry


@pytest.mark.parametrize(
    "field,value",
    [
        ("vector_space", "another-model"),
        ("metric", "ip"),  # the manager requests l2
        ("dim", 7),
        ("dim", 8.0),
        ("dim", True),
        ("count", 7),
        ("count", 8.0),
        ("count", True),
        ("object", None),
    ],
)
def test_inconsistent_build_is_not_published_and_remains_deliverable(field, value):
    class Backend(BruteForceIndexBackend):
        corrupt = True
        calls = 0

        def build(self, *args, **kwargs):
            built = super().build(*args, **kwargs)
            self.calls += 1
            # Partial build: the first head is valid and must also be discarded.
            if self.corrupt and self.calls == 2:
                return value if field == "object" else replace(built, **{field: value})
            return built

    backend = Backend()
    manager, budget = budgeted(backend)
    store, manifest, _, _ = stored_entry(manager)
    outcome = store.progress_prompt_indexes()
    gate = manager.gate_for(manifest.key.transfer_id)
    assert outcome["failed"] == 1
    assert gate.state is IndexState.FAILED
    assert not gate.searchable
    assert gate.deliverable
    assert "backend built index" in gate.snapshot()["error"]
    assert store.entries[manifest.key].state is EntryShardState.STORED
    assert not manager._entries[manifest.key.transfer_id].indexes
    assert budget.snapshot()["used_staging_bytes"] == 0
    # Refusal must not poison a later correct attempt or its separate owners.
    backend.corrupt = False
    assert store.progress_prompt_indexes()["built"] == 1
    assert gate.searchable
    manager.close(manifest.key.transfer_id)
    assert budget.snapshot()["used_staging_bytes"] == 0
