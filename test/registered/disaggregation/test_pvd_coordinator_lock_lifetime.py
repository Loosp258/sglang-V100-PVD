"""Coordinator operation locks must not become per-ID lifetime tombstones."""

import asyncio
import gc
from types import SimpleNamespace

import pytest
from sglang.srt.disaggregation.pvd.coordinator import (
    CoordinatorError,
    VectorCoordinator,
)
from sglang.srt.disaggregation.pvd.protocol import KVEntryKey


def test_unknown_operation_locks_do_not_accumulate_after_callers_exit():
    async def run():
        coordinator = VectorCoordinator(
            [SimpleNamespace(rank=0), SimpleNamespace(rank=1)]
        )
        for number in range(50):
            key = KVEntryKey.new("model", f"missing-{number}")
            with pytest.raises(CoordinatorError, match="unknown entry"):
                await coordinator.reserve_delivery(
                    key=key, delivery_id=f"delivery-{number}", destinations={}
                )
            with pytest.raises(KeyError):
                await coordinator.start_delivery(f"missing-{number}")
            with pytest.raises(KeyError):
                await coordinator.cancel_entry(key, "missing")
        gc.collect()
        assert len(coordinator._delivery_reserve_locks) == 0
        assert len(coordinator._delivery_start_locks) == 0
        assert len(coordinator._entry_cleanup_locks) == 0
        assert coordinator.entries == {}
        assert coordinator.deliveries == {}

    asyncio.run(run())
