import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from sglang.srt.disaggregation.pvd.control_server import create_coordinator_app
from sglang.srt.disaggregation.pvd.server import (
    _CoordinatorHealthView,
    _group_reaper,
    _MaintenanceReaperHealth,
    _reaper,
    _run_reaper_round,
)


def test_rank_reaper_retries_failed_steps_and_reports_health_recovery():
    async def run():
        expired = 0
        indexed = 0
        coordinator_rounds = 0
        pressure_rounds = 0
        recovered = asyncio.Event()

        def reap_expired(*, reap_entries):
            nonlocal expired
            assert reap_entries is False
            expired += 1
            if expired == 1:
                raise RuntimeError("injected shard reaper failure")

        def progress_indexes():
            nonlocal indexed
            indexed += 1

        async def reap_coordinator():
            nonlocal coordinator_rounds
            coordinator_rounds += 1
            if coordinator_rounds >= 2:
                recovered.set()

        async def relieve_pressure():
            nonlocal pressure_rounds
            pressure_rounds += 1

        store = SimpleNamespace(
            reap_expired=reap_expired,
            progress_prompt_indexes=progress_indexes,
        )
        coordinator = SimpleNamespace(
            reap_expired=reap_coordinator,
            relieve_index_pressure=relieve_pressure,
        )
        health = _MaintenanceReaperHealth()
        task = asyncio.create_task(_reaper(store, 0.001, coordinator, health))
        try:
            await asyncio.wait_for(recovered.wait(), timeout=1)
            assert not task.done()
            assert expired >= 2
            # The remaining steps still run after one step fails.
            assert indexed >= 2
            assert pressure_rounds >= 2
            status = health.snapshot()
            assert status["failed_rounds"] == 1
            assert status["successful_rounds"] >= 1
            assert status["consecutive_failures"] == 0
            assert status["status"] == "healthy"
            assert "injected shard reaper failure" in status["last_error"]

            class Coordinator:
                async def health(self):
                    return {"healthy": True, "role": "test"}

            visible = await _CoordinatorHealthView(Coordinator(), health).health()
            assert visible["healthy"] is True
            assert visible["maintenance_reaper"] == status
            json.dumps(visible)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_coordinator_http_health_exposes_reaper_status():
    async def run():
        class Coordinator:
            async def health(self):
                return {"healthy": True, "role": "test"}

        reaper_health = _MaintenanceReaperHealth()

        async def fail():
            raise RuntimeError("injected shard error")

        await _run_reaper_round(reaper_health, [("shard_expiration", fail)])
        app = create_coordinator_app(
            _CoordinatorHealthView(Coordinator(), reaper_health)
        )
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/health")
            assert response.status == 200
            payload = await response.json()
            assert payload["healthy"] is False
            assert payload["maintenance_reaper"]["status"] == "degraded"
            assert payload["maintenance_reaper"]["failed_rounds"] == 1

    asyncio.run(run())


def test_group_reaper_continues_other_shards_after_one_shard_raises():
    async def run():
        rank0_calls = 0
        rank1_calls = 0
        coordinator_calls = 0
        pressure_calls = 0
        rank1_reaped = asyncio.Event()
        coordinator_reaped = asyncio.Event()

        def make_store(rank):
            def reap_expired(*, reap_entries):
                nonlocal rank0_calls, rank1_calls
                assert reap_entries is False
                if rank == 0:
                    rank0_calls += 1
                    if rank0_calls == 1:
                        raise RuntimeError("injected rank-0 failure")
                else:
                    rank1_calls += 1
                    rank1_reaped.set()

            return SimpleNamespace(
                rank=rank,
                reap_expired=reap_expired,
                progress_prompt_indexes=lambda: None,
            )

        async def reap_coordinator():
            nonlocal coordinator_calls
            coordinator_calls += 1
            coordinator_reaped.set()

        async def relieve_pressure():
            nonlocal pressure_calls
            pressure_calls += 1

        health = _MaintenanceReaperHealth()
        task = asyncio.create_task(
            _group_reaper(
                [make_store(0), make_store(1)],
                0.001,
                SimpleNamespace(
                    reap_expired=reap_coordinator,
                    relieve_index_pressure=relieve_pressure,
                ),
                health,
            )
        )
        try:
            await asyncio.wait_for(coordinator_reaped.wait(), timeout=1)
            assert not task.done()
            assert rank1_reaped.is_set()
            assert rank1_calls >= 1
            assert coordinator_calls >= 1
            assert pressure_calls >= 1
            assert health.snapshot()["failed_rounds"] == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_reaper_round_propagates_cancellation():
    async def run():
        async def cancelled_step():
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await _run_reaper_round(
                _MaintenanceReaperHealth(), [("cancel", cancelled_step)]
            )

    asyncio.run(run())


def test_coordinator_health_marks_reaper_failure_and_recovers():
    async def run():
        class Coordinator:
            async def health(self):
                return {"healthy": True, "role": "test"}

        reaper_health = _MaintenanceReaperHealth()
        view = _CoordinatorHealthView(Coordinator(), reaper_health)

        async def fail():
            raise RuntimeError("injected coordinator expiration failure")

        await _run_reaper_round(reaper_health, [("coordinator_expiration", fail)])
        degraded = await view.health()
        assert degraded["healthy"] is False
        assert degraded["maintenance_reaper"]["status"] == "degraded"
        assert degraded["maintenance_reaper"]["last_error"] == (
            "coordinator_expiration: RuntimeError: "
            "injected coordinator expiration failure"
        )

        async def recover():
            return None

        await _run_reaper_round(reaper_health, [("coordinator_expiration", recover)])
        healthy = await view.health()
        assert healthy["healthy"] is True
        assert healthy["maintenance_reaper"]["status"] == "healthy"

    asyncio.run(run())
