"""Real CPU KV and HTTP; CUDA fence tests use an explicit policy double."""

import asyncio
import copy
import json

import pytest
from sglang.srt.disaggregation.pvd import vector_store
from sglang.srt.disaggregation.pvd.sparse_receiver import SparseReceiveError
from sglang.srt.disaggregation.pvd.v_source_profile import (
    SOURCE_PHASES, VSourceProfile, copy_source_profile,
)
from test_pvd_cuda_sparse_packing import cuda_policy
from test_pvd_sparse_receiver import finish, receiving


def test_http_source_profile_is_bounded_and_separate_from_native_proof():
    async def run():
        async with receiving() as c:
            assert not await c.record.start()
            assert "v_source" not in c.record.profile
            delivery = finish(c)
            assert await c.record.poll()
            profile = c.record.profile["v_source"]
            assert profile == delivery.source_profile.snapshot()
            assert profile["cuda"] is False and profile["kernel"] == "torch"
            assert profile["nbytes"] == c.manifest.nbytes
            assert len(json.dumps(profile)) < 1500
            for name, phase in profile["phases"].items():
                assert phase["calls"] == phase["successes"] == (
                    0 if name in ("pack_fence", "outer_fence") else 1
                )
                assert phase["seconds"] >= 0
            c.record.stage(c.group, c.epoch)
            assert c.group.try_install(c.epoch, {0: 4})
            await c.record.ack()
            assert await c.record.close()
            assert c.record.profile["v_source"] == profile
    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, "copy", "pack_fence", "outer_fence", "register"])
def test_source_phases_count_failed_attempts_without_repairing_unknown(monkeypatch, failure):
    store, entry, _, _, target, delivery, _ = cuda_policy(monkeypatch)
    real_copy = vector_store.copy_sparse_kv_into

    def copy_rows(*args, **kwargs):
        real_copy(*args, **kwargs)
        if failure == "copy":
            raise RuntimeError("injected partial copy")

    fences = []
    def fence(device):
        fences.append(device)
        if (failure == "pack_fence" and len(fences) == 1
                or failure == "outer_fence" and len(fences) == 2):
            raise RuntimeError("injected CUDA policy fence failure")

    if failure == "register":
        def register(*args, **kwargs):
            raise RuntimeError("injected unknown registration")
        monkeypatch.setattr(store.transfer_engine, "register_memory", register)
    monkeypatch.setattr(vector_store, "copy_sparse_kv_into", copy_rows)
    monkeypatch.setattr(vector_store.torch.cuda, "synchronize", fence)
    store.start_delivery(entry.key, delivery.delivery_id)
    profile = delivery.source_profile.snapshot()
    assert copy_source_profile(profile, nbytes=delivery.sparse_manifest.nbytes) == profile
    phases = profile["phases"]
    assert phases["pack"]["successes"] == int(failure != "copy")
    assert phases["pack_fence"]["calls"] == phases["outer_fence"]["calls"] == 1
    assert phases["pack_fence"]["successes"] == int(failure != "pack_fence")
    assert phases["outer_fence"]["successes"] == int(failure != "outer_fence")
    assert phases["submit"]["calls"] == int(failure is None)
    if failure in ("pack_fence", "outer_fence", "register"):
        assert not store.fence_write(delivery.authorization.identity)["fenced"]
    if failure is None:
        store.transfer_engine.finish(delivery.transfer_handle)
        store.poll_delivery(entry.key, delivery.delivery_id)
        store.ack_delivery(entry.key, delivery.delivery_id)
    store.close()
    store.transfer_engine.release_memory(target)


@pytest.mark.parametrize("bad", ["schema", "bytes", "phase", "calls", "success", "nan", "negative", "cuda"])
def test_optional_profile_parser_refuses_unbounded_or_invalid_diagnostics(bad):
    profile = VSourceProfile(nbytes=32, cuda=False, kernel="torch").snapshot()
    if bad == "schema": profile["schema"] = 2
    if bad == "bytes": profile["nbytes"] = True
    if bad == "phase": profile["phases"]["other"] = {}
    if bad == "calls": profile["phases"]["pack"]["calls"] = 2
    if bad == "success": profile["phases"]["pack"]["successes"] = 1
    if bad == "nan": profile["phases"]["pack"]["seconds"] = float("nan")
    if bad == "negative": profile["phases"]["pack"]["seconds"] = -1
    if bad == "cuda": profile["cuda"] = "yes"
    with pytest.raises(ValueError):
        copy_source_profile(profile, nbytes=32)


def test_remote_profile_cannot_override_exact_terminal_and_byte_checks():
    async def run():
        async with receiving() as c:
            await c.record.start()
            delivery = finish(c)
            c.store.poll_delivery(c.entry.key, delivery.delivery_id)
            reply = delivery.to_dict()
            reply["source_profile"] = VSourceProfile(
                nbytes=c.manifest.nbytes, cuda=True, kernel="torch").snapshot()
            reply["transferred_bytes"] -= 1
            with pytest.raises(SparseReceiveError, match="exact successful terminal proof"):
                c.record._observe(reply)
            assert not c.record._ready and "v_source" not in c.record.profile
            assert await c.record.poll()
            good = copy.deepcopy(delivery.to_dict())
            good["source_profile"]["phases"]["pack"]["seconds"] = -1
            assert c.record._observe(good)
            assert c.record.profile["v_source_profile_invalid"] is True
    asyncio.run(run())
