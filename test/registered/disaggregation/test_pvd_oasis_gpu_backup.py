"""Actual CPU bytes and event/lifetime policy; native CUDA is a separate gate."""
from dataclasses import replace
import threading

import pytest
import torch

from sglang.srt.disaggregation.pvd.oasis_gpu_backup import OasisGPUBackupPool
from sglang.srt.disaggregation.pvd.oasis_transport import _HeadCPUCache
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
from test_pvd_sparse_copy import setup


class Complete:
    def synchronize(self):
        return None


def fixture(*, max_bytes=1 << 20, gate=None, inference=False):
    _, source, arguments = setup()
    manifest = arguments['manifest']
    manifest = replace(manifest, specs=tuple(replace(spec, layer=0, kv_head=head)
                                           for head, spec in enumerate(manifest.specs)))
    # Independent byte groups are enough here; native uses source K/V oracles.
    dtype = getattr(torch, manifest.dtype.split('.')[-1])
    item_size = torch.empty(0, dtype=dtype).element_size()
    source = torch.arange(manifest.nbytes // item_size).remainder(31).to(dtype).view(torch.uint8)
    with torch.inference_mode(inference):
        rows = torch.empty((4, 32, 2, manifest.head_dim), dtype=dtype)
        valid = torch.zeros((4, 32), dtype=torch.bool)
    cache = [[_HeadCPUCache(rows[h], valid[h]) for h in range(4)]]
    budget = TransferBudget(1 << 20, 4)
    first = manifest.specs[0]
    pool = OasisGPUBackupPool(cache, budget, device='cpu', request_id=first.request_id,
        incarnation=first.incarnation, max_bytes=max_bytes, workers=1,
        before_publish=(lambda: gate.wait(5)) if gate else None, allow_cpu_for_tests=True)
    return pool, manifest, source, valid, budget


def test_gpu_reader_and_cpu_backup_hold_independent_pins_and_exact_cpu_bytes(monkeypatch):
    gate = threading.Event()
    pool, manifest, source, valid, budget = fixture(gate=gate)
    reader = pool.copy_and_enqueue(manifest, source, 0)
    source.fill_(193)  # Simulate the retired MR being reused for unrelated data.
    assert not valid.any() and pool.snapshot()['charged_bytes'] == 2 * manifest.nbytes
    first = manifest.specs[0]
    borrowed = pool.acquire(0, first.kv_head, first.token_ids[0])
    assert borrowed is not None and pool.contains(0, first.kv_head, first.token_ids[0])
    expected = {key: value.clone() for key, value in reader.rows.items()}
    borrowed_mapping = reader.rows
    reader.release_after_copy(Complete())
    assert borrowed_mapping == {}, 'borrowed row mapping survived retirement'
    gate.set()
    for future in pool._futures:
        future.result(5)
    assert valid.sum().item() == sum(len(spec.token_ids) for spec in manifest.specs)
    assert pool.snapshot()['charged_bytes'] > 0, 'borrowed bank reader lost its charge'
    original_unpin = borrowed.guard.unpin
    def refund_at_last_unpin(owner):
        outcome = original_unpin(owner)
        pool._refund_retired(borrowed.guard)
        assert not borrowed.rows, 'budget refunded while borrowed tensors survived'
        return outcome
    monkeypatch.setattr(borrowed.guard, 'unpin', refund_at_last_unpin)
    borrowed.release_after_copy(Complete())
    pool.close()
    for (head, token), value in expected.items():
        assert torch.equal(pool.cache[0][head][token], value)
    assert pool.snapshot()['retained_owners'] == pool.snapshot()['charged_bytes'] == 0
    assert budget.snapshot()['used_staging_bytes'] == 0


def test_budget_and_duplicate_pending_rows_refused_before_copy():
    pool, manifest, source, _, budget = fixture(max_bytes=1)
    with pytest.raises(RuntimeError, match='budget'):
        pool.copy_and_enqueue(manifest, source, 0)
    assert budget.snapshot()['used_staging_bytes'] == 0
    pool.close()
    gate = threading.Event()
    pool, manifest, source, _, _ = fixture(gate=gate)
    reader = pool.copy_and_enqueue(manifest, source, 0)
    with pytest.raises(RuntimeError, match='duplicate'):
        pool.copy_and_enqueue(manifest, source, 0)
    reader.release_after_copy(Complete())
    gate.set()
    pool.close()


def test_foreign_incarnation_rejected_before_admission():
    pool, manifest, source, _, budget = fixture()
    bad = replace(manifest, specs=tuple(replace(spec, incarnation='other') for spec in manifest.specs))
    with pytest.raises(ValueError, match='scope'):
        pool.copy_and_enqueue(bad, source, 0)
    assert budget.snapshot()['used_staging_bytes'] == 0
    pool.close()


def test_inference_cache_writes_are_legal_on_the_background_thread():
    pool, manifest, source, valid, _ = fixture(inference=True)
    reader = pool.copy_and_enqueue(manifest, source, 0)
    reader.release_after_copy(Complete())
    pool.close()
    assert valid.sum().item() > 0


def test_failed_local_reader_fence_retains_charge_and_never_releases_storage():
    pool, manifest, source, _, budget = fixture()
    reader = pool.copy_and_enqueue(manifest, source, 0)
    class Unknown:
        def synchronize(self):
            raise RuntimeError('unknown completion')
    with pytest.raises(RuntimeError, match='unknown'):
        reader.release_after_copy(Unknown())
    for future in pool._futures:
        future.result(5)
    assert reader.guard.value is not None and budget.snapshot()['used_staging_bytes'] > 0
    with pytest.raises(RuntimeError, match='retain'):
        pool.close()
    # Controlled CPU cleanup only; production has no UNKNOWN repair operation.
    reader.release_after_copy(Complete())
