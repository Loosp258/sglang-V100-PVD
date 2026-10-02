"""Owned GPU rows with asynchronous, proven-complete historical CPU backup.

The native receive record remains responsible for remote terminal proof and
GPUDirect ordering. This owner copies its bytes before that MR may retire. A
bank reader and the CPU backup each hold an independent pin on charged storage.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid

import torch

from sglang.srt.disaggregation.pvd.transfer_lifecycle import ResourceGuard


class GPUBackupReader:
    def __init__(self, pool, guard, owner, rows):
        self.pool, self.guard, self.owner, self.rows = pool, guard, owner, rows
        self.closed = False

    def release_after_copy(self, completion):
        if self.closed:
            raise RuntimeError('GPU backup reader already retired')
        try:
            completion.synchronize()
        except BaseException:
            self.pool.quarantined = True
            raise
        self.rows.clear()
        # The backup thread can refund immediately after the last unpin.
        # Drop borrowed tensor aliases before exposing that transition.
        self.guard.unpin(self.owner)
        self.closed = True
        self.pool._refund_retired(self.guard)


class OasisGPUBackupPool:
    def __init__(self, cache, budget, *, device, request_id, incarnation,
                 max_bytes=33554432, workers=2, before_publish=None,
                 allow_cpu_for_tests=False):
        self.device = torch.device(device)
        if (self.device.type != 'cuda' and not allow_cpu_for_tests
                or not request_id or not incarnation or max_bytes <= 0
                or not 1 <= workers <= 4):
            raise ValueError('bounded request-owned CUDA backup required')
        self.cache, self.budget = cache, budget
        self.request_id, self.incarnation = request_id, incarnation
        self.max_bytes, self.before_publish = max_bytes, before_publish
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='oasis-cpu-backup')
        self._pending, self._owners, self._charges, self._futures = {}, {}, {}, []
        self._bytes, self._peak, self._submitted, self._completed = 0, 0, 0, 0
        self._rows_copied, self._seconds = 0, 0.0
        self.quarantined = self.closed = self.closing = False
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None

    def contains(self, layer, head, token):
        with self._lock:
            return (layer, head, token) in self._pending

    def acquire(self, layer, head, token):
        with self._lock:
            if self.closed or self.closing or self.quarantined:
                raise RuntimeError('GPU backup reader admission closed')
            item = self._pending.get((layer, head, token))
            if item is None:
                return None
            guard, row = item
            owner = 'bank-reader:' + uuid.uuid4().hex
            guard.pin(owner)
            return GPUBackupReader(self, guard, owner, {(head, token): row})

    def _refund_retired(self, guard):
        # Refund only AFTER ResourceGuard has dropped its GPU/host tuple.
        if guard.value is not None:
            return
        with self._lock:
            for identity, owned in tuple(self._owners.items()):
                if owned is guard:
                    byte_count = self._charges.pop(identity)
                    self._owners.pop(identity)
                    self._bytes -= byte_count
                    self.budget.release(identity)

    def copy_and_enqueue(self, manifest, source, layer):
        """Caller has exact remote success + ordering; fence clone before ACK."""
        if (source.device != self.device or source.dtype != torch.uint8
                or source.ndim != 1 or not source.is_contiguous()
                or type(layer) is not int or not 0 <= layer < len(self.cache)
                or source.numel() != manifest.nbytes
                or any(spec.layer != layer or spec.request_id != self.request_id
                       or spec.incarnation != self.incarnation
                       or spec.kv_head >= len(self.cache[layer])
                       or any(token >= len(self.cache[layer][spec.kv_head].valid)
                              for token in spec.token_ids)
                       for spec in manifest.specs)):
            raise ValueError('received rows differ from exact backup scope/layout')
        identity = 'gpu-backup:' + uuid.uuid4().hex
        byte_count = 2 * manifest.nbytes
        with self._lock:
            if self.closed or self.closing or self.quarantined or self._bytes + byte_count > self.max_bytes:
                raise RuntimeError('GPU backup admission closed or budget exceeded')
            keys = [(layer, spec.kv_head, token) for spec in manifest.specs for token in spec.token_ids]
            if any(key in self._pending or key[2] in self.cache[key[0]][key[1]] for key in keys):
                raise RuntimeError('duplicate historical backup row')
            self.budget.reserve(identity, byte_count, 0)
            self._bytes += byte_count
            self._peak = max(self._peak, self._bytes)
        gpu = host = None
        try:
            gpu = source.clone()
            if self.device.type == 'cuda':
                clone_complete = torch.cuda.Event()
                clone_complete.record(torch.cuda.current_stream(self.device))
                clone_complete.synchronize()
                host = torch.empty_like(gpu, device='cpu', pin_memory=True)
                with torch.cuda.stream(self.stream):
                    self.stream.wait_event(clone_complete)
                    host.copy_(gpu, non_blocking=True)
                    copy_complete = torch.cuda.Event()
                    copy_complete.record(self.stream)
            else:
                host = gpu.clone()
                copy_complete = None
            gpu_views = manifest.payload_views(gpu)
            host_views = manifest.payload_views(host)
        except BaseException:
            # A failed CUDA allocation/copy fence cannot prove local safety.
            # Charge and retain every possibly active owner until process exit.
            with self._lock:
                self.quarantined = True
                self._owners[identity] = (gpu, host, source)
            raise

        def release():
            for payload in (*gpu_views, *host_views):
                payload.close()

        guard = ResourceGuard((gpu, host, gpu_views, host_views, copy_complete), release)
        backup_owner, reader_owner = identity + ':cpu', identity + ':bank'
        guard.pin(backup_owner)
        guard.pin(reader_owner)
        rows = {(payload.spec.kv_head, token): payload.tensor[:, offset]
                for payload in gpu_views for offset, token in enumerate(payload.spec.token_ids)}
        with self._lock:
            self._owners[identity] = guard
            self._charges[identity] = byte_count
            for (head, token), row in rows.items():
                self._pending[layer, head, token] = guard, row
            self._submitted += 1

        @torch.inference_mode()
        def backup():
            started = time.perf_counter()
            try:
                if copy_complete is not None:
                    copy_complete.synchronize()
                if self.before_publish is not None:
                    self.before_publish()
                with self._lock:
                    for payload in host_views:
                        for offset, token in enumerate(payload.spec.token_ids):
                            self.cache[layer][payload.spec.kv_head][token] = payload.tensor[:, offset].clone()
                    for key in keys:
                        del self._pending[key]
                    self._completed += 1
                    self._rows_copied += len(keys)
                    self._seconds += time.perf_counter() - started
                guard.request_release()
                guard.unpin(backup_owner)
                self._refund_retired(guard)
            except BaseException:
                self.quarantined = True
                raise  # Pending rows, charge and GPU/host owners stay pinned.

        try:
            future = self._pool.submit(backup)
            with self._lock:
                self._futures.append(future)
        except BaseException:
            self.quarantined = True
            raise
        return GPUBackupReader(self, guard, reader_owner, rows)

    def snapshot(self):
        with self._lock:
            return dict(closed=self.closed, closing=self.closing, quarantined=self.quarantined,
                submitted=self._submitted, completed=self._completed,
                pending_rows=len(self._pending), retained_owners=len(self._owners),
                charged_bytes=self._bytes, peak_bytes=self._peak,
                rows_copied=self._rows_copied, background_seconds=self._seconds)

    def close(self):
        self.closing = True
        self._pool.shutdown(wait=True, cancel_futures=False)
        for future in self._futures:
            future.result()
        with self._lock:
            if self.quarantined or self._pending or self._owners or self._bytes:
                raise RuntimeError('retain unproven GPU backup owners')
            self.closed = True
