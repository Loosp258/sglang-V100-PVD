"""One V writer's bounded, source-pinned full-KV fan-in PUT batch.

The store must supply its authoritative Entry key/layout/allocation guard and
deduplicate reservations by subdelivery identity before exposing this writer.
Only layout-derived relative ranges are submitted. No CUDA repacking is done.
"""

import copy
import logging
import threading
import time

from sglang.srt.disaggregation.pvd.full_kv_fanin_plan import (
    FULL_KV_FANIN_PROTOCOL,
    validate_fanin_plan,
)
from sglang.srt.disaggregation.pvd.protocol import (
    PVD_GENERATION_METADATA_KEY,
    PVD_RECEIVER_EPOCH_METADATA_KEY,
    PVD_TRANSFER_LIFECYCLE_PROTOCOL,
    ProtocolValidationError,
    WriteIdentity,
)
from sglang.srt.disaggregation.pvd.transfer_authorization import WriteAuthorization
from sglang.srt.disaggregation.pvd.transfer_engine import (
    MemorySlice,
    TransferEngine,
    TransferHandle,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import (
    ResourceGuard,
    TransportState,
)

logger = logging.getLogger(__name__)

# Classic Mooncake 0.3.13.post1 can time out one aggregate batch containing
# more than 100k disjoint GPU slices even when the payload is only tens of MB.
# A bounded native handle per chunk preserves the existing whole-delivery fence:
# source and destination stay pinned until *every* handle is terminal.
NATIVE_BATCH_MAX_SLICES = 8192


class FullKVFanInWriter:
    def __init__(
        self,
        manifest,
        *,
        engine,
        source,
        source_guard,
        source_key,
        source_layout,
        source_rank,
        sender_epoch,
        max_slices,
        max_inflight,
        use_native_batch=False,
    ):
        plan = validate_fanin_plan(manifest, max_slices=max_slices)
        if (
            not isinstance(engine, TransferEngine)
            or not isinstance(source, MemorySlice)
            or not isinstance(source_guard, ResourceGuard)
            or source_guard.value is None
            or type(source_rank) is not int
            or source_rank not in plan.writers
            or type(max_inflight) is not int
            or max_inflight <= 0
            or type(use_native_batch) is not bool
            or (
                use_native_batch
                and not callable(getattr(engine, "submit_batch_put", None))
            )
            or type(source.offset) is not int
            or type(source.length) is not int
            or source_key != plan.key
            or source_layout.fingerprint != plan.storage.fingerprint
        ):
            raise ProtocolValidationError(
                "authoritative bounded fan-in source required"
            )
        expected_source = (
            sum(plan.storage.extra["component_bytes_per_token"]) * plan.token_count
        )
        if (
            source.length != expected_source
            or source.registration.descriptor.rank != source_rank
        ):
            raise ProtocolValidationError(
                "fan-in source rank/length differs from stored shard"
            )
        MemorySlice(source.registration, source.offset, source.length)
        if (
            not source.registration.buffer.is_contiguous()
            or source.registration.buffer.data_ptr()
            != source.registration.descriptor.address
            or source.registration.buffer.numel()
            * source.registration.buffer.element_size()
            < source.registration.descriptor.length
        ):
            raise ProtocolValidationError("fan-in source backing storage mismatch")
        if source.registration.descriptor.rail != plan.destination.rail:
            raise ProtocolValidationError(
                "fan-in source and destination require compatible rail"
            )
        identity = WriteIdentity(
            protocol=PVD_TRANSFER_LIFECYCLE_PROTOCOL,
            sender_epoch=sender_epoch,
            receiver_epoch=plan.destination.backend_metadata.get(
                PVD_RECEIVER_EPOCH_METADATA_KEY
            ),
            transfer_id=f"{plan.delivery_id}:d{plan.destination.rank}:v{source_rank}",
            region_id=plan.destination.region_id,
            generation=plan.destination.backend_metadata.get(
                PVD_GENERATION_METADATA_KEY
            ),
            shard_rank=plan.destination.rank,
            key=plan.key,
        )
        identity.validate_destination(plan.destination)
        self.plan, self.source_rank, self.identity = plan, source_rank, identity
        self._source, self._engine = source, engine
        self._source_descriptor = copy.deepcopy(source.registration.descriptor)
        self._parts, self._max_inflight = plan.writers[source_rank], max_inflight
        self._native_batch = use_native_batch
        self._lock, self._drive_lock = threading.RLock(), threading.Lock()
        self._cursor, self._bytes = 0, 0
        self._pending = []
        self._seen_transfers = set()
        self._started = self._attempted = self._submitting = False
        self._cancelled = self._failed = self._unknown = False
        self._terminal = None
        self.error = None
        self._authorization = WriteAuthorization(identity, source_guard)
        # Local diagnostics only. Never include them in the wire proof: the
        # receiver validates that proof's exact field set.
        self._started_at = None
        self._submit_calls = self._poll_calls = 0
        self._submit_seconds = 0.0

    def snapshot(self):
        with self._lock:
            state = self._terminal or (
                TransportState.UNKNOWN
                if self._unknown
                else TransportState.DRAINING
                if self._cancelled and self._attempted
                else TransportState.IN_FLIGHT
                if self._attempted
                else TransportState.NOT_SUBMITTED
            )
            return {
                "protocol": FULL_KV_FANIN_PROTOCOL,
                "plan_fingerprint": self.plan.fingerprint,
                "source_rank": self.source_rank,
                "identity": self.identity.to_dict(),
                "fenced": self._authorization.fence(self.identity)["fenced"],
                "transport_state": state.value,
                "transferred_bytes": self._bytes,
            }

    def cleanup_complete(self) -> bool:
        with self._lock:
            terminal = self._terminal
            pending = bool(self._pending)
        return (
            terminal is not None
            and terminal.is_locally_safe_to_release
            and not pending
            and self._authorization.cleanup_complete
        )

    def start(self):
        with self._lock:
            if not self._cancelled and self._terminal is None:
                self._started = True
                if self._started_at is None:
                    self._started_at = time.monotonic()
        return self.poll()

    def cancel(self):
        with self._lock:
            self._cancelled = True
            self._authorization.close()  # Prevent every subsequent submit.
        return self.poll()

    def _drain(self):
        # Only the drive-lock holder mutates the pending handle collection.
        for handle, expected in tuple(self._pending):
            try:
                if self._cancelled:
                    try:
                        self._engine.abort(handle)
                    except Exception as exc:
                        self.error = f"abort unconfirmed: {exc}"
                self._poll_calls += 1
                self._engine.poll(handle)
            except BaseException as exc:
                with self._lock:
                    self._unknown = self._cancelled = True
                    self.error = f"poll uncertain: {exc}"
                    self._authorization.close()
                if not isinstance(exc, Exception):
                    raise
                continue
            with handle._lock:
                state, byte_count = handle.transport_state, handle.transferred_bytes
            with self._lock:
                if (
                    not isinstance(state, TransportState)
                    or state == TransportState.UNKNOWN
                ):
                    self._unknown = self._cancelled = True
                    self._authorization.close()
                    continue
                if not state.is_locally_safe_to_release:
                    continue
                if (
                    type(byte_count) is not int
                    or not 0 <= byte_count <= expected
                    or (
                        state == TransportState.TERMINAL_SUCCESS
                        and byte_count != expected
                    )
                ):
                    self._unknown = self._cancelled = True
                    self.error = "invalid native transfer byte count"
                    self._authorization.close()
                    continue
                self._pending.remove((handle, expected))
                self._bytes += byte_count
                if state != TransportState.TERMINAL_SUCCESS:
                    self._failed = self._cancelled = True
                    self._authorization.close()

    def _finish(self):
        report = None
        with self._lock:
            if self._submitting or self._pending or self._unknown:
                return
            if self._terminal is None:
                if not self._cancelled and not (
                    self._started and self._cursor == len(self._parts)
                ):
                    return
                self._terminal = (
                    TransportState.NOT_SUBMITTED
                    if not self._attempted
                    else TransportState.TERMINAL_FAILED
                    if self._failed or self._cursor != len(self._parts)
                    else TransportState.TERMINAL_SUCCESS
                )
                self._authorization.close()
                report = (
                    self.identity.transfer_id,
                    self.source_rank,
                    self._terminal.value,
                    len(self._parts),
                    self._submit_calls,
                    self._poll_calls,
                    self._submit_seconds,
                    (
                        time.monotonic() - self._started_at
                        if self._started_at is not None
                        else 0.0
                    ),
                    self._bytes,
                )
            state = self._terminal
        if report is not None:
            logger.info(
                "PVD full-KV fan-in writer terminal: transfer=%s rank=%s "
                "state=%s planned_slices=%s submit_calls=%s poll_calls=%s "
                "submit_seconds=%.6f elapsed_seconds=%.6f bytes=%s",
                *report,
            )
        try:
            # Native closure and local source cleanup are distinct: retry only
            # local cleanup on later polls; never replay a PUT.
            self._authorization.observe_terminal(self.identity, state)
        except Exception as exc:
            self.error = f"source cleanup retained: {exc}"
        else:
            self._source = None

    def poll(self):
        if not self._drive_lock.acquire(blocking=False):
            return self.snapshot()
        try:
            self._drain()
            if self._native_batch:
                # Fill at most one window per poll. A fast terminal chunk can
                # free a slot immediately, but the bounded loop prevents the
                # reaper interval from serializing every native submission.
                for _ in range(self._max_inflight):
                    before = self._cursor
                    self._submit_batch()
                    self._drain()
                    if self._cursor == before:
                        break
                self._finish()
                return self.snapshot()
            for _ in range(self._max_inflight):
                with self._lock:
                    if (
                        not self._started
                        or self._cancelled
                        or self._unknown
                        or self._terminal is not None
                        or self._cursor == len(self._parts)
                        or len(self._pending) >= self._max_inflight
                    ):
                        break
                    source = self._source
                    if (
                        source.registration.descriptor != self._source_descriptor
                        or source.registration.buffer.data_ptr()
                        != self._source_descriptor.address
                    ):
                        self._failed = self._cancelled = True
                        self.error = "source registration changed"
                        self._authorization.close()
                        break
                    part = self._parts[self._cursor]
                    local = MemorySlice(
                        source.registration,
                        source.offset + part.local_offset,
                        part.length,
                    )
                    if not self._attempted:
                        self._authorization.begin(self.identity)
                    self._attempted = self._submitting = True
                try:
                    submit_started = time.monotonic()
                    self._submit_calls += 1
                    handle = self._engine.submit_put(
                        local, self.plan.destination, remote_offset=part.remote_offset
                    )
                    if not isinstance(handle, TransferHandle):
                        raise TypeError("native submission did not return a handle")
                    if (
                        not isinstance(handle.transfer_id, str)
                        or not handle.transfer_id
                        or handle.transfer_id in self._seen_transfers
                    ):
                        raise ValueError("native handle identity was missing or reused")
                    self._seen_transfers.add(handle.transfer_id)
                except BaseException as exc:
                    with self._lock:
                        self._unknown = self._cancelled = True
                        self.error = f"submission uncertain: {exc}"
                        self._authorization.close()
                    if not isinstance(exc, Exception):
                        raise
                else:
                    with self._lock:
                        self._pending.append((handle, part.length))
                        self._cursor += 1
                        if handle.transport_state in (
                            TransportState.UNKNOWN,
                            TransportState.TERMINAL_FAILED,
                            TransportState.NOT_SUBMITTED,
                        ):
                            self._failed = self._cancelled = True
                            self._authorization.close()
                finally:
                    self._submit_seconds += time.monotonic() - submit_started
                    with self._lock:
                        self._submitting = False
                self._drain()
            self._finish()
            return self.snapshot()
        finally:
            self._drive_lock.release()

    def _submit_batch(self):
        # The validated plan already proves disjoint destination ranges. Bound
        # each native batch's work; the writer still owns all handles until the
        # complete plan has locally safe terminal evidence.
        with self._lock:
            if (
                not self._started
                or self._cancelled
                or self._unknown
                or self._terminal is not None
                or self._cursor == len(self._parts)
                or len(self._pending) >= self._max_inflight
            ):
                return
            source = self._source
            if (
                source.registration.descriptor != self._source_descriptor
                or source.registration.buffer.data_ptr()
                != self._source_descriptor.address
            ):
                self._failed = self._cancelled = True
                self.error = "source registration changed"
                self._authorization.close()
                return
            end = min(self._cursor + NATIVE_BATCH_MAX_SLICES, len(self._parts))
            parts = self._parts[self._cursor : end]
            slices = tuple(
                MemorySlice(
                    source.registration,
                    source.offset + part.local_offset,
                    part.length,
                )
                for part in parts
            )
            offsets = tuple(part.remote_offset for part in parts)
            expected = sum(part.length for part in parts)
            if not self._attempted:
                self._authorization.begin(self.identity)
            self._attempted = self._submitting = True
        submit_started = time.monotonic()
        self._submit_calls += 1
        try:
            handle = self._engine.submit_batch_put(
                slices, self.plan.destination, remote_offsets=offsets
            )
            if not isinstance(handle, TransferHandle):
                raise TypeError("native batch submission did not return a handle")
            if (
                not isinstance(handle.transfer_id, str)
                or not handle.transfer_id
                or handle.transfer_id in self._seen_transfers
            ):
                raise ValueError("native batch handle identity missing or reused")
            self._seen_transfers.add(handle.transfer_id)
        except BaseException as exc:
            with self._lock:
                self._unknown = self._cancelled = True
                self.error = f"batch submission uncertain: {exc}"
                self._authorization.close()
            if not isinstance(exc, Exception):
                raise
        else:
            with self._lock:
                self._pending.append((handle, expected))
                self._cursor = end
                if handle.transport_state in (
                    TransportState.UNKNOWN,
                    TransportState.TERMINAL_FAILED,
                    TransportState.NOT_SUBMITTED,
                ):
                    self._failed = self._cancelled = True
                    self._authorization.close()
        finally:
            self._submit_seconds += time.monotonic() - submit_started
            with self._lock:
                self._submitting = False
