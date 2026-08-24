"""Scheduler-facing PVD sender/receiver adapters.

The existing PD queues remain responsible for request admission and KV page
allocation.  These adapters replace only their point-to-point handshake and
data movement when ``disaggregation_topology == 'pvd'``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.distributed

from sglang.srt.disaggregation.base.conn import KVTransferMetric, KVPoll
from sglang.srt.disaggregation.pvd.client import PVDCoordinatorClient
from sglang.srt.disaggregation.pvd.kv_packer import (
    PVD_TENSOR_LAYOUT,
    describe_kv_layout,
    pack_full_prompt_kv,
    unpack_full_prompt_kv,
)
from sglang.srt.disaggregation.pvd.mooncake_engine import MooncakePVDTransferEngine
from sglang.srt.disaggregation.pvd.protocol import (
    FirstTokenMetadata,
    KVEntryKey,
    KVEntryManifest,
    KVLayoutSignature,
    KVShardManifest,
    RemoteRegionDescriptor,
)
from sglang.srt.disaggregation.pvd.runtime import (
    PVDDecodeRuntime,
    PVDEntryLease,
    PVDPrefillRuntime,
)
from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine


class PVDConnectionError(RuntimeError):
    pass


class _AsyncControlLoop:
    """One daemon asyncio loop per scheduler process for aiohttp operations."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run, name="pvd-control-plane", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coroutine) -> concurrent.futures.Future:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)


class PVDKVManager:
    """Minimal shared context used by both scheduler-facing PVD adapters."""

    def __init__(
        self,
        *,
        scheduler,
        kv_pool,
        metadata_buffers,
        tp_rank: int,
        tp_size: int,
        gloo_group,
    ) -> None:
        if tp_size != 2:
            raise PVDConnectionError("PVD v1 requires exactly two TP ranks")
        if scheduler.ps.pp_size != 1:
            raise PVDConnectionError("PVD v1 does not support pipeline parallelism")
        if scheduler.tp_worker.is_hybrid_swa:
            raise PVDConnectionError("PVD v1 does not support hybrid/SWA KV pools")
        if hasattr(kv_pool, "get_state_buf_infos"):
            state_ptrs, _, _ = kv_pool.get_state_buf_infos()
            if state_ptrs:
                raise PVDConnectionError(
                    "PVD v1 does not support KV pools with SWA/DSA/Mamba state buffers"
                )
        req_to_token_pool = getattr(scheduler, "req_to_token_pool", None)
        if req_to_token_pool is not None and hasattr(
            req_to_token_pool, "get_state_buf_infos"
        ):
            state_ptrs, _, _ = req_to_token_pool.get_state_buf_infos()
            if state_ptrs:
                raise PVDConnectionError(
                    "PVD v1 does not support request-scoped Mamba state buffers"
                )

        shared_engine = get_mooncake_transfer_engine()
        if shared_engine is None:
            raise PVDConnectionError("PVD requires the shared Mooncake transfer engine")

        self.scheduler = scheduler
        self.kv_pool = kv_pool
        self.metadata_buffers = metadata_buffers
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.gloo_group = gloo_group
        self.page_size = kv_pool.page_size
        self.rails = [item.strip() for item in scheduler.server_args.pvd_rank_rails.split(",")]
        self.rail = self.rails[tp_rank]
        self.model_instance_id = scheduler.server_args.pvd_model_instance_id
        self.control = _AsyncControlLoop()
        self.client = PVDCoordinatorClient(
            scheduler.server_args.pvd_vector_coordinator_url,
            timeout_seconds=300.0,
        )
        self.transfer_engine = MooncakePVDTransferEngine.from_existing(
            shared_engine, rail=self.rail
        )
        self.prefill_runtime = PVDPrefillRuntime(
            model_instance_id=self.model_instance_id,
            coordinator=self.client,
            transfer_engine=self.transfer_engine,
        )
        self.decode_runtime = PVDDecodeRuntime(
            coordinator=self.client,
            transfer_engine=self.transfer_engine,
        )
        self.kv_args = SimpleNamespace(state_types=[])
        self.is_dummy_cp_rank = False
        self._layout_description = describe_kv_layout(kv_pool)

    def key_for(self, req) -> KVEntryKey:
        transfer_id = getattr(req, "pvd_transfer_id", None)
        if not transfer_id:
            raise PVDConnectionError(
                "PVD request is missing pvd_transfer_id; send it through the PVD Gateway"
            )
        # P and D may independently assign SGLang rid values. The Gateway's
        # transfer id is therefore also the cross-role request identity.
        return KVEntryKey(
            model_instance_id=self.model_instance_id,
            req_id=transfer_id,
            transfer_id=transfer_id,
        )

    def layout(self) -> KVLayoutSignature:
        model_config = self.scheduler.model_config
        components = self._layout_description
        kv_heads = getattr(self.kv_pool, "head_num", None)
        if kv_heads is None:
            token_shape = components["component_token_shapes"][0]
            kv_heads = token_shape[-2] if len(token_shape) >= 2 else 1
        return KVLayoutSignature(
            model_id=str(model_config.model_path),
            model_revision=str(self.scheduler.server_args.revision or ""),
            kv_dtype=components["component_dtypes"][0],
            page_size=self.page_size,
            num_layers=int(model_config.num_hidden_layers),
            kv_heads_per_rank=int(kv_heads),
            head_dim=int(model_config.head_dim),
            tp_size=self.tp_size,
            pp_size=1,
            tensor_layout=PVD_TENSOR_LAYOUT,
            extra=components,
        )

    def local_shard_manifest(self, prompt_tokens: int) -> KVShardManifest:
        page_count = math.ceil(prompt_tokens / self.page_size)
        bytes_per_token = sum(
            self._layout_description["component_bytes_per_token"]
        )
        start_layer = int(getattr(self.kv_pool, "start_layer", 0))
        end_layer_value = getattr(self.kv_pool, "end_layer", None)
        end_layer = int(
            end_layer_value
            if end_layer_value is not None
            else start_layer + self.scheduler.model_config.num_hidden_layers
        )
        return KVShardManifest(
            rank=self.tp_rank,
            rail=self.rail,
            expected_bytes=page_count * self.page_size * bytes_per_token,
            page_count=page_count,
            last_page_valid_tokens=prompt_tokens % self.page_size or self.page_size,
            layer_start=start_layer,
            layer_end=end_layer,
        )

    def gather_rank_objects(self, local: Dict[str, Any]) -> List[Dict[str, Any]]:
        gathered: List[Optional[Dict[str, Any]]] = [None] * self.tp_size
        torch.distributed.all_gather_object(
            gathered, local, group=self.gloo_group
        )
        result = [item for item in gathered if item is not None]
        if sorted(int(item["rank"]) for item in result) != [0, 1]:
            raise PVDConnectionError("PVD rank object exchange did not produce ranks 0 and 1")
        return result

    async def wait_for_stored_entry(
        self, key: KVEntryKey, *, timeout_seconds: float = 300.0
    ) -> Dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            match = await self.decode_runtime.select_entry(key)
            state = match.get("state")
            if match.get("found") and state == "stored":
                return match["entry"]
            if state in ("failed", "cancelled", "released"):
                raise PVDConnectionError(f"PVD Entry became terminal before delivery: {match}")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"timed out waiting for PVD Entry {key}")
            await asyncio.sleep(0.01)


class PVDKVSender:
    def __init__(self, *, mgr: PVDKVManager, req) -> None:
        self.kv_mgr = mgr
        self.req = req
        self.key = mgr.key_for(req)
        self.conclude_state = None
        self._error: Optional[BaseException] = None
        self._lease: Optional[PVDEntryLease] = None
        self._publish_future: Optional[concurrent.futures.Future] = None
        self._started_at: Optional[float] = None
        self._metric = KVTransferMetric()

        local_shard = mgr.local_shard_manifest(len(req.origin_input_ids))
        gathered = mgr.gather_rank_objects(
            {"rank": mgr.tp_rank, "shard": local_shard.to_dict()}
        )
        shards = {
            int(item["rank"]): KVShardManifest.from_dict(item["shard"])
            for item in gathered
        }
        self._create_future = mgr.control.submit(
            mgr.prefill_runtime.create_entry(
                req_id=self.key.req_id,
                transfer_id=self.key.transfer_id,
                layout=mgr.layout(),
                prompt_token_count=len(req.origin_input_ids),
                shards=shards,
            )
        )
        self._expected_pages = local_shard.page_count

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        if num_kv_indices != self._expected_pages:
            raise PVDConnectionError(
                f"PVD Prefill must publish every prompt page: "
                f"expected {self._expected_pages}, got {num_kv_indices}"
            )

    def pop_decode_prefix_len(self) -> int:
        return 0

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        # Entry is immutable and always contains the complete prompt KV.
        return last_chunk

    def _first_token(self) -> Optional[FirstTokenMetadata]:
        if self.kv_mgr.tp_rank != 0:
            return None
        req = self.req
        logprob = req.logprob.output_token_logprobs_val
        logprob_idx = req.logprob.output_token_logprobs_idx
        top_values = req.logprob.output_top_logprobs_val
        top_indices = req.logprob.output_top_logprobs_idx
        return FirstTokenMetadata(
            output_token_id=int(req.output_ids[0]),
            cached_tokens=int(req.cached_tokens),
            cached_tokens_device=int(req.cached_tokens_device),
            cached_tokens_host=int(req.cached_tokens_host),
            cached_tokens_storage=int(req.cached_tokens_storage),
            output_token_logprob=float(logprob[0]) if logprob else None,
            output_token_logprob_index=int(logprob_idx[0]) if logprob_idx else None,
            output_top_logprobs_values=list(top_values[0]) if top_values else None,
            output_top_logprobs_indices=list(top_indices[0]) if top_indices else None,
        )

    def send(self, kv_indices, state_indices: Optional[List] = None):
        try:
            self._send(kv_indices, state_indices)
        except BaseException as exc:
            self._error = exc
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.control.submit(
                self.kv_mgr.client.cancel_entry(self.key, str(exc))
            )

    def _send(self, kv_indices, state_indices: Optional[List] = None):
        if state_indices and any(item is not None for item in state_indices):
            raise PVDConnectionError("PVD v1 does not support auxiliary KV state")
        if self._lease is None:
            self._lease = self._create_future.result()
        packed = pack_full_prompt_kv(
            self.kv_mgr.kv_pool, kv_indices, page_size=self.kv_mgr.page_size
        )
        if packed.page_count != self._expected_pages:
            raise PVDConnectionError(
                f"PVD packed {packed.page_count} pages, expected {self._expected_pages}"
            )
        self._started_at = time.monotonic()
        self._metric.transfer_total_bytes = packed.expected_bytes
        self._publish_future = self.kv_mgr.control.submit(
            self.kv_mgr.prefill_runtime.publish_tensor_shard(
                lease=self._lease,
                rank=self.kv_mgr.tp_rank,
                tensor=packed.tensor,
                endpoint="pvd-prefill",
                rail=self.kv_mgr.rail,
                first_token=self._first_token(),
            )
        )

    def poll(self) -> int:
        if self.conclude_state is not None:
            return self.conclude_state
        try:
            if self._lease is None:
                if not self._create_future.done():
                    return KVPoll.Bootstrapping
                self._lease = self._create_future.result()
                return KVPoll.WaitingForInput
            if self._publish_future is None:
                return KVPoll.WaitingForInput
            if not self._publish_future.done():
                return KVPoll.Transferring
            self._publish_future.result()
            if self._started_at is not None:
                self._metric.transfer_latency_s = time.monotonic() - self._started_at
            self.conclude_state = KVPoll.Success
        except BaseException as exc:
            self._error = exc
            self.conclude_state = KVPoll.Failed
        return self.conclude_state

    def get_transfer_metric(self) -> KVTransferMetric:
        return self._metric

    def failure_exception(self):
        raise PVDConnectionError(str(self._error or "unknown PVD Prefill failure"))

    def abort(self):
        self.conclude_state = KVPoll.Failed
        self.kv_mgr.control.submit(
            self.kv_mgr.client.cancel_entry(self.key, "Prefill request aborted")
        )

    def clear(self):
        pass


class PVDKVReceiver:
    def __init__(self, *, mgr: PVDKVManager, req) -> None:
        self.kv_mgr = mgr
        self.req = req
        self.key = mgr.key_for(req)
        self.delivery_id = getattr(req, "pvd_delivery_id", None)
        if not self.delivery_id:
            raise PVDConnectionError(
                "PVD request is missing pvd_delivery_id; send it through the PVD Gateway"
            )
        self.conclude_state = None
        self.require_staging = False
        self._error: Optional[BaseException] = None
        self._entry_record: Optional[Dict[str, Any]] = None
        self._entry_future: Optional[concurrent.futures.Future] = None
        self._delivery_future: Optional[concurrent.futures.Future] = None
        self._ack_future: Optional[concurrent.futures.Future] = None
        self._registration = None
        self._staging: Optional[torch.Tensor] = None
        self._page_indices = None
        self._unpacked = False
        self._started_at: Optional[float] = None

    def init(self, prefill_dp_rank: int):
        self._entry_future = self.kv_mgr.control.submit(
            self.kv_mgr.wait_for_stored_entry(self.key)
        )

    def _validate_entry(self, record: Dict[str, Any]) -> KVEntryManifest:
        manifest = KVEntryManifest.from_dict(record["manifest"])
        if manifest.layout.fingerprint != self.kv_mgr.layout().fingerprint:
            raise PVDConnectionError("PVD Entry layout does not match Decode KV layout")
        if manifest.prompt_token_count != len(self.req.origin_input_ids):
            raise PVDConnectionError("PVD Entry prompt length does not match Decode request")
        return manifest

    def _write_first_token_metadata(self, aux_index: int) -> None:
        token = self._entry_record.get("first_token")
        if token is None:
            raise PVDConnectionError("stored PVD Entry has no first-token metadata")
        metadata = FirstTokenMetadata.from_dict(token)
        buffers = self.kv_mgr.metadata_buffers
        buffers.output_ids[aux_index, 0] = metadata.output_token_id
        buffers.cached_tokens[aux_index, 0] = metadata.cached_tokens
        buffers.cached_tokens[aux_index, 1] = metadata.cached_tokens_device
        buffers.cached_tokens[aux_index, 2] = metadata.cached_tokens_host
        buffers.cached_tokens[aux_index, 3] = metadata.cached_tokens_storage
        if metadata.output_token_logprob is not None:
            buffers.output_token_logprobs_val[aux_index, 0] = metadata.output_token_logprob
        if metadata.output_token_logprob_index is not None:
            buffers.output_token_logprobs_idx[aux_index, 0] = metadata.output_token_logprob_index
        if metadata.output_top_logprobs_values:
            values = torch.tensor(
                metadata.output_top_logprobs_values,
                dtype=buffers.output_top_logprobs_val.dtype,
                device=buffers.output_top_logprobs_val.device,
            )
            buffers.output_top_logprobs_val[aux_index, : values.numel()] = values
        if metadata.output_top_logprobs_indices:
            indices = torch.tensor(
                metadata.output_top_logprobs_indices,
                dtype=buffers.output_top_logprobs_idx.dtype,
                device=buffers.output_top_logprobs_idx.device,
            )
            buffers.output_top_logprobs_idx[aux_index, : indices.numel()] = indices
        buffers.bootstrap_room[aux_index, 0] = self.req.bootstrap_room or 0

    def send_metadata(
        self,
        kv_indices,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        try:
            self._send_metadata(
                kv_indices,
                aux_index=aux_index,
                state_indices=state_indices,
                decode_prefix_len=decode_prefix_len,
            )
        except BaseException as exc:
            self._error = exc
            self._release_registration()
            self.conclude_state = KVPoll.Failed

    def _send_metadata(
        self,
        kv_indices,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if aux_index is None:
            raise PVDConnectionError("PVD Decode requires a metadata buffer index")
        if decode_prefix_len not in (None, 0):
            raise PVDConnectionError("PVD v1 requires Decode radix cache to be disabled")
        if state_indices and any(item is not None for item in state_indices):
            raise PVDConnectionError("PVD v1 does not support auxiliary KV state")
        if self._entry_record is None:
            self._entry_record = self._entry_future.result()
        manifest = self._validate_entry(self._entry_record)
        shard = manifest.shard(self.kv_mgr.tp_rank)
        if len(kv_indices) != shard.page_count:
            raise PVDConnectionError(
                f"Decode allocated {len(kv_indices)} pages, Entry requires {shard.page_count}"
            )
        self._page_indices = kv_indices
        self._staging = torch.empty(
            shard.expected_bytes,
            dtype=torch.uint8,
            device=f"cuda:{self.kv_mgr.scheduler.ps.gpu_id}",
        )
        self._registration = self.kv_mgr.transfer_engine.register_memory(
            self._staging,
            endpoint="pvd-decode",
            rank=self.kv_mgr.tp_rank,
            rail=self.kv_mgr.rail,
            metadata={"delivery_id": self.delivery_id},
        )
        gathered = self.kv_mgr.gather_rank_objects(
            {
                "rank": self.kv_mgr.tp_rank,
                "destination": self._registration.descriptor.to_dict(),
            }
        )
        destinations = {
            int(item["rank"]): RemoteRegionDescriptor.from_dict(item["destination"])
            for item in gathered
        }
        self._write_first_token_metadata(aux_index)
        self._started_at = time.monotonic()
        self._delivery_future = self.kv_mgr.control.submit(
            self.kv_mgr.decode_runtime.deliver(
                key=self.key,
                destinations=destinations,
                delivery_id=self.delivery_id,
            )
        )

    def poll(self) -> int:
        if self.conclude_state is not None:
            return self.conclude_state
        try:
            if self._entry_record is None:
                if self._entry_future is None or not self._entry_future.done():
                    return KVPoll.Bootstrapping
                self._entry_record = self._entry_future.result()
                self._validate_entry(self._entry_record)
                return KVPoll.WaitingForInput
            if self._delivery_future is None:
                return KVPoll.WaitingForInput
            if not self._delivery_future.done():
                return KVPoll.Transferring
            self._delivery_future.result()
            if not self._unpacked:
                torch.cuda.synchronize(self._staging.device)
                unpack_full_prompt_kv(
                    self._staging,
                    self.kv_mgr.kv_pool,
                    self._page_indices,
                    page_size=self.kv_mgr.page_size,
                )
                self._unpacked = True
                self._ack_future = self.kv_mgr.control.submit(
                    self.kv_mgr.decode_runtime.ack(self.delivery_id)
                )
                return KVPoll.Transferring
            if not self._ack_future.done():
                return KVPoll.Transferring
            self._ack_future.result()
            self._release_registration()
            self.conclude_state = KVPoll.Success
        except BaseException as exc:
            self._error = exc
            self._release_registration()
            self.conclude_state = KVPoll.Failed
        return self.conclude_state

    def _release_registration(self) -> None:
        if self._registration is not None:
            self.kv_mgr.transfer_engine.release_memory(self._registration)
            self._registration = None

    def failure_exception(self):
        raise PVDConnectionError(str(self._error or "unknown PVD Decode failure"))

    def abort(self):
        self.conclude_state = KVPoll.Failed
        self.kv_mgr.control.submit(
            self.kv_mgr.client.cancel_delivery(self.delivery_id, "Decode request aborted")
        )
        self._release_registration()

    def clear(self):
        self._release_registration()
