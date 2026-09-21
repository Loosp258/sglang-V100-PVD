"""Opt-in offline CPU sparse Decode consumer, not a registered serving backend.

Keep the caller's runner quiescent. A binding scope owns bank readers across
the ENTIRE model forward. Generated KV remains in the runner pool; Prompt rows
in that pool are never read. Only CPU FP32, full causal MHA/GQA, TP1 are covered.
Temporary attention allocations are NOT production-budgeted here. On a failed
model forward, newly written generated rows must be discarded by the caller;
the scope releases readers but does not roll back an in-place model execution.
"""

import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch
from sglang.srt.disaggregation.pvd.prompt_vectors import QueryHeadMapping
from sglang.srt.disaggregation.pvd.sparse_install import CPUInstalledPromptView
from sglang.srt.disaggregation.pvd.sparse_payload import SparsePayloadError
from sglang.srt.disaggregation.pvd.sparse_working_set import CPUSparseWorkingSet
from torch.nn.functional import scaled_dot_product_attention


@dataclass(frozen=True)
class SparseDecodeBinding:
    slot: int
    request_id: str
    incarnation: str
    query_position: int
    bank: CPUSparseWorkingSet | CPUInstalledPromptView


class CPUSparseDecodeConsumer:
    """Backend-independent pool adapter; no Scheduler or distributed state."""

    def __init__(self, req_pool, kv_pool, *, layers, mapping: QueryHeadMapping):
        self.req_pool, self.kv_pool = req_pool, kv_pool
        self.layers = tuple(layers)
        if (
            not self.layers
            or len(set(self.layers)) != len(self.layers)
            or any(type(i) is not int or i < 0 for i in self.layers)
        ):
            raise SparsePayloadError("explicit unique layer ids required")
        self.mapping = mapping
        table = req_pool.req_to_token
        if (
            not isinstance(mapping, QueryHeadMapping)
            or table.device.type != "cpu"
            or table.ndim != 2
            or table.dtype not in (torch.int32, torch.int64)
        ):
            raise SparsePayloadError(
                "explicit head mapping and CPU integral request map required"
            )
        self._bound = None
        self._seen = set()

    @staticmethod
    def _thread():
        if threading.current_thread() is not threading.main_thread():
            raise SparsePayloadError("offline sparse backend is main-thread-only")

    @contextmanager
    def bind(self, bindings):
        """One model forward, all requests explicitly bound; no dense fallback."""
        self._thread()
        if self._bound is not None:
            raise SparsePayloadError("sparse forward scopes cannot nest")
        bindings = tuple(bindings)
        if not bindings:
            raise SparsePayloadError("explicit request bindings required")
        expected_groups = {
            (layer, head)
            for layer in self.layers
            for head in range(self.mapping.total_kv_heads)
        }
        bound, identities = {}, set()
        with ExitStack() as stack:
            for binding in bindings:
                if not isinstance(binding, SparseDecodeBinding):
                    raise SparsePayloadError("explicit SparseDecodeBinding required")
                bank = binding.bank
                if (
                    type(binding.slot) is not int
                    or not 0 < binding.slot < self.req_pool.req_to_token.shape[0]
                    or binding.slot in bound
                ):
                    raise SparsePayloadError("duplicate or invalid request slot")
                if (
                    not isinstance(bank, (CPUSparseWorkingSet, CPUInstalledPromptView))
                    or bank.identity[:2] != (binding.request_id, binding.incarnation)
                    or bank.expected_groups != expected_groups
                ):
                    raise SparsePayloadError(
                        "binding request/instance/layer/head mismatch"
                    )
                if bank.identity[:2] in identities:
                    raise SparsePayloadError("request instance is bound more than once")
                identities.add(bank.identity[:2])
                if (
                    type(binding.query_position) is not int
                    or binding.query_position < bank.prompt_tokens
                    or (
                        isinstance(bank, CPUInstalledPromptView)
                        and binding.query_position
                        != bank.prompt_tokens + bank.decode_tokens
                    )
                ):
                    raise SparsePayloadError(
                        "decode position must follow the complete Prompt"
                    )
                groups = stack.enter_context(bank.read())
                bound[binding.slot] = (binding, groups)
            self._bound, self._seen = bound, set()
            try:
                yield self
                expected = {(slot, layer) for slot in bound for layer in self.layers}
                if self._seen != expected:
                    raise SparsePayloadError(
                        "forward did not consume every bound request/layer"
                    )
            finally:
                self._bound = None
                self._seen.clear()
                bound.clear()

    def forward_decode(self, q, k, v, layer, batch, save_kv_cache=True):
        self._thread()
        if self._bound is None:
            raise SparsePayloadError(
                "sparse decode requires a whole-forward binding scope"
            )
        if (
            not batch.forward_mode.is_decode()
            or batch.encoder_lens is not None
            or getattr(batch, "spec_info", None) is not None
            or getattr(batch, "spec_algorithm", None) is not None
            or getattr(batch, "pvd_query_capture", None) is not None
        ):
            raise SparsePayloadError("only ordinary causal decode is supported")
        if (
            layer.layer_id not in self.layers
            or layer.is_cross_attention
            or getattr(layer.attn_type, "value", layer.attn_type) != "decoder"
            or layer.logit_cap != 0
            or layer.sliding_window_size not in (None, -1)
            or getattr(layer, "quant_method", None) is not None
            or getattr(layer, "use_irope", False)
            or getattr(layer, "k_scale", None) is not None
            or getattr(layer, "v_scale", None) is not None
            or layer.tp_q_head_num != self.mapping.num_query_heads
            or layer.tp_k_head_num != self.mapping.total_kv_heads
            or layer.tp_v_head_num != self.mapping.total_kv_heads
            or layer.qk_head_dim != layer.v_head_dim
        ):
            raise SparsePayloadError(
                "unsupported sparse layer/head/attention configuration"
            )
        if not save_kv_cache or k is None or v is None:
            raise SparsePayloadError("decode must append its own generated K/V")
        slots = self._indices(batch.req_pool_indices, "request slots")
        positions = self._indices(batch.positions, "positions")
        lengths = self._indices(batch.seq_lens, "sequence lengths")
        destinations = self._indices(batch.out_cache_loc, "write locations")
        count, dim = len(slots), layer.qk_head_dim
        if (
            count != len(self._bound)
            or len(set(slots)) != count
            or set(slots) != set(self._bound)
            or not len(positions) == len(lengths) == len(destinations) == count
            or len(set(destinations)) != count
        ):
            raise SparsePayloadError("batch does not match bound requests")
        for value, heads in (
            (q, self.mapping.num_query_heads),
            (k, self.mapping.total_kv_heads),
            (v, self.mapping.total_kv_heads),
        ):
            if (
                value.device.type != "cpu"
                or value.dtype != torch.float32
                or tuple(value.shape) not in ((count, heads * dim), (count, heads, dim))
            ):
                raise SparsePayloadError(
                    "sparse consumer requires CPU FP32 Q/K/V shapes"
                )
        keys = self.kv_pool.get_key_buffer(layer.layer_id)
        values = self.kv_pool.get_value_buffer(layer.layer_id)
        for pool in (keys, values):
            if (
                pool.device.type != "cpu"
                or pool.dtype != torch.float32
                or pool.ndim != 3
                or tuple(pool.shape[1:]) != (self.mapping.total_kv_heads, dim)
            ):
                raise SparsePayloadError("unsupported generated KV pool layout")
        plans, occupied = [], set()
        for slot, pos, length, dest in zip(
            slots, positions, lengths, destinations, strict=True
        ):
            binding, groups = self._bound[slot]
            bank = binding.bank
            if (
                pos != binding.query_position
                or length != pos + 1
                or length > self.req_pool.req_to_token.shape[1]
                or bank.head_dim != dim
                or (slot, layer.layer_id) in self._seen
            ):
                raise SparsePayloadError("stale position, shape or repeated layer")
            rows = self.req_pool.req_to_token[slot, bank.prompt_tokens : length].long()
            row_ids = rows.tolist()
            if (
                not row_ids
                or row_ids[-1] != dest
                or len(set(row_ids)) != len(row_ids)
                or any(
                    r <= 0 or r >= min(keys.shape[0], values.shape[0]) for r in row_ids
                )
                or occupied.intersection(row_ids)
            ):
                raise SparsePayloadError(
                    "generated KV mapping is invalid or aliases another request"
                )
            occupied.update(row_ids)
            plans.append((slot, groups, rows))
        # Validate the whole layer/batch BEFORE mutating any generated KV row.
        self.kv_pool.set_kv_buffer(
            layer,
            batch.out_cache_loc,
            k.reshape(count, self.mapping.total_kv_heads, dim),
            v.reshape(count, self.mapping.total_kv_heads, dim),
        )
        outputs = []
        q = q.reshape(count, self.mapping.num_query_heads, dim)
        for row, (slot, groups, generated_rows) in enumerate(plans):
            generated_k, generated_v = keys[generated_rows], values[generated_rows]
            head_outputs = []
            for head in range(self.mapping.num_query_heads):
                kv_head = self.mapping.kv_head_for(head)
                _, prompt = groups[(layer.layer_id, kv_head)]
                all_k = torch.cat((prompt[0], generated_k[:, kv_head]))
                all_v = torch.cat((prompt[1], generated_v[:, kv_head]))
                # Validated one-token Decode: all selected Prompt positions and
                # generated positions <= query position. No compact-index RoPE.
                head_outputs.append(
                    scaled_dot_product_attention(
                        q[row, head].reshape(1, 1, dim),
                        all_k.unsqueeze(0),
                        all_v.unsqueeze(0),
                        scale=layer.scaling,
                        dropout_p=0.0,
                        is_causal=False,
                    ).reshape(dim)
                )
            outputs.append(torch.stack(head_outputs))
            self._seen.add((slot, layer.layer_id))
        return torch.stack(outputs).reshape(count, self.mapping.num_query_heads * dim)

    @staticmethod
    def _indices(tensor, name):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.ndim != 1
            or tensor.dtype not in (torch.int32, torch.int64)
        ):
            raise SparsePayloadError(f"CPU integral {name} required")
        return tensor.tolist()


def make_offline_sparse_backend(runner):
    """Construct explicitly, never register as a production backend/CLI option."""
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
    from sglang.srt.models.llama import LlamaForCausalLM

    if (
        type(runner.model) is not LlamaForCausalLM
        or runner.device != "cpu"
        or runner.tp_size != 1
        or runner.pp_size != 1
        or runner.attn_cp_size != 1
        or type(runner.attn_backend) is not TorchNativeAttnBackend
        or runner.server_args.enable_dp_attention
        or runner.server_args.speculative_algorithm is not None
        or runner.model.quant_config is not None
        or runner.server_args.page_size != 1
        or not runner.server_args.disable_overlap_schedule
    ):
        raise SparsePayloadError(
            "offline backend requires unquantized CPU TP1/PP1 Llama, native attention, page size 1 and no overlap scheduling"
        )

    class OfflineSparseBackend(TorchNativeAttnBackend):
        def __init__(self):
            super().__init__(runner)
            config = runner.model.config
            self.consumer = CPUSparseDecodeConsumer(
                self.req_to_token_pool,
                self.token_to_kv_pool,
                layers=range(config.num_hidden_layers),
                mapping=QueryHeadMapping(
                    config.num_attention_heads, config.num_key_value_heads
                ),
            )

        def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            return self.consumer.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        def forward_extend(self, *args, **kwargs):
            raise SparsePayloadError(
                "offline sparse backend supports bound decode only"
            )

    return OfflineSparseBackend()
