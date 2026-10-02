"""Paired target forward using an existing TP1 SGLang Qwen2 model's weights.

The explicit owner supplies sparse per-layer banks and EAGLE's one-token
proposal. This adapter is not a Scheduler admission or KV transport factory.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.disaggregation.pvd.oasis_attention import (
    PairedForwardOwner, PairedLayerAttention,
)


class SGLangQwenPairedDecode:
    supports_early_publication = True
    def __init__(self, runner, *, execution_lock, workspace=None):
        from sglang.srt.models.qwen2 import Qwen2ForCausalLM

        model = runner.model
        if (type(model) is not Qwen2ForCausalLM or runner.tp_size != 1
                or runner.pp_size != 1 or runner.attn_cp_size != 1
                or model.quant_config is not None
                or not runner.server_args.disable_cuda_graph
                or not runner.server_args.disable_overlap_schedule):
            raise ValueError("paired target requires TP1/PP1 unquantized graph-disabled Qwen2")
        cfg = model.config
        if (cfg.num_hidden_layers != 28 or cfg.num_attention_heads != 28
                or cfg.num_key_value_heads != 4
                or getattr(cfg, "use_sliding_window", False)):
            raise ValueError("validated Qwen2.5-7B target required")
        if not all(callable(getattr(execution_lock, name, None))
                   for name in ("acquire", "release")):
            raise ValueError("shared target execution lock required")
        self.target, self.lock = model, execution_lock
        self.layers, self.q_heads, self.kv_heads = 28, 28, 4
        self.dim = cfg.hidden_size // self.q_heads
        self.max_context_tokens = runner.model_config.context_len
        self.feature_layers = (1, 13, 24)
        self.generated = [[] for _ in range(self.layers)]
        self.owner = PairedForwardOwner(model.model.embed_tokens.weight.device)
        self._next_position = None
        self.workspace = workspace

    @torch.inference_mode()
    def step(self, current, predicted, position, banks, *, publish=None, capture=None, project=None):
        if (any(type(token) is not int or not 0 <= token < self.target.config.vocab_size
                for token in (current, predicted))
                or type(position) is not int or position < 0
                or position + 1 >= self.max_context_tokens
                or self._next_position is not None and position != self._next_position):
            raise ValueError("valid actual/lookahead token and sequential positions required")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("target execution is busy")
        context = PairedLayerAttention(self.generated, banks, q_heads=self.q_heads,
            kv_heads=self.kv_heads, head_dim=self.dim, feature_layers=self.feature_layers,
            publish=publish, capture=capture, project=project, workspace=self.workspace)
        begun, success = False, False
        try:
            self.owner.begin(context)
            begun = True
            ids = torch.tensor([current, predicted], device=self.owner.device)
            positions = torch.tensor([position, position + 1], device=self.owner.device)
            batch = SimpleNamespace(pvd_oasis_context=context, pvd_query_capture=None)
            context.inputs = (ids, positions, batch)
            hidden = self.target.model(ids, positions, batch)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            logits = F.linear(hidden[:1], self.target.lm_head.weight)
            features = torch.cat(context.features, dim=-1).unsqueeze(0)
            context.outputs = (hidden, logits, features)
            self.owner.complete(commit=True)
            self._next_position = position + 1
            success = True
            return logits, features
        finally:
            if begun and not success and not self.owner.quarantined:
                self.owner.complete(commit=False)
            if not self.owner.quarantined:
                self.lock.release()
