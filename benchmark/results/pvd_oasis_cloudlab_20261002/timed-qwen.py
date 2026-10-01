"""Explicit HF Qwen2.5 TP1 experimental paired sparse forward.

The model's target projections and MLP process two rows together. Only row zero
is committed; row one is a rejected lookahead. Not a Scheduler attention backend.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PromptBank:
    ids: tuple
    keys: torch.Tensor  # KV heads, padded resident rows, head dimension
    values: torch.Tensor
    valid: torch.Tensor  # KV heads, padded resident rows
    completion: object = None


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class QwenPairedDecode:
    def __init__(self, target):
        cfg = target.config
        if cfg.model_type != "qwen2" or getattr(cfg, "sliding_window", None) is not None:
            # Qwen2.5 config can declare a disabled sliding window.
            if cfg.model_type != "qwen2" or getattr(cfg, "use_sliding_window", False):
                raise ValueError("ordinary Qwen2 target required")
        self.target = target
        self.layers = len(target.model.layers)
        self.q_heads, self.kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.dim = cfg.hidden_size // self.q_heads
        self.repeat = self.q_heads // self.kv_heads
        self.feature_layers = (1, self.layers // 2 - 1, self.layers - 4)
        if self.layers != 28 or self.q_heads != 28 or self.kv_heads != 4:
            raise ValueError("this bounded adapter validates Qwen2.5-7B only")
        self.generated = [[] for _ in range(self.layers)]

    @torch.inference_mode()
    def step(self, current, predicted, position, banks, *, publish=None, capture=None):
        device = self.target.model.embed_tokens.weight.device
        ids = torch.tensor([[current, predicted]], device=device)
        hidden = self.target.model.embed_tokens(ids)
        positions = torch.tensor([[position, position + 1]], device=device)
        cos, sin = self.target.model.rotary_emb(hidden, positions)
        cos, sin = cos[0, :, None, :], sin[0, :, None, :]
        features = []
        for layer_id, layer in enumerate(self.target.model.layers):
            normalized = layer.input_layernorm(hidden)[0]
            attn = layer.self_attn
            q = attn.q_proj(normalized).view(2, self.q_heads, self.dim)
            k = attn.k_proj(normalized).view(2, self.kv_heads, self.dim)
            v = attn.v_proj(normalized).view(2, self.kv_heads, self.dim)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            bank = banks(layer_id) if callable(banks) else banks[layer_id]
            if bank.completion is not None:
                torch.cuda.current_stream(device).wait_event(bank.completion)
            if capture is not None:
                capture(layer_id, q[0])
            if publish is not None:
                publish(layer_id, q[1], bank)
            history = self.generated[layer_id]
            prior_k = [item[0] for item in history]
            prior_v = [item[1] for item in history]
            # The final two rows are current actual K/V and private candidate K/V.
            keys = torch.cat([bank.keys, *prior_k, k.transpose(0, 1)], dim=1)
            values = torch.cat([bank.values, *prior_v, v.transpose(0, 1)], dim=1)
            valid = torch.cat([bank.valid, torch.ones(
                (self.kv_heads, len(history) + 2), device=device, dtype=torch.bool)], dim=1)
            mask = valid.repeat_interleave(self.repeat, dim=0)[None, :, None, :].expand(
                1, self.q_heads, 2, -1).clone()
            mask[:, :, 0, -1] = False  # Actual token cannot see future candidate.
            output = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                keys.repeat_interleave(self.repeat, dim=0)[None],
                values.repeat_interleave(self.repeat, dim=0)[None],
                attn_mask=mask, dropout_p=0.0,
            )[0].transpose(0, 1).reshape(1, 2, -1)
            hidden = hidden + attn.o_proj(output)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            # Copy actual rows, retaining no candidate owner in committed history.
            history.append((k[0, :, None, :].clone(), v[0, :, None, :].clone()))
            if layer_id in self.feature_layers:
                features.append(hidden[:, :1].clone())
        logits = self.target.lm_head(self.target.model.norm(hidden[:, :1]))[:, 0]
        return logits, torch.cat(features, dim=-1)


def full_prompt_banks(cache):
    """P-side / local reference only; never use full Prompt banks to seed D."""
    result = []
    for item in cache.layers:
        keys, values = item.keys[0].clone(), item.values[0].clone()
        heads, rows, _ = keys.shape
        result.append(PromptBank(tuple(tuple(range(rows)) for _ in range(heads)),
            keys, values, torch.ones((heads, rows), device=keys.device, dtype=torch.bool)))
    return result
