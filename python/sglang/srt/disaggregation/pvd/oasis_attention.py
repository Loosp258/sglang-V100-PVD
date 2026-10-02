"""Request-owned paired attention shared by HF and SGLang target forwards.

No model weights, formal token pools, full-prefix probe or sampler are owned
here. A failed forward must drain before pending rows/banks can be released.
"""

import torch
import torch.nn.functional as F


class PairedLayerAttention:
    def __init__(self, history, banks, *, q_heads, kv_heads, head_dim,
                 feature_layers, publish=None, capture=None):
        if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads or head_dim <= 0:
            raise ValueError("valid GQA dimensions required")
        self.history, self.banks = history, banks
        self.q_heads, self.kv_heads, self.head_dim = q_heads, kv_heads, head_dim
        self.feature_layers = tuple(feature_layers)
        self.publish, self.capture = publish, capture
        self.pending, self.features, self.owners = [], [], []
        self._committed = False

    def attention(self, layer, q, k, v):
        if layer != len(self.pending) or self._committed:
            raise RuntimeError("paired layers must execute once in order")
        q = q.reshape(2, self.q_heads, self.head_dim)
        k = k.reshape(2, self.kv_heads, self.head_dim)
        v = v.reshape(2, self.kv_heads, self.head_dim)
        self.owners.extend((q, k, v))
        # Calling this provider may wait for this layer only. Later layers are
        # not consumed or awaited until their own QKV projection is complete.
        bank = self.banks(layer) if callable(self.banks) else self.banks[layer]
        expected = (self.kv_heads, bank.keys.shape[1], self.head_dim)
        if (tuple(bank.keys.shape) != expected or bank.values.shape != bank.keys.shape
                or bank.valid.shape != bank.keys.shape[:2]
                or bank.keys.device != q.device or bank.values.device != q.device
                or bank.valid.device != q.device or bank.keys.dtype != q.dtype
                or bank.values.dtype != q.dtype or bank.valid.dtype != torch.bool):
            raise ValueError("resident layer bank layout differs from paired QKV")
        self.owners.append(bank)
        if bank.completion is not None:
            torch.cuda.current_stream(q.device).wait_event(bank.completion)
        if self.capture is not None:
            self.capture(layer, q[0])
        if self.publish is not None:
            self.publish(layer, q[1], bank)
        prior = self.history[layer]
        keys = torch.cat([bank.keys, *(item[0] for item in prior), k.transpose(0, 1)], dim=1)
        values = torch.cat([bank.values, *(item[1] for item in prior), v.transpose(0, 1)], dim=1)
        valid = torch.cat([bank.valid, torch.ones(
            (self.kv_heads, len(prior) + 2), device=q.device, dtype=torch.bool)], dim=1)
        repeat = self.q_heads // self.kv_heads
        mask = valid.repeat_interleave(repeat, dim=0)[None, :, None, :].expand(
            1, self.q_heads, 2, -1).clone()
        mask[:, :, 0, -1] = False
        self.owners.extend((keys, values, valid, mask))
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None], keys.repeat_interleave(repeat, dim=0)[None],
            values.repeat_interleave(repeat, dim=0)[None], attn_mask=mask,
            dropout_p=0.0)[0].transpose(0, 1).reshape(2, -1)
        # Only actual rows may become committed history. Commit is atomic at
        # successful end of the target forward, rather than once per layer.
        self.pending.append((k[0, :, None, :].clone(), v[0, :, None, :].clone()))
        return output

    def after_layer(self, layer, hidden):
        self.owners.append(hidden)
        if layer in self.feature_layers:
            self.features.append(hidden[:1].clone())

    def commit(self):
        if (self._committed or len(self.pending) != len(self.history)
                or len(self.features) != len(self.feature_layers)):
            raise RuntimeError("complete actual layer states required before commit")
        for history, rows in zip(self.history, self.pending, strict=True):
            history.append(rows)
        self._committed = True


class PairedForwardOwner:
    """Keep failed native owners if a completion fence cannot prove drainage."""

    def __init__(self, device, synchronize=None):
        self.device = torch.device(device)
        self._synchronize = synchronize or (
            (lambda: torch.cuda.current_stream(self.device).synchronize())
            if self.device.type == "cuda" else lambda: None)
        self.active = None
        self.quarantined = False

    def begin(self, context):
        if self.active is not None or self.quarantined:
            raise RuntimeError("paired forward is active or quarantined")
        self.active = context

    def complete(self, *, commit):
        if self.active is None or self.quarantined:
            raise RuntimeError("live paired owner required for completion")
        try:
            self._synchronize()
        except BaseException:
            self.quarantined = True
            raise
        context = self.active
        if commit:
            context.commit()
        self.active = None
