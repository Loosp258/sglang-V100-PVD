"""Bounded paired-attention storage; graph mode is an explicit diagnostic.

Every call uses the original variable span, GQA expansion and causal mask.
Only SDPA may be captured. Query publication and bank futures stay outside.
The caller serializes attention on one CUDA stream and owns final drainage.
"""
import torch
import torch.nn.functional as F


class PairedAttentionWorkspace:
    def __init__(self, *, device, dtype, q_heads, kv_heads, head_dim,
                 max_bank_rows, max_history, max_bytes=33554432, graph=False):
        device = torch.device(device)
        if (dtype not in (torch.float16, torch.float32) or q_heads <= 0 or kv_heads <= 0
                or q_heads % kv_heads or head_dim <= 0 or max_bank_rows <= 0
                or max_history < 0 or type(graph) is not bool
                or graph and device.type != 'cuda'):
            raise ValueError('bounded paired attention layout and explicit CUDA graph required')
        span = max_bank_rows + max_history + 2
        elements = 2 * (q_heads + kv_heads) * span * head_dim + 2 * q_heads * head_dim
        byte_count = elements * (2 if dtype == torch.float16 else 4)
        byte_count += kv_heads * span + 2 * q_heads * span + kv_heads * (max_history + 2)
        if byte_count > max_bytes:
            raise ValueError('attention workspace exceeds declared scratch budget')
        self.device, self.dtype = device, dtype
        self.q_heads, self.kv_heads, self.head_dim = q_heads, kv_heads, head_dim
        self.max_bank_rows, self.max_history, self.max_span = max_bank_rows, max_history, span
        self.max_bytes, self.base_bytes = max_bytes, byte_count
        self.graph, self.closed, self.quarantined = graph, False, False
        self.k = torch.zeros(kv_heads * span * head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.expanded_k = torch.zeros(q_heads * span * head_dim, device=device, dtype=dtype)
        self.expanded_v = torch.zeros_like(self.expanded_k)
        self.q = torch.zeros((2, q_heads, head_dim), device=device, dtype=dtype)
        self.valid = torch.zeros(kv_heads * span, device=device, dtype=torch.bool)
        self.mask = torch.zeros(2 * q_heads * span, device=device, dtype=torch.bool)
        self.ones = torch.ones(kv_heads * (max_history + 2), device=device, dtype=torch.bool)
        self.graphs = {}
        self.graph_reserved_bytes = self.graph_allocated_bytes = 0
        self._graph_pool = torch.cuda.graph_pool_handle() if graph else None
        self._capture_stream = torch.cuda.Stream(device=device) if graph else None

    def _views(self, span):
        return (self.k[:self.kv_heads * span * self.head_dim].view(self.kv_heads, span, self.head_dim),
            self.v[:self.kv_heads * span * self.head_dim].view(self.kv_heads, span, self.head_dim),
            self.expanded_k[:self.q_heads * span * self.head_dim].view(self.q_heads, span, self.head_dim),
            self.expanded_v[:self.q_heads * span * self.head_dim].view(self.q_heads, span, self.head_dim),
            self.valid[:self.kv_heads * span].view(self.kv_heads, span),
            self.mask[:2 * self.q_heads * span].view(1, self.q_heads, 2, span))

    def _sdpa(self, span):
        _, _, keys, values, _, mask = self._views(span)
        return F.scaled_dot_product_attention(self.q.transpose(0, 1)[None],
            keys[None], values[None], attn_mask=mask, dropout_p=0.0)

    @torch.inference_mode()
    def prime(self, spans):
        if not self.graph or self.closed or self.quarantined:
            raise RuntimeError('live explicit graph diagnostic required')
        spans = tuple(sorted(set(spans)))
        if any(type(span) is not int or not 2 <= span <= self.max_span for span in spans):
            raise ValueError('captured span exceeds bounded workspace')
        before_reserved = torch.cuda.memory_reserved(self.device)
        before_allocated = torch.cuda.memory_allocated(self.device)
        try:
            self._capture_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._capture_stream):
                for span in spans:
                    if span in self.graphs:
                        continue
                    *_, mask = self._views(span)
                    mask.fill_(True)
                    mask[:, :, 0, -1] = False
                    for _ in range(2):
                        self._sdpa(span)
                    self._capture_stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self._capture_stream, pool=self._graph_pool):
                        output = self._sdpa(span)
                    self.graphs[span] = graph, output
            self._capture_stream.synchronize()
            self.graph_reserved_bytes += max(0, torch.cuda.memory_reserved(self.device) - before_reserved)
            self.graph_allocated_bytes += max(0, torch.cuda.memory_allocated(self.device) - before_allocated)
            if self.base_bytes + max(self.graph_reserved_bytes, self.graph_allocated_bytes) > self.max_bytes:
                raise RuntimeError('captured graph storage exceeds declared scratch budget')
        except BaseException:
            self.quarantined = True
            raise

    def attention(self, q, k, v, bank, prior):
        if self.closed or self.quarantined:
            raise RuntimeError('attention workspace closed or quarantined')
        if (q.device != self.device or q.dtype != self.dtype
                or tuple(q.shape) != (2, self.q_heads, self.head_dim)
                or bank.keys.shape[1] > self.max_bank_rows or len(prior) > self.max_history):
            raise ValueError('paired attention exceeds workspace scope')
        span = bank.keys.shape[1] + len(prior) + 2
        keys, values, expanded_k, expanded_v, valid, mask = self._views(span)
        torch.cat([bank.keys, *(item[0] for item in prior), k.transpose(0, 1)], dim=1, out=keys)
        torch.cat([bank.values, *(item[1] for item in prior), v.transpose(0, 1)], dim=1, out=values)
        ones = self.ones[:self.kv_heads * (len(prior) + 2)].view(self.kv_heads, len(prior) + 2)
        torch.cat([bank.valid, ones], dim=1, out=valid)
        repeat = self.q_heads // self.kv_heads
        expanded_k.view(self.kv_heads, repeat, span, self.head_dim).copy_(keys[:, None])
        expanded_v.view(self.kv_heads, repeat, span, self.head_dim).copy_(values[:, None])
        mask.view(self.kv_heads, repeat, 2, span).copy_(valid[:, None, None])
        mask[:, :, 0, -1] = False
        if self.graph:
            if span not in self.graphs:
                raise RuntimeError('diagnostic graph shape not primed; no silent capture/fallback')
            self.q.copy_(q)
            graph, output = self.graphs[span]
            graph.replay()
        else:
            output = F.scaled_dot_product_attention(q.transpose(0, 1)[None],
                expanded_k[None], expanded_v[None], attn_mask=mask, dropout_p=0.0)
        return output[0].transpose(0, 1).reshape(2, -1)

    def snapshot(self):
        return dict(base_bytes=self.base_bytes, max_bytes=self.max_bytes,
            graph=self.graph, graph_shapes=sorted(self.graphs),
            graph_reserved_bytes=self.graph_reserved_bytes,
            graph_allocated_bytes=self.graph_allocated_bytes,
            closed=self.closed, quarantined=self.quarantined)

    def close(self):
        if self.closed:
            return
        if self.quarantined:
            raise RuntimeError('retain unproven captured attention owners')
        try:
            if self.device.type == 'cuda':
                torch.cuda.current_stream(self.device).synchronize()
                if self._capture_stream is not None:
                    self._capture_stream.synchronize()
        except BaseException:
            self.quarantined = True
            raise
        self.graphs.clear()
        self._graph_pool = self._capture_stream = None
        self.k = self.v = self.expanded_k = self.expanded_v = None
        self.q = self.valid = self.mask = self.ones = None
        self.closed = True
