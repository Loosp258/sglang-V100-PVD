"""Install proven CPU sparse payloads into the charged monotonic Prompt cache.

No native or D2H ordering is established here. The receiver must prove that
first, fence before ACK, and retain source owners when local completion is unknown.
"""

import torch


def cache_install_tensor_bound(capacity):
    """Index+validity-check tensor bytes, at most four groups/job (not KV data)."""
    if type(capacity) is not int or not 1 <= capacity <= 2048:
        raise ValueError('bounded cache-install capacity required')
    return 4 * capacity * 9 + 4


def row_cache_profile(payloads):
    rows = sum(len(p.spec.token_ids) for p in payloads)
    return dict(cache_install_mode='rows', cache_installed_rows=rows,
        cache_groups=len(payloads),
        cache_kv_bytes=sum(p.tensor.numel() * p.tensor.element_size() for p in payloads),
        cache_row_clones=rows,
        cache_kv_copy_calls=rows, cache_valid_write_calls=rows,
        cache_index_bytes=0)


@torch.inference_mode()
def install_cpu_payloads(payloads, cache, *, capacity, retained):
    """Batch head groups without KV clones; validate every group before writes.

    Cache rows are [Prompt,2,128] f16 CPU, validity is owned bool CPU. A payload
    is [2,selected,128]; its transposed view is the index_copy_ source. Tokens
    are declared valid only after their group's synchronous KV copy returns.
    Failed partial installation is never sufficient evidence for receiver ACK.
    """
    bound = cache_install_tensor_bound(capacity)
    if (not isinstance(retained, list) or len(cache) != 4
            or not 1 <= len(payloads) <= 4):
        raise ValueError('bounded four-head cache and caller retention required')
    destinations = []
    for entry in cache:
        rows, valid = getattr(entry, 'rows', None), getattr(entry, 'valid', None)
        if (not isinstance(rows, torch.Tensor) or not isinstance(valid, torch.Tensor)
                or rows.ndim != 3 or rows.shape[1:] != (2, 128)
                or valid.shape != rows.shape[:1] or not rows.shape[0]
                or rows.dtype != torch.float16 or valid.dtype != torch.bool
                or rows.device.type != 'cpu' or valid.device.type != 'cpu'
                or not rows.is_contiguous() or not valid.is_contiguous()
                or rows.untyped_storage().data_ptr() == valid.untyped_storage().data_ptr()):
            raise ValueError('owned contiguous f16 CPU cache rows and validity required')
        destinations.extend((rows, valid))
    plan, seen, layer = [], set(), payloads[0].spec.layer
    for payload in payloads:
        head, ids, source = payload.spec.kv_head, payload.spec.token_ids, payload.tensor
        if (type(head) is not int or not 0 <= head < 4 or head in seen
                or payload.spec.layer != layer
                or not 1 <= len(ids) <= capacity or len(set(ids)) != len(ids)
                or any(type(t) is not int or not 0 <= t < cache[head].rows.shape[0] for t in ids)
                or source.shape != (2, len(ids), 128) or source.dtype != torch.float16
                or source.device.type != 'cpu'):
            raise ValueError('unique bounded one-layer CPU sparse head groups required')
        if any(source.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
               for t in destinations):
            raise ValueError('payload must not alias CPU cache storage')
        indexes = torch.tensor(ids, dtype=torch.int64)
        if cache[head].valid.index_select(0, indexes).any().item():
            raise RuntimeError('duplicate remote cache row')
        seen.add(head)
        plan.append((cache[head], indexes, source.permute(1, 0, 2)))
    # CPU writes are synchronous. Keep all aliases/metadata in the receiver's
    # owner list nevertheless; exceptions must not free unknown D2H owners.
    for entry, indexes, source in plan:
        retained.extend((entry, indexes, source))
    for entry, indexes, source in plan:
        entry.rows.index_copy_(0, indexes, source)
        entry.valid.index_fill_(0, indexes, True)
    rows = sum(len(p.spec.token_ids) for p in payloads)
    return dict(cache_install_mode='batched', cache_installed_rows=rows,
        cache_groups=len(plan),
        cache_kv_bytes=rows * 512, cache_row_clones=0,
        cache_kv_copy_calls=len(plan), cache_valid_write_calls=len(plan),
        cache_index_bytes=rows * 8, cache_tensor_bound_bytes=bound)
