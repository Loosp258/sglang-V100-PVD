"""Bounded all-head Prompt-bank installation, without changing selected KV.

The caller owns the stream and must fence before releasing ``retained``. Every
allocation is appended before dependent copies launch, including failure paths.
CPU execution checks layout/bytes only; it does not qualify CUDA performance.
"""

import torch


def install_tensor_bound(capacity):
    """Conservative live tensor bytes/job, excluding allocator reservation.

    Four heads, f16 K/V of dimension128: output+mask, stack/pinned host copies,
    miss upload/hit gathers, and ordinary/pinned/device int64 index metadata.
    Host misses and GPU hits partition the same bounded selected rows.
    """
    if type(capacity) is not int or not 1 <= capacity <= 2048:
        raise ValueError('bounded Prompt-bank capacity required')
    return 4 * capacity * (513 + 3 * 512 + 3 * 2 * 8)


def _ids(groups, capacity, prompt_tokens):
    if len(groups) != 4:
        raise ValueError('four KV heads required')
    result = tuple(tuple(ids) for ids in groups)
    if any(len(ids) > capacity or len(set(ids)) != len(ids)
           or any(type(t) is not int or not 0 <= t < prompt_tokens for t in ids)
           for ids in result):
        raise ValueError('unique bounded Prompt IDs required')
    return result


@torch.inference_mode()
def install_batched_bank(chosen, resident, cache, *, device, capacity,
                         prompt_tokens, retained):
    """Return K,V,valid,profile with the same bank order as per-head install.

    Only nonresident rows cross CPU->device. CPU-cache membership is proven
    before allocating/enqueuing output. Resident masks are producer-owned;
    padding is never gathered because lookup uses only resident.ids.
    """
    install_tensor_bound(capacity)
    if (type(prompt_tokens) is not int or prompt_tokens <= 0 or len(cache) != 4
            or not isinstance(retained, list)):
        raise ValueError('bounded cache and caller-owned retention list required')
    chosen = _ids(chosen, capacity, prompt_tokens)
    device = torch.device(device)
    width = max(map(len, chosen))
    hit_src, hit_dst, miss_dst, rows = [], [], [], []
    old_width = 0
    if resident is not None:
        old_ids = _ids(resident.ids, capacity, prompt_tokens)
        old_width = resident.keys.shape[1] if resident.keys.ndim == 3 else -1
        if (old_width != max(map(len, old_ids))
                or resident.keys.shape != (4, old_width, 128)
                or resident.values.shape != resident.keys.shape
                or resident.valid.shape != (4, old_width)
                or resident.valid.dtype != torch.bool
                or any(t.device != device or not t.is_contiguous()
                       for t in (resident.keys, resident.values, resident.valid))
                or any(t.dtype != torch.float16 for t in (resident.keys, resident.values))):
            raise ValueError('owned contiguous resident bank required')
    else:
        old_ids = ((),) * 4
    for head, ids in enumerate(chosen):
        old = {token: index for index, token in enumerate(old_ids[head])}
        for index, token in enumerate(ids):
            dst = head * width + index
            if token in old:
                hit_src.append(head * old_width + old[token])
                hit_dst.append(dst)
            else:
                row = cache[head][token]  # terminal-proven monotonic CPU cache
                if (row.device.type != 'cpu' or row.dtype != torch.float16
                        or row.shape != (2, 128)):
                    raise ValueError('owned f16 CPU K/V cache row required')
                rows.append(row)
                miss_dst.append(dst)

    # Retain source aliases as well as new allocations on partial launch error.
    retained.extend(rows)
    if resident is not None:
        retained.append(resident)
        if device.type == 'cuda' and resident.completion is not None:
            torch.cuda.current_stream(device).wait_event(resident.completion)
    keys = torch.zeros((4, width, 128), device=device, dtype=torch.float16)
    retained.append(keys)
    values = torch.zeros_like(keys)
    retained.append(values)
    valid = torch.zeros((4, width), device=device, dtype=torch.bool)
    retained.append(valid)
    metadata = torch.tensor(hit_src + hit_dst + miss_dst, dtype=torch.int64)
    retained.append(metadata)
    if device.type == 'cuda' and metadata.numel():
        metadata = metadata.pin_memory()
        retained.append(metadata)
    indexes = metadata.to(device, non_blocking=True)
    retained.append(indexes)
    hits = len(hit_src)
    if hits:
        src, dst = indexes[:hits], indexes[hits:2 * hits]
        gathered_k = resident.keys.view(4 * old_width, 128).index_select(0, src)
        retained.append(gathered_k)
        gathered_v = resident.values.view(4 * old_width, 128).index_select(0, src)
        retained.append(gathered_v)
        keys.view(4 * width, 128).index_copy_(0, dst, gathered_k)
        values.view(4 * width, 128).index_copy_(0, dst, gathered_v)
    if rows:
        host = torch.stack(rows)
        retained.append(host)
        if device.type == 'cuda':
            host = host.pin_memory()
            retained.append(host)
        gpu = host.to(device, non_blocking=True)
        retained.append(gpu)
        dst = indexes[2 * hits:]
        keys.view(4 * width, 128).index_copy_(0, dst, gpu[:, 0])
        values.view(4 * width, 128).index_copy_(0, dst, gpu[:, 1])
    for head, ids in enumerate(chosen):
        if ids:
            valid[head, :len(ids)] = True
    profile = dict(mode='batched', selected_rows=sum(map(len, chosen)),
        resident_rows=hits, cpu_rows=len(rows), kv_h2d_bytes=len(rows) * 512,
        kv_h2d_calls=int(bool(rows)), resident_gather_calls=2 * int(bool(hits)),
        resident_scatter_calls=2 * int(bool(hits)), cpu_scatter_calls=2 * int(bool(rows)),
        index_metadata_bytes=metadata.numel() * 8,
        tensor_bound_bytes=install_tensor_bound(capacity), cuda=device.type == 'cuda')
    return keys, values, valid, profile
