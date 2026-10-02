"""Real packed KV arrival replay through manager gates and filtered searches."""

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import cuvs
import torch

from sglang.srt.disaggregation.pvd.cagra_backend import CagraIndexBackend
from sglang.srt.disaggregation.pvd.cagra_kv_update import CagraKVUpdateBackend
from sglang.srt.disaggregation.pvd.prompt_index import (
    PromptIndexManager,
    SearchRequestIdentity,
)
from sglang.srt.disaggregation.pvd.protocol import KVLayoutSignature, KVShardManifest
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget


def load(path, rank):
    fixture = torch.load(path, map_location="cpu", weights_only=False)[0]
    return load_fixture(fixture, rank)


def load_fixture(fixture, rank):
    n = fixture["local_sources"].shape[1]
    k = (
        fixture["local_sources"]
        .reshape(28, 2, n, 128)
        .permute(0, 2, 1, 3)
        .contiguous()
        .half()
        .cuda(rank)
    )
    packed = torch.cat((k.flatten(), torch.zeros_like(k).flatten())).view(torch.uint8)
    layout = KVLayoutSignature(
        model_id="Qwen2.5-7B-Instruct",
        model_revision="fixture",
        kv_dtype="float16",
        page_size=1,
        num_layers=28,
        total_kv_heads=4,
        kv_heads_per_rank=2,
        head_dim=128,
        tp_size=2,
        pp_size=1,
        tensor_layout="component-major",
        extra={
            "component_count": 56,
            "component_dtypes": ["float16"] * 56,
            "component_token_shapes": [[2, 128]] * 56,
            "component_bytes_per_token": [512] * 56,
        },
    )
    manifest = KVShardManifest(
        rank=rank,
        rail="fixture",
        expected_bytes=packed.numel(),
        page_count=n,
        last_page_valid_tokens=1,
        layer_start=0,
        layer_end=28,
    )
    return fixture, packed, layout, manifest


def quality(manager, key, fixture):
    k = fixture["local_sources"].half().float().to(manager.backend_device)
    q = fixture["local_queries"].float().to(manager.backend_device)
    recalls, invalid = [], 0
    for head, (layer, kv_head) in enumerate(sorted(manager._entries[key].vectors)):
        wanted = (q[head] @ k[head].T).topk(10, dim=1).indices.cpu().tolist()
        values = []
        for query, exact in zip(q[head], wanted):
            identity = SearchRequestIdentity(
                vector_space="qwen25-7b-pvd",
                positional_encoding="rope_applied",
                entry_transfer_id=key,
                layer=layer,
                kv_head=kv_head,
            )
            result = manager.search(
                identity, queries=query[None].contiguous(), top_k=10
            )
            ids = result.selection.token_ids
            invalid += sum(token < 0 or token >= len(k[head]) for token in ids)
            values.append(len(set(ids) & set(exact)) / 10)
        recalls.append(sum(values) / len(values))
    return {
        "mean_head_top10_recall": sum(recalls) / len(recalls),
        "worst_head_top10_recall": min(recalls),
        "invalid_ids": invalid,
        "per_head_recall": recalls,
    }


def run(
    inputs,
    mode,
    boundaries,
    *,
    measure_quality=True,
    small_tail_max_rows=0,
    small_tail_prune=False,
    fused_core_block=0,
    fused_core_rows=1,
    fused_selection=False,
    cuda_graph_min_rows=0,
    fused_prepare=False,
    batched_k_extraction=False,
    fused_k_centering=False,
    profile_chunk_stages=False,
    planned_tail=False,
    reuse_scores=False,
    fused_edge_write=False,
    new_top16=False,
    stream_completion=False,
    profile_gpu=False,
):
    setup_started = time.perf_counter()
    managers = []
    for rank in range(len(inputs)):
        torch.cuda.reset_peak_memory_stats(rank)
        cls = CagraKVUpdateBackend if mode == "kv2" else CagraIndexBackend
        backend = cls(
            device=f"cuda:{rank}",
            native_bytes_per_index=536870912,
            global_native_cap_bytes=671088640,
            graph_degree=16,
            intermediate_degree=16,
            itopk_size=2048,
            exact_head_groups=4,
            nogil_extend=True,
            **(
                {
                    "routing_edges": 2,
                    "small_tail_max_rows": small_tail_max_rows,
                    "small_tail_prune": small_tail_prune,
                    "fused_core_block": fused_core_block,
                    "fused_core_rows": fused_core_rows,
                    "fused_selection": fused_selection,
                    "cuda_graph_min_rows": cuda_graph_min_rows,
                    "fused_prepare": fused_prepare,
                    "prepared_tail": planned_tail,
                    "fixed_native_views": planned_tail,
                    "ahead_capture": planned_tail,
                    "reuse_scores": reuse_scores,
                    "fused_edge_write": fused_edge_write,
                    "new_top16": new_top16,
                    "stream_completion": stream_completion,
                    "profile_gpu": profile_gpu,
                }
                if mode == "kv2"
                else {}
            ),
        )
        manager = PromptIndexManager(
            vector_space="qwen25-7b-pvd",
            backend=backend,
            group_heads=4,
            batched_k_extraction=batched_k_extraction,
            fused_k_centering=fused_k_centering,
            profile_chunk_stages=profile_chunk_stages,
            prepared_tail=planned_tail,
            budget=TransferBudget(2147483648, 1),
        )
        manager.open(f"probe-{rank}")
        manager._probe_backend_seconds = 0.0
        for name in (
            ("build_many", "extend_many") if mode == "kv2" else ("build", "extend")
        ):
            original = getattr(backend, name)

            def timed(*args, _original=original, _manager=manager, **kwargs):
                start = time.perf_counter()
                result = _original(*args, **kwargs)
                _manager._probe_backend_seconds += time.perf_counter() - start
                if isinstance(result, list) and result:
                    batch = result[0].handle.auxiliary[0]
                    _manager._probe_capture_seconds = batch.capture_seconds
                    _manager._probe_prepare_batched = batch.last_prepare_batched
                    _manager._probe_gpu_seconds = batch.last_gpu_seconds
                    _manager._probe_update_gpu_seconds = batch.last_update_gpu_seconds
                    _manager._probe_completion_wait_seconds = (
                        batch.last_completion_wait_seconds
                    )
                    _manager._probe_replayed = batch.last_replayed
                    _manager._probe_reused = batch.last_reused_scores
                    _manager._probe_ahead_capture_seconds = batch.ahead_capture_seconds
                return result

            setattr(backend, name, timed)
        managers.append(manager)
    result = {
        "mode": mode,
        "boundaries": boundaries,
        "steps": [],
        "quality": [],
        "setup_seconds": time.perf_counter() - setup_started,
    }
    for pages in boundaries:
        barrier = threading.Barrier(len(inputs))
        times, failures = {}, []

        def worker(rank):
            try:
                torch.cuda.set_device(rank)
                _, packed, layout, manifest = inputs[rank]
                final = pages == manifest.page_count
                if final:
                    managers[rank].note_kv_readable(f"probe-{rank}")
                barrier.wait()
                managers[rank]._probe_backend_seconds = 0.0
                start = time.perf_counter()
                outcome = managers[rank].progress_chunked(
                    f"probe-{rank}",
                    packed,
                    layout=layout,
                    manifest=manifest,
                    complete_pages=pages,
                    stored=final,
                )
                times[rank] = {
                    "seconds": time.perf_counter() - start,
                    "outcome": outcome,
                    "backend_seconds": managers[rank]._probe_backend_seconds,
                    "capture_seconds": getattr(
                        managers[rank], "_probe_capture_seconds", 0.0
                    ),
                    "prepare_batched": getattr(
                        managers[rank], "_probe_prepare_batched", False
                    ),
                    "stage_timings": getattr(
                        managers[rank], "_last_chunk_stage_timings", {}
                    ),
                    "gpu_compute_seconds": getattr(
                        managers[rank], "_probe_gpu_seconds", 0
                    ),
                    "gpu_update_seconds": getattr(
                        managers[rank], "_probe_update_gpu_seconds", 0
                    ),
                    "completion_wait_seconds": getattr(
                        managers[rank], "_probe_completion_wait_seconds", 0
                    ),
                    "replayed": getattr(managers[rank], "_probe_replayed", False),
                    "reused_scores": getattr(managers[rank], "_probe_reused", False),
                    "ahead_capture_seconds": getattr(
                        managers[rank], "_probe_ahead_capture_seconds", 0
                    ),
                }
            except BaseException as exc:
                failures.append(exc)
                barrier.abort()

        threads = [
            threading.Thread(target=worker, args=(rank,)) for rank in range(len(inputs))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            raise failures[0]
        result["steps"].append({"pages": pages, "ranks": times})
    for rank, manager in enumerate(managers):
        torch.cuda.set_device(rank)
        if measure_quality:
            result["quality"].append(quality(manager, f"probe-{rank}", inputs[rank][0]))
            if mode == "kv2":
                batch = next(iter(manager.backend._owners.values())).auxiliary[0]
                record = manager._entries[f"probe-{rank}"]
                result.setdefault("batch_fingerprints", {})[rank] = {
                    name: hashlib.sha256(
                        tensor.contiguous().cpu().numpy().tobytes()
                    ).hexdigest()
                    for name, tensor in {
                        "native_data": batch.native_data[:, : 4 * batch.count],
                        "native_graph": batch.native_graph[:, : 4 * batch.count],
                        "mean": torch.stack(
                            [
                                record.group_means[key]
                                for key in sorted(record.group_means)
                            ]
                        ),
                    }.items()
                }
        result.setdefault("budget_before_close", {})[rank] = manager.budget.snapshot()
        result.setdefault("torch_memory_before_close", {})[rank] = {
            "allocated": torch.cuda.memory_allocated(rank),
            "peak_allocated": torch.cuda.max_memory_allocated(rank),
        }
        manager.close(f"probe-{rank}")
        assert not manager.backend._owners
        assert manager.backend.runtime.global_allocated_bytes() == 0
        assert (
            manager.budget.snapshot()["used_staging_bytes"]
            == manager.shared_native_budget_bytes
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, nargs=2, required=True)
    parser.add_argument("--schedule", default="native,kv2,kv2,native")
    parser.add_argument(
        "--boundaries", default="256,512,768,1024,1280,1536,1792,2048,2159"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = [load(path, rank) for rank, path in enumerate(args.fixtures)]
    boundaries = [int(value) for value in args.boundaries.split(",")]
    if (
        not boundaries
        or any(a >= b for a, b in zip([0] + boundaries, boundaries))
        or boundaries[-1] != inputs[0][3].page_count
    ):
        parser.error("boundaries must increase to the complete Prompt")
    for mode in dict.fromkeys(args.schedule.split(",")):
        run(inputs, mode, boundaries)
    results = []
    for mode in args.schedule.split(","):
        row = run(inputs, mode, boundaries)
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "cuvs": cuvs.__version__,
                "fixture_hashes": [
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in args.fixtures
                ],
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
