"""Explicit bounded TP1 Oasis Decode pilot on the existing PVD bootstrap.

The pilot retains full initial V->D admission. A single private Prompt pass
seeds EAGLE features and root Q; it is charged as initialization and is never
called during Decode/refresh. Steady-state selection and misses use V/CAGRA
and native per-layer CPU receive -> cache -> H2D banks.
"""

import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import torch

from sglang.srt.disaggregation.pvd.cuda_target_probe import CUDAQwen2TargetProbe
from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerLookahead
from sglang.srt.disaggregation.pvd.oasis_request import OasisRequestDecoder
from sglang.srt.disaggregation.pvd.oasis_scheduler import OasisSchedulerBinding
from sglang.srt.disaggregation.pvd.oasis_sglang import SGLangQwenPairedDecode
from sglang.srt.disaggregation.pvd.oasis_transport import OasisLayerTransport
from sglang.srt.disaggregation.pvd.prediction import CommittedPrefix, DraftPrediction, ProbeConfig
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

logger = logging.getLogger(__name__)
_QUARANTINE = []


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eagle(config, target, device):
    from safetensors.torch import load_file

    source, checkpoint = Path(config["eagle_source"]), Path(config["eagle_checkpoint"])
    metadata = json.loads(Path(config["eagle_manifest"]).read_text())
    if (sha256(checkpoint / "model.safetensors") != metadata["weights_sha256"]
            or subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                text=True).strip() != metadata["eagle_commit"]):
        raise ValueError("pinned EAGLE3 source/checkpoint identity changed")
    # A pinned source checkout is explicit startup code, not input supplied by
    # a request. Import only after its recorded author revision is verified.
    sys.path.insert(0, str(source))
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig
    cfg = EConfig.from_pretrained(str(checkpoint), local_files_only=True)
    cfg.rope_scaling = metadata["config"].get("rope_scaling")
    cfg.rope_theta = metadata["config"]["rope_theta"]
    if cfg.hidden_size != target.config.hidden_size:
        raise ValueError("EAGLE3 target hidden dimension differs")
    draft = Model(cfg, bias=False, top_k=1, depth=1, total_tokens=2).half()
    status = draft.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=False)
    if status.missing_keys != ["embed_tokens.weight"] or status.unexpected_keys:
        raise ValueError("EAGLE3 checkpoint shape differs")
    draft.embed_tokens = target.model.embed_tokens
    draft = draft.to(device).eval()
    mapping = draft.d2t + torch.arange(cfg.draft_vocab_size, device=device)
    if (not torch.equal(mapping, draft.t2d.nonzero().flatten())
            or mapping.max().item() >= target.config.vocab_size):
        raise ValueError("EAGLE3 reduced-vocabulary mapping differs")
    return draft, mapping


class OasisServingOwner(OasisRequestDecoder):
    def __init__(self, *args, transport, resources, reservation, **kwargs):
        super().__init__(*args, **kwargs)
        self.transport, self.resources, self.reservation = transport, resources, reservation
        self._retired, self._retirement_error = False, ()

    def close(self):
        if self._retired:
            return ()
        if self._retirement_error:
            return self._retirement_error
        errors = super().close()
        if errors:
            # Keep transport caches/draft state and charge until worker exit.
            return errors
        try:
            self.transport.close()
        except BaseException as error:
            self._retirement_error = (error,)
            return self._retirement_error
        self.predict_one = self.fetch_layer = None
        self.resources.budget.release(self.reservation)
        self.resources.owners.remove(self)
        self._retired = True
        logger.info("PVD Oasis retired rid=%s cache_rows=%s layer_wait_ms=%.3f",
            self.request_id, sum(t["remote_rows"] for t in self.transport.trace),
            sum(t["consumer_wait_seconds"] for t in self.pipeline.trace) * 1000)
        logger.info("PVD Oasis trace rid=%s data=%s", self.request_id,
            json.dumps(dict(layers=self.pipeline.trace, transport=self.transport.trace,
                io=self.transport.io_snapshot(),
                attention_workspace=self.decoder.workspace.snapshot()
                    if self.decoder.workspace is not None else None), separators=(",", ":")))
        return ()


class OasisResources:
    def __init__(self, scheduler, config):
        self.scheduler, self.config = scheduler, config
        self.runner = scheduler.tp_worker.model_runner
        self.manager = scheduler.disagg_decode_prealloc_queue.kv_manager
        self.device = torch.device(f"cuda:{self.runner.gpu_id}")
        self.lock = threading.RLock()
        self.budget = TransferBudget(config["request_budget_bytes"], 1)
        self.probe_budget = TransferBudget(config["bootstrap_budget_bytes"], 1)
        self.owners = []
        # Keep partial startup owners alive if native initialization fails.
        _QUARANTINE.append(self)
        SGLangQwenPairedDecode(self.runner, execution_lock=self.lock)
        self.draft, self.mapping = load_eagle(config, self.runner.model, self.device)
        self.probe = CUDAQwen2TargetProbe(self.runner,
            ProbeConfig(config["vector_space"], tuple(range(28)), 0, 28),
            device=self.device, execution_lock=self.lock,
            target_model_id=config["vector_space"], max_tokens=config["max_sequence_tokens"],
            max_predict_tokens=1, transient_bytes_bound=config["bootstrap_transient_bytes"],
            budget=self.probe_budget)

    @torch.inference_mode()
    def prepare(self, req, receipt):
        cfg = self.config
        steps = req.sampling_params.max_new_tokens - len(req.output_ids)
        if (len(receipt.prompt) + steps + 2 > cfg["max_sequence_tokens"]
                or not 1 <= steps <= cfg["max_decode_steps"] or self.owners
                or req.lora_id is not None or req.return_hidden_states
                or getattr(req.sampling_params, "repetition_penalty", 1.0) <= 0):
            raise ValueError("request exceeds bounded Oasis admission")
        started = time.perf_counter()
        incarnation, reservation = str(uuid.uuid4()), f"oasis:{uuid.uuid4().hex}"
        # Explicit upper charge: monotonic CPU cache plus three sparse banks,
        # actual history, EAGLE cached state/features and declared scratch.
        rows = len(receipt.prompt)
        charge = (rows * 28 * 4 * (128 * 2 * 2 + 1)
            + 3 * 28 * 4 * cfg["capacity"] * (128 * 2 * 2 + 1)
            + steps * 28 * 4 * 128 * 2 * 2
            + (rows + steps) * 3584 * 12 + cfg["request_scratch_bytes"])
        self.budget.reserve(reservation, charge, 1)
        pending = dict(resources=self, reservation=reservation)
        _QUARANTINE.append(pending)
        features, queries, hooks = {}, None, []
        transport = bootstrap = owner = None
        try:
            selected = self.manager.control.submit(self.manager.client_for(req).selected_shard_routes(
                receipt.key)).result(timeout=cfg["timeout_seconds"])
            if (selected.manifest.key != receipt.key or selected.manifest.prompt_token_count != rows
                    or selected.manifest.layout.num_layers != 28):
                raise ValueError("selected V Entry differs from initial receipt")
            # One-time actual Prompt pass. Clone only Prompt features, excluding
            # the known root row, and capture only that root's true target Q.
            for layer in (1, 13, 24):
                def capture(module, inputs, output, layer=layer):
                    features[layer] = (output[0] + output[1])[:rows].clone()
                hooks.append(self.runner.model.model.layers[layer].register_forward_hook(capture))
            prefix = CommittedPrefix(req.rid, receipt.prompt, 0, incarnation)
            with self.probe.branch():
                found = self.probe.capture(prefix, DraftPrediction(req.rid, incarnation, receipt.outputs))
                queries = [item.vectors[0].clone() for item in found]
                torch.cuda.current_stream(self.device).synchronize()
            for hook in hooks:
                hook.remove()
            hooks.clear()
            if set(features) != {1, 13, 24}:
                raise RuntimeError("EAGLE actual feature seed missing")
            seed = torch.cat([features[l] for l in (1, 13, 24)], dim=-1).unsqueeze(0)
            state = dict(cache=None, seed=seed, seen=list(receipt.prompt) + list(receipt.outputs))
            pending["draft_state"] = state
            penalty = getattr(req.sampling_params, "repetition_penalty", 1.0)

            def propose(current, previous_features):
                tick = time.perf_counter()
                if previous_features is None:
                    feature = state["seed"]
                    inputs = torch.tensor([state["seen"][1:]], device=self.device)
                else:
                    feature = previous_features
                    state["seen"].append(current)
                    inputs = torch.tensor([[current]], device=self.device)
                hidden, cache = self.draft(feature, input_ids=inputs,
                    past_key_values=state["cache"], use_cache=True)
                scores = self.draft.lm_head(self.draft.norm(hidden[:, -1])).float()
                if not torch.isfinite(scores).all().item():
                    raise RuntimeError("nonfinite EAGLE proposal")
                repeated = torch.isin(self.mapping, torch.tensor(list(set(state["seen"])), device=self.device))
                values = scores[:, repeated]
                scores[:, repeated] = torch.where(values < 0, values * penalty, values / penalty)
                candidate = int(self.mapping[scores.argmax(-1)].item())
                state["cache"], state["seed"] = cache, None
                logger.info("PVD Oasis draft rid=%s ms=%.3f", req.rid, (time.perf_counter() - tick) * 1000)
                return candidate

            primed = [propose(receipt.outputs[-1], None)]

            def predict_one(current, previous_features):
                if previous_features is None:
                    if current != receipt.outputs[-1] or not primed:
                        raise RuntimeError("invalid initial EAGLE root")
                    return primed.pop()
                return propose(current, previous_features)

            transport = OasisLayerTransport(self.manager, selected, request_id=req.rid,
                incarnation=incarnation, device=self.device, vector_space=cfg["vector_space"],
                capacity=cfg["capacity"], max_new=cfg["max_new"], top_k=cfg["top_k"],
                timeout=cfg["timeout_seconds"], reuse_io=cfg.get("reuse_io", False),
                combine_reserve_start=cfg.get("combine_reserve_start", False),
                reuse_receive_slots=cfg.get("reuse_receive_slots", False), workers=cfg["workers"],
                gpu_receive_to_bank=cfg.get('gpu_receive_to_bank', False),
                staged_transport=cfg.get('staged_transport', False),
                sort_missing_tokens=cfg.get('sort_missing_tokens', False),
                batched_bank_install=cfg.get('batched_bank_install', False),
                batched_cache_install=cfg.get('batched_cache_install', False),
                ready_before_cleanup=cfg.get('ready_before_cleanup', False),
                binary_queries=cfg.get('binary_queries', False),
                fused_search_delivery=cfg.get('fused_search_delivery',False),
                compact_cache_snapshots=cfg.get('compact_cache_snapshots',False),
                reuse_pinned_scratch=cfg.get('reuse_pinned_scratch',False),
                event_bank_ready=cfg.get('event_bank_ready',False),
                binary_control_channel=cfg.get('binary_control_channel',False),
                fused_zero_miss_proof=cfg.get('fused_zero_miss_proof',False),
                async_layer_jobs=cfg.get('async_layer_jobs',False),
                parallel_owned_cleanup=cfg.get('parallel_owned_cleanup',False),
                install_scratch_bytes=cfg['request_scratch_bytes'],
                backup_budget_bytes=cfg['request_scratch_bytes'])
            pending["transport"] = transport
            bootstrap = LayerLookahead(req.rid, incarnation, layers=28,
                workers=cfg["workers"], timeout=cfg["timeout_seconds"])
            pending["bootstrap"] = bootstrap
            for layer, query in enumerate(queries):
                bootstrap.publish(0, layer, transport.job(query, None, bootstrap=True))
            banks = [bootstrap.consume(0, layer) for layer in range(28)]
            errors = bootstrap.close()
            if errors:
                raise RuntimeError("bootstrap layer transfer failed")
            workspace = self._attention_workspace(steps)
            pending['attention_workspace'] = workspace
            owner = OasisServingOwner(req.rid, incarnation,
                decoder=SGLangQwenPairedDecode(self.runner, execution_lock=self.lock,
                    workspace=workspace),
                initial_banks=banks, predict_one=predict_one, fetch_layer=transport.job,
                current_token=receipt.outputs[-1], position=rows, max_steps=steps,
                workers=cfg["workers"], timeout=cfg["timeout_seconds"],
                transport=transport, resources=self, reservation=reservation,
                overlap=cfg["overlap"])
            self.owners.append(owner)
            pending["owner"] = owner
            _QUARANTINE.remove(pending)
            logger.info("PVD Oasis initialized rid=%s prompt_tokens=%d seconds=%.6f "
                "initial_full_kv=true bootstrap_prefix_passes=1",
                req.rid, rows, time.perf_counter() - started)
            return owner
        except BaseException:
            if bootstrap is not None:
                bootstrap.close()
            # Keep partial GPU state, unknown registrations and budget until
            # process exit. No default probe/ordinary-forward fallback.
            pending.update(features=features, queries=queries)
            raise
        finally:
            for hook in hooks:
                hook.remove()

    def _attention_workspace(self, steps):
        if not self.config.get('attention_workspace', False):
            return None
        from sglang.srt.disaggregation.pvd.oasis_attention_workspace import PairedAttentionWorkspace
        return PairedAttentionWorkspace(device=self.device, dtype=torch.float16,
            q_heads=28, kv_heads=4, head_dim=128, max_bank_rows=self.config['capacity'],
            max_history=steps, max_bytes=self.config['request_scratch_bytes'])


def maybe_install_oasis(scheduler):
    path = getattr(scheduler.server_args, "pvd_oasis_config", None)
    if not path:
        return None
    cfg = json.loads(Path(path).read_text())
    fields = {"eagle_source", "eagle_checkpoint", "eagle_manifest", "vector_space", "capacity",
        "max_new", "top_k", "workers", "timeout_seconds", "max_sequence_tokens", "max_decode_steps",
        "request_budget_bytes", "request_scratch_bytes", "bootstrap_budget_bytes", "bootstrap_transient_bytes", "overlap"}
    if set(cfg) - {"reuse_io", "combine_reserve_start", "reuse_receive_slots", 'gpu_receive_to_bank', 'staged_transport', 'attention_workspace', 'sort_missing_tokens', 'batched_bank_install', 'batched_cache_install', 'ready_before_cleanup', 'binary_queries','fused_search_delivery','compact_cache_snapshots','reuse_pinned_scratch','event_bank_ready','binary_control_channel','fused_zero_miss_proof','async_layer_jobs','parallel_owned_cleanup'} != fields:
        raise ValueError("Oasis config must contain exactly the documented bounds and pins")
    cfg.setdefault("reuse_io", False)
    cfg.setdefault("combine_reserve_start", False)
    cfg.setdefault("reuse_receive_slots", False)
    cfg.setdefault('gpu_receive_to_bank', False)
    cfg.setdefault('staged_transport', False)
    cfg.setdefault('attention_workspace', False)
    cfg.setdefault('sort_missing_tokens', False)
    cfg.setdefault('batched_bank_install', False)
    cfg.setdefault('batched_cache_install', False)
    cfg.setdefault('ready_before_cleanup', False)
    cfg.setdefault('binary_queries', False)
    cfg.setdefault('fused_search_delivery',False)
    cfg.setdefault('compact_cache_snapshots',False)
    cfg.setdefault('reuse_pinned_scratch',False)
    cfg.setdefault('event_bank_ready',False)
    cfg.setdefault('binary_control_channel',False)
    cfg.setdefault('fused_zero_miss_proof',False)
    cfg.setdefault('async_layer_jobs',False)
    cfg.setdefault('parallel_owned_cleanup',False)
    if cfg['parallel_owned_cleanup'] and not cfg['ready_before_cleanup']:
        raise ValueError('parallel rank cleanup requires owned READY cleanup')
    if cfg['binary_control_channel'] and (not cfg['fused_search_delivery'] or not cfg['binary_queries'] or cfg['reuse_io']):
        raise ValueError('binary channel requires fused binary Q and owned search clients')
    if cfg['compact_cache_snapshots'] and not cfg['fused_search_delivery']:
        raise ValueError('compact cache snapshots require fused delivery')
    if cfg['fused_zero_miss_proof'] and not cfg['fused_search_delivery']:
        raise ValueError('zero-miss proof requires fused delivery')
    if cfg['async_layer_jobs']:
        from sglang.srt.disaggregation.pvd.oasis_async_jobs import async_tensor_bound
        if (not cfg['binary_control_channel'] or not cfg['ready_before_cleanup']
                or cfg['request_scratch_bytes'] < async_tensor_bound(cfg['capacity'])):
            raise ValueError('async layer jobs require binary channel, READY cleanup and admitted scratch')
    for name in fields - {"eagle_source", "eagle_checkpoint", "eagle_manifest", "vector_space", "max_new", "overlap"}:
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f"positive integer Oasis {name} required")
    if (type(cfg["overlap"]) is not bool or type(cfg["reuse_io"]) is not bool
            or type(cfg["combine_reserve_start"]) is not bool
            or type(cfg["reuse_receive_slots"]) is not bool
            or type(cfg['gpu_receive_to_bank']) is not bool
            or type(cfg['staged_transport']) is not bool
            or type(cfg['attention_workspace']) is not bool
            or type(cfg['sort_missing_tokens']) is not bool
            or type(cfg['batched_bank_install']) is not bool
            or type(cfg['batched_cache_install']) is not bool
            or type(cfg['ready_before_cleanup']) is not bool
            or type(cfg['binary_queries']) is not bool
            or type(cfg['fused_search_delivery']) is not bool
            or type(cfg['compact_cache_snapshots']) is not bool
            or type(cfg['reuse_pinned_scratch']) is not bool or type(cfg['event_bank_ready']) is not bool
            or type(cfg['binary_control_channel']) is not bool
            or type(cfg['fused_zero_miss_proof']) is not bool
            or type(cfg['async_layer_jobs']) is not bool
            or type(cfg['parallel_owned_cleanup']) is not bool
            or type(cfg["max_new"]) is not int or not 0 <= cfg["max_new"] <= cfg["capacity"]
            or cfg["workers"] > 4 or cfg["capacity"] > 2048 or cfg["top_k"] > 512
            or cfg["max_sequence_tokens"] > scheduler.tp_worker.model_runner.model_config.context_len):
        raise ValueError("unsupported Oasis bounds")
    if cfg['reuse_pinned_scratch'] or cfg['event_bank_ready']:
        from sglang.srt.disaggregation.pvd.oasis_pinned_scratch import scratch_bytes
        if (cfg['workers'] != 2 or cfg['capacity'] > 32
                or any(cfg[name] for name in ('staged_transport','gpu_receive_to_bank',
                       'batched_bank_install','batched_cache_install','attention_workspace'))
                or (cfg['ready_before_cleanup'] and not cfg['fused_search_delivery'])
                or cfg['workers']*scratch_bytes(cfg['capacity']) > cfg['request_scratch_bytes']
                or (cfg['event_bank_ready'] and not cfg['reuse_pinned_scratch'])):
            raise ValueError('pinned/event mode requires bounded ordinary owned scratch')
    if cfg['ready_before_cleanup'] and not cfg['fused_search_delivery'] and (cfg['workers'] != 2 or any(cfg[name] for name in
            ('reuse_io', 'combine_reserve_start', 'reuse_receive_slots', 'gpu_receive_to_bank',
             'staged_transport', 'attention_workspace', 'batched_bank_install', 'batched_cache_install'))):
        raise ValueError('READY cleanup requires isolated two-worker baseline')
    if cfg['fused_search_delivery'] and (cfg['workers'] != 2 or cfg['capacity'] > 32 or any(cfg[name] for name in
            ('reuse_io','combine_reserve_start',
             'gpu_receive_to_bank','staged_transport','attention_workspace','sort_missing_tokens',
             'batched_bank_install','batched_cache_install'))):
        raise ValueError('fused selection requires isolated bounded two-worker baseline')
    if cfg['binary_queries'] and (cfg['workers'] != 2 or any(cfg[name] for name in
            ('reuse_io','combine_reserve_start',
             'gpu_receive_to_bank','staged_transport','attention_workspace','batched_bank_install','batched_cache_install'))
            or ((cfg['ready_before_cleanup'] or cfg['reuse_receive_slots']) and not cfg['fused_search_delivery'])):
        raise ValueError('binary Q requires isolated two-worker baseline')
    if cfg['batched_bank_install']:
        from sglang.srt.disaggregation.pvd.oasis_bank_install import install_tensor_bound
        if (cfg['gpu_receive_to_bank'] or cfg['staged_transport'] or cfg['attention_workspace']
                or cfg['workers'] * install_tensor_bound(cfg['capacity']) > cfg['request_scratch_bytes']):
            raise ValueError('batched install requires isolated mode and admitted request scratch')
    if cfg['batched_cache_install']:
        from sglang.srt.disaggregation.pvd.oasis_cache_install import cache_install_tensor_bound
        if (cfg['gpu_receive_to_bank'] or cfg['staged_transport'] or cfg['attention_workspace']
                or cfg['batched_bank_install']
                or cfg['workers'] * cache_install_tensor_bound(cfg['capacity']) > cfg['request_scratch_bytes']):
            raise ValueError('batched cache install requires isolated mode and admitted request scratch')
    binding = OasisSchedulerBinding(scheduler, lambda req, receipt: None)
    resources = OasisResources(scheduler, cfg)
    binding.prepare_request = resources.prepare
    scheduler.pvd_oasis_resources = resources
    return binding
