"""Explicit, spawn-safe capture and READY-bank replay of the live Oasis path.

This diagnostic is installed by the accompanying sitecustomize loader. It
captures the two selected real requests, drains their native owners, and uses
the same process, SGLang runner, target weights, EAGLE proposal closure and
ordinary runner.sample for replay. It never loads an HF target or supplies
recorded tokens as sampler outputs. Capturing, reset, warmups, validation and
serialization are excluded from measured replay intervals.

The foreground scope retains OasisRequestDecoder, query clone/event setup,
two executor workers, exact layer tickets and resident handoffs. Background
callbacks return already resident, captured next-step banks; network, D2H,
native receive and V work are absent. This is a counterfactual with those
resources removed, not an estimate obtained by subtracting live wait times.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback
from types import SimpleNamespace

LOGGER = logging.getLogger(__name__)
SCHEMA = "pvd-oasis-ready-replay-v1"
_CONTROLLERS = {}
_QUARANTINE = []
_INSTALLED = False


def prompt_ids_sha256(ids):
    """Hash exact IDs using compact JSON, without a trailing newline."""
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode("utf-8")).hexdigest()


def save_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked_artifact_path(value):
    root = Path(__file__).resolve().parents[1]
    path = Path(value).resolve()
    if not path.is_relative_to(root / "artifacts"):
        raise ValueError("diagnostic outputs must be inside this checkout's artifacts directory")
    return path


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"capture_directory", "cases", "expected_prompt_tokens", "output_tokens",
                "workers", "warmup_replays", "measured_replays", "diagnostic_gpu_budget_bytes",
                "expected_prompt_ids_sha256"}
    if set(config) != required:
        raise ValueError("exact documented diagnostic fields required")
    if (config["cases"] != [99401, 99402] or config["expected_prompt_tokens"] != 2159
            or config["output_tokens"] != 16 or config["workers"] != 2
            or config["warmup_replays"] != 2 or config["measured_replays"] != 3
            or config["diagnostic_gpu_budget_bytes"] != 256 << 20):
        raise ValueError("bounded two-case, two-worker diagnostic policy required")
    hashes = config["expected_prompt_ids_sha256"]
    if (set(hashes) != {"99401", "99402"}
            or any(not re.fullmatch("[0-9a-f]{64}", value) for value in hashes.values())
            or len(set(hashes.values())) != 2):
        raise ValueError("two distinct exact Prompt-ID hashes required")
    config["capture_directory"] = checked_artifact_path(config["capture_directory"])
    return config


def tree_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tree_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tree_tensors(item)
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from tree_tensors(getattr(value, field.name))


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if value is None or type(value) in (int, float, str, bool):
        return value
    raise TypeError(f"unsupported trajectory value: {type(value).__name__}")


def clone_tree(value):
    """Independent mutable EAGLE state without shared tensor storage."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if value is None or type(value) in (int, float, str, bool):
        return value
    raise TypeError(f"unsupported mutable EAGLE state: {type(value).__name__}")


def snapshot_sampling(fb):
    """Freeze the supported ordinary greedy metadata; do not synthesize logits."""
    import torch
    info = fb.sampling_info
    orch = info.penalizer_orchestrator
    if (not info.is_all_greedy or info.grammars or info.has_custom_logit_processor
            or info.custom_logit_processor is not None or info.vocab_mask is not None
            or info.logit_bias is not None or info.acc_additive_penalties is not None
            or info.acc_scaling_penalties is not None or orch is not None and orch.is_required
            or fb.return_logprob or fb.ngram_embedding_info is not None):
        raise ValueError("diagnostic supports only the actual unmodified greedy/default-penalty pilot")
    # Inactive orchestrators have no numerical effect, and the live result
    # processor releases them. Fresh replay metadata must not borrow one.
    values = {field.name: getattr(info, field.name) for field in dataclasses.fields(info)}
    values["penalizer_orchestrator"] = None
    for name, value in list(values.items()):
        if isinstance(value, torch.Tensor):
            values[name] = value.detach().clone()
    frozen = type(info)(**values)
    return SimpleNamespace(sampling_info=frozen, positions=fb.positions.detach().clone(),
        forward_mode=fb.forward_mode, return_logprob=fb.return_logprob,
        top_logprobs_nums=copy.deepcopy(fb.top_logprobs_nums),
        token_ids_logprobs=copy.deepcopy(fb.token_ids_logprobs),
        ngram_embedding_info=None, out_cache_loc=fb.out_cache_loc.detach().clone(),
        req_pool_indices=fb.req_pool_indices.detach().clone())


class Capture:
    def __init__(self, controller, resources, req, receipt, case):
        from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget
        self.controller, self.resources, self.req, self.receipt, self.case = controller, resources, req, receipt, case
        self.steps, self.seed_call, self.owner = [], None, None
        self.predict_one = self.propose = self.draft_state = self.primed = None
        self.retained_storages, self.retained_bytes = {}, 0
        self.budget = TransferBudget(controller.config["diagnostic_gpu_budget_bytes"], 1)
        self.reservation = f"diagnostic-ready:{case}:{req.rid}"
        # Explicit separate reservation covers retained live owners plus replay
        # EAGLE/actual-history/sampler/private-row scratch. Default serving's
        # request budget is not borrowed or silently enlarged.
        self.budget.reserve(self.reservation, controller.config["diagnostic_gpu_budget_bytes"], 1)
        self.scratch_bound = 64 << 20
        self.initial_outputs = tuple(req.output_ids)
        self.closed = False
        self.replay_failures = []
        self.session = resources.manager.decode_sessions[resources.manager.key_for(req)]
        self.old_receive_guard = self.session.receive_guard

    def retain(self, value):
        for tensor in tree_tensors(value):
            if tensor.device.type != "cuda":
                continue
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if identity not in self.retained_storages:
                needed = self.retained_bytes + storage.nbytes()
                if needed + self.scratch_bound > self.controller.config["diagnostic_gpu_budget_bytes"]:
                    _QUARANTINE.append(self)
                    raise RuntimeError("diagnostic retained GPU owners exceed separately charged bound")
                self.retained_storages[identity] = tensor
                self.retained_bytes = needed

    def bind(self, owner):
        self.owner = owner
        self.predict_one = owner.predict_one
        closure = inspect.getclosurevars(owner.predict_one).nonlocals
        if set(closure) != {"primed", "propose", "receipt"}:
            raise ValueError("validated EAGLE prediction closure changed")
        self.propose, self.primed = closure["propose"], closure["primed"]
        self.draft_state = inspect.getclosurevars(self.propose).nonlocals["state"]
        if self.seed_call is None or self.seed_call["past_key_values"] is not None or len(self.primed) != 1:
            raise ValueError("one actual Prompt seed and initial EAGLE proposal required")
        self.initial_predicted = self.primed[0]
        self.initial_seen = tuple(self.draft_state["seen"])
        # The real EAGLE cache may be mutated by later proposals. Freeze it at
        # the post-prime boundary, while capture performance is out of scope.
        self.initial_cache = clone_tree(self.draft_state["cache"])
        self.retain(self.initial_cache)
        original = owner.decoder.step
        self.original_target_step = original
        self.live_max_steps = owner.max_steps

        def target_step(current, predicted, position, banks, **kwargs):
            row = dict(step=len(self.steps), current=current, predicted=predicted,
                       position=position, banks=[None] * 28, sample=None,
                       eagle_seen=list(self.draft_state["seen"]))
            self.steps.append(row)

            def bank(layer):
                value = banks(layer) if callable(banks) else banks[layer]
                if row["banks"][layer] is not None:
                    raise RuntimeError("a live layer bank was consumed twice")
                self.retain((value.keys, value.values, value.valid))
                row["banks"][layer] = value
                return value

            logits, features = original(current, predicted, position, bank, **kwargs)
            self.retain((logits, features))
            row.update(logits=logits, features=features,
                       actual_kv=tuple(history[-1] for history in owner.decoder.generated))
            self.retain(row["actual_kv"])
            return logits, features

        owner.decoder.step = target_step

    def sample(self, original, logits, fb):
        row = self.steps[-1]
        if row["sample"] is not None:
            raise RuntimeError("one ordinary sampler call per live target step required")
        row["sample"] = snapshot_sampling(fb)
        self.retain(vars(row["sample"]))
        tokens = original(logits, fb)
        row["sampled_tensor"] = tokens.detach().clone()
        self.retain(row["sampled_tensor"])
        return tokens

    def validate_live(self):
        import torch
        if (len(self.steps) != 15 or len(self.req.output_ids) != 16
                or self.owner.step != 15 or len(self.initial_outputs) != 1):
            raise RuntimeError("capture must include all 16 actual outputs and 15 formal Decode steps")
        outputs = tuple(self.req.output_ids)
        for index, row in enumerate(self.steps):
            if (row["current"] != outputs[index] or row["position"] != 2159 + index
                    or int(row["sampled_tensor"].item()) != outputs[index + 1]
                    or any(bank is None for bank in row["banks"]) or row["sample"] is None
                    or len(row["actual_kv"]) != 28):
                raise RuntimeError("captured bank/token/formal-step trajectory is incomplete")
            for bank in row["banks"]:
                if bank.completion is not None:
                    bank.completion.synchronize()
                if (bank.keys.device != self.resources.device or bank.values.shape != bank.keys.shape
                        or bank.valid.shape != bank.keys.shape[:2] or bank.valid.dtype != torch.bool
                        or len(bank.ids) != 4 or max(map(len, bank.ids)) != bank.keys.shape[1]):
                    raise RuntimeError("captured GPU sparse bank layout changed")
        return outputs

    def reset_draft(self):
        # Live initialization already performed EAGLE prefill before target
        # step 0. Restore that exact post-prime boundary with independent cache
        # storage; all subsequent proposals execute the original real closure.
        started = time.perf_counter()
        self.draft_state.clear()
        self.draft_state.update(cache=clone_tree(self.initial_cache), seed=None, seen=list(self.initial_seen))
        self.primed[:] = [self.initial_predicted]
        return (time.perf_counter() - started) * 1000

    def replay(self, *, profile, repetition, warmup):
        import torch
        with torch.inference_mode():
            return self._replay(profile=profile, repetition=repetition, warmup=warmup)

    def _replay(self, *, profile, repetition, warmup):
        import torch
        from sglang.srt.disaggregation.pvd.draft_forward_adapter import PrivatePoolAllocator
        from sglang.srt.disaggregation.pvd.oasis_pipeline import LayerReply
        from sglang.srt.disaggregation.pvd.oasis_request import OasisRequestDecoder
        from sglang.srt.disaggregation.pvd.oasis_sglang import SGLangQwenPairedDecode
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        runner, device = self.resources.runner, self.resources.device
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("entire EAGLE/target/formal-write/sampler replay requires inference mode")
        reset_ms = self.reset_draft()
        allocator = PrivatePoolAllocator(runner.req_to_token_pool, runner.token_to_kv_pool_allocator)
        slot, rows, owner = None, None, None
        retained, stages = [], []
        scratch_owners = dict(capture=self, allocator=allocator, rows=rows)
        _QUARANTINE.append(scratch_owners)
        slot = allocator.alloc_request()
        prior_mapping = runner.req_to_token_pool.req_to_token[slot].clone()
        rows = allocator.alloc_kv(15)
        scratch_owners.update(rows=rows, slot=slot, prior_mapping=prior_mapping)
        allocator.write_mapping(slot, 2159, rows)
        pool = runner.token_to_kv_pool_allocator.get_kvcache()

        def fetch_layer(query, resident):
            # Match foreground transport.job query ownership and completion
            # marker. Background replay neither reads query nor touches native IO.
            query = query.detach().clone()
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))

            def ready(ticket):
                if callable(resident):
                    resident()
                retained.append((query, event))
                # Preserve the live max_steps/publication policy. If admission
                # permits an unused terminal prefetch, retain that publication
                # and drain it; its last captured bank is never consumed by a
                # target step or counted as verified next-step retrieval.
                bank_step = min(ticket.step + 1, len(self.steps) - 1)
                return LayerReply(ticket, self.steps[bank_step]["banks"][ticket.layer])
            return ready

        decoder = SGLangQwenPairedDecode(runner, execution_lock=self.resources.lock)
        owner = OasisRequestDecoder(self.req.rid, "ready-replay", decoder=decoder,
            initial_banks=self.steps[0]["banks"], predict_one=self.predict_one,
            fetch_layer=fetch_layer, current_token=self.initial_outputs[-1], position=2159,
            max_steps=self.live_max_steps, workers=2, timeout=60, overlap=True)
        scratch_owners["owner"] = owner
        if (any(decoder.generated) or decoder._next_position is not None
                or decoder.owner.active is not None or decoder.owner.quarantined):
            raise RuntimeError("fresh paired decoder state required for each replay")
        outputs, replay_tensors, fb_rows = [], [], []
        for index, row in enumerate(self.steps):
            fb = copy.copy(row["sample"])
            fb.sampling_info = copy.copy(fb.sampling_info)
            fb.out_cache_loc = torch.tensor([rows[index]], dtype=row["sample"].out_cache_loc.dtype, device=device)
            fb.req_pool_indices = torch.tensor([slot], dtype=row["sample"].req_pool_indices.dtype, device=device)
            fb_rows.append(fb)

        class Stage:
            def __init__(stage, name, step):
                stage.item = dict(name=name, step=step)
            def __enter__(stage):
                if profile:
                    stage.events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    stage.events[0].record(torch.cuda.current_stream(device))
                stage.started = time.perf_counter()
            def __exit__(stage, *error):
                stage.item["wall_ms"] = (time.perf_counter() - stage.started) * 1000
                if profile:
                    stage.events[1].record(torch.cuda.current_stream(device))
                    stage.item["events"] = stage.events
                stages.append(stage.item)

        original_predict = owner.predict_one
        original_target = decoder.step
        original_fence = decoder.owner._synchronize

        def predict(current, features):
            with Stage("eagle_prediction", owner.step):
                return original_predict(current, features)

        def target(*args, **kwargs):
            with Stage("paired_target", owner.step):
                return original_target(*args, **kwargs)

        def target_fence():
            with Stage("paired_owner_fence", owner.step):
                return original_fence()

        owner.predict_one, decoder.step, decoder.owner._synchronize = predict, target, target_fence
        try:
            torch.cuda.current_stream(device).synchronize()
            started_unix = time.time()
            started = time.perf_counter()
            for index, (row, fb) in enumerate(zip(self.steps, fb_rows, strict=True)):
                step_started = time.perf_counter()
                logits = owner.forward(owner.current_token, owner.position)
                if owner.predicted != row["predicted"] or self.draft_state["seen"] != row["eagle_seen"]:
                    raise RuntimeError("EAGLE proposal or committed-prefix state diverged; no teacher forcing")
                with Stage("formal_kv_writes_28", index):
                    for layer, history in enumerate(decoder.generated):
                        key, value = history[-1]
                        pool.set_kv_buffer(runner.model.model.layers[layer].self_attn.attn,
                            fb.out_cache_loc, key.transpose(0, 1), value.transpose(0, 1))
                with Stage("formal_kv_fence", index):
                    torch.cuda.current_stream(device).synchronize()
                with Stage("ordinary_sampler_commit", index):
                    sampled = self.controller.original_samples[id(runner)](
                        LogitsProcessorOutput(next_token_logits=logits.float()), fb)
                    actual = int(sampled.item())
                    if actual != int(self.req.output_ids[index + 1]):
                        raise RuntimeError("ordinary sampler diverged; no teacher forcing")
                    owner.actual_committed(actual)
                # Reference retention and Python bookkeeping are part of this
                # small diagnostic wall overhead, never hidden as CUDA compute.
                outputs.append(actual)
                replay_tensors.append((logits, owner.features,
                    tuple(history[-1] for history in decoder.generated), owner.predicted))
                stages.append(dict(name="foreground_total", step=index,
                    wall_ms=(time.perf_counter() - step_started) * 1000))
            torch.cuda.current_stream(device).synchronize()
            total_ms = (time.perf_counter() - started) * 1000
            finished_unix = time.time()
            if (owner.step != 15 or any(len(history) != 15 for history in decoder.generated)
                    or decoder._next_position != self.steps[-1]["position"] + 1):
                raise RuntimeError("replay omitted actual history/position commits")
            trace = list(owner.pipeline.trace)
            publication_count = sum(step + 1 for step in owner.pipeline._published)
            errors = owner.close()
            if errors or decoder.owner.active is not None or decoder.owner.quarantined:
                raise RuntimeError("replay owner could not prove completion")
            # All validation is outside timing and happens before another trial.
            if outputs != list(self.req.output_ids)[1:]:
                raise RuntimeError("ordinary sampler diverged from actual live outputs")
            for row, (logits, features, actual_kv, predicted) in zip(self.steps, replay_tensors, strict=True):
                if predicted != row["predicted"] or not torch.equal(logits, row["logits"]) or not torch.equal(features, row["features"]):
                    raise RuntimeError("target/EAGLE replay diverged from captured paired trajectory")
                for expected, actual in zip(row["actual_kv"], actual_kv, strict=True):
                    if any(not torch.equal(a, b) for a, b in zip(expected, actual, strict=True)):
                        raise RuntimeError("actual committed target KV changed during replay")
            for layer in range(28):
                for index, row in enumerate(self.steps):
                    key, value = row["actual_kv"][layer]
                    if (not torch.equal(pool.get_key_buffer(layer)[rows[index]], key[:, 0])
                            or not torch.equal(pool.get_value_buffer(layer)[rows[index]], value[:, 0])):
                        raise RuntimeError("private formal KV writes differ from actual committed rows")
            for stage in stages:
                if "events" in stage:
                    start_event, end_event = stage.pop("events")
                    stage["gpu_event_ms"] = start_event.elapsed_time(end_event)
            return dict(repetition=repetition, warmup=warmup, profiled=profile,
                started_unix=started_unix, finished_unix=finished_unix,
                total_wall_ms=total_ms, steady_token_wall_mean_ms=sum(t["wall_ms"] for t in stages
                    if t["name"] == "foreground_total" and t["step"] > 0) / 14,
                eagle_state_reset_wall_ms=reset_ms, inference_mode=True, stages=stages, layers=trace,
                all_banks_preloaded=True, all_layer_callbacks_ready_before_consume=all(t["ready_before_consume"] for t in trace),
                consumed_layers=len(trace), ordinary_sampled_outputs=outputs,
                published_layer_callbacks=publication_count,
                unused_terminal_prefetch_layers=publication_count - len(trace),
                live_max_steps_preserved=self.live_max_steps,
                formal_kv_write_count=15 * 28, formal_kv_written_rows_bitwise=True,
                target_logits_features_actual_kv_bitwise=True, predicted_tokens_identical=True,
                private_formal_rows=rows, private_request_slot=slot)
        except BaseException:
            # Unknown CUDA work or owner drainage retains allocator, bank/query
            # owners and charged diagnostic memory. Never force-free on failure.
            scratch_owners.update(retained=retained, replay_tensors=replay_tensors, fb_rows=fb_rows)
            self.replay_failures.append(dict(profiled=profile, repetition=repetition, warmup=warmup,
                started_unix=locals().get("started_unix"), finished_unix=time.time(),
                completed_foreground_stages=[item for item in stages if "events" not in item],
                completed_outputs=outputs, actual_layer_trace=list(owner.pipeline.trace),
                owner_state=owner.state, cuda_owner_quarantined=decoder.owner.quarantined))
            raise
        finally:
            if owner.state == "closed" and not decoder.owner.quarantined:
                torch.cuda.current_stream(device).synchronize()
                runner.req_to_token_pool.req_to_token[slot].copy_(prior_mapping)
                torch.cuda.current_stream(device).synchronize()
                allocator.free_kv(rows)
                allocator.free_request(slot)
                torch.cuda.current_stream(device).synchronize()
                _QUARANTINE.remove(scratch_owners)
                # Remove diagnostic method closures holding owner/capture GPU
                # references before a successful trial returns.
                del decoder.step
                decoder.owner._synchronize = original_fence
                owner.predict_one = owner.fetch_layer = None
                retained.clear()
                replay_tensors.clear()
                fb_rows.clear()
                scratch_owners.clear()

    def finish(self):
        import torch
        config = self.controller.config
        allocator = self.resources.runner.token_to_kv_pool_allocator
        if allocator.is_not_in_free_group is not True:
            raise RuntimeError("formal allocator free group must finish before replay allocation")
        if self.session._close_future is None:
            raise RuntimeError("original initial-KV session close has not been submitted")
        # Initial Prompt receive retirement runs asynchronously and calls a
        # CUDA device fence. Join that exact future outside timing so it cannot
        # deregister memory or synchronize a measured replay behind its back.
        if self.session._close_future.result(timeout=self.resources.config["timeout_seconds"]) is not True:
            raise RuntimeError("original initial-KV session close lacks terminal release proof")
        if (self.session._refresh_owner is not None or self.session.registration is not None
                or self.session.staging is not None or self.session.receive_guard is not None
                or self.session in self.resources.manager.pending_decode_closes
                or self.old_receive_guard is not None and self.old_receive_guard.value is not None):
            raise RuntimeError("original Prompt receive owners remain live; replay is prohibited")
        outputs = self.validate_live()
        if (not self.owner._retired or not self.owner.transport.closed
                or self.owner.transport.workers or self.owner.transport.quarantined
                or self.req.req_pool_idx is not None or self.resources.owners):
            raise RuntimeError("live request/native/formal ownership must retire before replay")
        trials = []
        for repetition in range(config["warmup_replays"] + config["measured_replays"]):
            trials.append(self.replay(profile=False, repetition=repetition,
                warmup=repetition < config["warmup_replays"]))
        # Events are a separate diagnostic trial, keeping event-record overhead
        # out of primary repeated wall measurements. No per-layer sync is added.
        event_trial = self.replay(profile=True, repetition=0, warmup=False)
        prefix = config["capture_directory"] / str(self.case)
        prefix.mkdir(parents=True, exist_ok=False)
        payload = dict(schema=SCHEMA, case=self.case, request_id=self.req.rid,
            prompt_ids=list(self.receipt.prompt), output_ids=list(outputs),
            eagle_seed=cpu_tree(self.seed_call), initial_eagle_cache=cpu_tree(self.initial_cache),
            initial_eagle_seen=list(self.initial_seen), initial_eagle_primed=self.initial_predicted,
            steps=[dict(step=row["step"], current=row["current"], predicted=row["predicted"],
                eagle_seen=row["eagle_seen"],
                position=row["position"], logits=cpu_tree(row["logits"]), features=cpu_tree(row["features"]),
                actual_kv=cpu_tree(row["actual_kv"]), banks=[dict(ids=bank.ids,
                    keys=cpu_tree(bank.keys), values=cpu_tree(bank.values), valid=cpu_tree(bank.valid))
                    for bank in row["banks"]],
                sampler={field.name: cpu_tree(getattr(row["sample"].sampling_info, field.name))
                    for field in dataclasses.fields(row["sample"].sampling_info)
                    if field.name not in ("penalizer_orchestrator", "apply_mask_func")},
                positions=cpu_tree(row["sample"].positions)) for row in self.steps])
        torch.save(payload, prefix / "trajectory.pt")
        report = dict(schema=SCHEMA, status="passed", case=self.case, request_id=self.req.rid,
            pid=os.getpid(), thread_id=threading.get_ident(), runner_id=id(self.resources.runner),
            target_model_id=id(self.resources.runner.model), target_class=type(self.resources.runner.model).__name__,
            device=str(self.resources.device), gpu_name=torch.cuda.get_device_name(self.resources.device),
            prompt_ids_sha256=prompt_ids_sha256(self.receipt.prompt), prompt_tokens=len(self.receipt.prompt),
            output_tokens=len(outputs), target_steps=len(self.steps), captured_layer_banks=15 * 28,
            config={key: str(value) if isinstance(value, Path) else value for key, value in config.items()},
            scope="same-runner READY-bank foreground replay; real paired target/EAGLE/sampler/formal writes and original query-ownership/executor/ticket/handoff path; no V/network/native receive/background GPU copy contention",
            timing_policy="two excluded replay warmups + three unprofiled wall trials per case, then one separate GPU-event trial; per-case repeated diagnostic, not ABBA; capture/prime/reset/validation/D2H/save excluded; existing safety fences retained",
            stage_scope="paired_owner_fence is included in paired_target; foreground_total envelopes all stages, so nested wall/event stages must not be added; CUDA event elapsed may include stream idle/CPU submission gaps and is not GPU utilization",
            trials=trials, event_trial=event_trial, budget=dict(limit_bytes=config["diagnostic_gpu_budget_bytes"],
                retained_storage_bytes=self.retained_bytes, replay_scratch_bound_bytes=self.scratch_bound,
                reservation_bytes=config["diagnostic_gpu_budget_bytes"], released=False),
            live_native_retired_before_replay=True, formal_private_rows_released=True,
            initial_session_close_joined=True, initial_receive_guard_released=True,
            formal_allocator_free_group_finished=True,
            trajectory_sha256=hashlib.sha256((prefix / "trajectory.pt").read_bytes()).hexdigest())
        # CPU evidence no longer borrows GPU storage. Prove completion, remove
        # every retained live bank/seed/cache/output reference and break the
        # patched live decoder closure before releasing its separate charge.
        torch.cuda.current_stream(self.resources.device).synchronize()
        del self.owner.decoder.step
        self.original_target_step = None
        self.draft_state.clear()
        self.primed.clear()
        self.steps.clear()
        self.retained_storages.clear()
        self.seed_call = self.initial_cache = None
        self.predict_one = self.propose = self.draft_state = self.primed = None
        self.owner = None
        self.session = self.old_receive_guard = None
        self.budget.release(self.reservation)
        report["budget"].update(released=True, gpu_refs_cleared=True,
                                final=self.budget.snapshot())
        save_json_atomic(prefix / "replay.json", report)
        self.closed = True
        LOGGER.info("PVD no-wait replay_saved case=%s rid=%s pid=%s directory=%s", self.case, self.req.rid, os.getpid(), prefix)


class Controller:
    def __init__(self, path):
        self.config = load_config(path)
        self.captures, self.completed, self.original_samples = {}, set(), {}
        self.pending = []
        self.active = None

    def selected_case(self, req):
        digest = prompt_ids_sha256(req.origin_input_ids)
        for case, expected in self.config["expected_prompt_ids_sha256"].items():
            if digest == expected:
                return int(case)
        return None

    def error(self, capture, error):
        directory = self.config["capture_directory"] / str(capture.case)
        directory.mkdir(parents=True, exist_ok=True)
        evidence = dict(schema=SCHEMA, status="failed", case=capture.case,
            request_id=capture.req.rid, pid=os.getpid(), failed_unix=time.time(),
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
            completed_capture_steps=len(capture.steps), replay_failures=capture.replay_failures,
            diagnostic_charge_retained=not capture.closed, budget=capture.budget.snapshot())
        save_json_atomic(directory / "error.json", evidence)
        LOGGER.error("PVD no-wait replay_failed case=%s pid=%s error=%s", capture.case, os.getpid(), error)

    def finish_pending(self):
        # Called only after the PUBLIC result-processing method exits its
        # Oasis context: native release, actual token commit, formal request
        # release and allocator.free_group_end have all completed on this thread.
        for capture in tuple(self.pending):
            if capture.req.req_pool_idx is not None or capture.session._close_future is None:
                continue  # Early abort/release must await actual formal freeing.
            try:
                capture.finish()
            except BaseException as error:
                self.error(capture, error)
                raise
            self.pending.remove(capture)
            del self.captures[capture.req.rid]
            self.completed.add(capture.case)
            _QUARANTINE.remove(capture)

    def prepare(self, original, resources, req, receipt):
        case = self.selected_case(req)
        if case is None:
            return original(resources, req, receipt)
        if (case in self.completed or self.captures or len(receipt.prompt) != 2159
                or req.sampling_params.max_new_tokens != 16 or resources.config["workers"] != 2):
            raise RuntimeError("duplicate, concurrent or wrong-shape formal capture")
        capture = Capture(self, resources, req, receipt, case)
        self.captures[req.rid] = capture
        _QUARANTINE.append(capture)
        original_draft = resources.draft.forward

        def draft_forward(features, *args, **kwargs):
            if capture.seed_call is None:
                capture.seed_call = dict(features=features,
                    input_ids=kwargs["input_ids"], past_key_values=kwargs.get("past_key_values"))
                capture.retain(capture.seed_call)
            return original_draft(features, *args, **kwargs)

        try:
            resources.draft.forward = draft_forward
            try:
                owner = original(resources, req, receipt)
            finally:
                resources.draft.forward = original_draft
            capture.bind(owner)
        except BaseException as error:
            self.error(capture, error)
            raise
        runner = resources.runner
        if id(runner) not in self.original_samples:
            original_sample = runner.sample
            self.original_samples[id(runner)] = original_sample

            def sample(logits, fb):
                if self.active is not None:
                    return self.active.sample(original_sample, logits, fb)
                return original_sample(logits, fb)
            runner.sample = sample
        LOGGER.info("PVD no-wait capture_enabled case=%s rid=%s pid=%s runner=%s device=%s prompt_ids_sha256=%s",
            case, req.rid, os.getpid(), id(runner), resources.device, prompt_ids_sha256(receipt.prompt))
        return owner


def controller():
    path = os.environ["PVD_OASIS_REPLAY_CONFIG"]
    if path not in _CONTROLLERS:
        _CONTROLLERS[path] = Controller(path)
    return _CONTROLLERS[path]


def patch_module(module):
    if module.__name__.endswith("oasis_startup"):
        original = module.OasisResources.prepare

        def prepare(resources, req, receipt):
            return controller().prepare(original, resources, req, receipt)
        module.OasisResources.prepare = prepare
    elif module.__name__.endswith("oasis_scheduler"):
        original_forward, original_release = module.OasisSchedulerBinding.forward, module.OasisSchedulerBinding.release

        def forward(binding, batch):
            state = controller()
            capture = state.captures.get(batch.reqs[0].rid)
            if capture is None:
                return original_forward(binding, batch)
            state.active = capture
            try:
                return original_forward(binding, batch)
            except BaseException as error:
                state.error(capture, error)
                raise
            finally:
                state.active = None

        def release(binding, req):
            state = controller()
            capture = state.captures.get(req.rid)
            deferred = original_release(binding, req)
            if capture is not None and not deferred:
                if capture not in state.pending:
                    state.pending.append(capture)
            return deferred
        module.OasisSchedulerBinding.forward, module.OasisSchedulerBinding.release = forward, release
    elif module.__name__.endswith("batch_result_processor"):
        original = module.SchedulerBatchResultProcessor.process_batch_result_decode

        def process(processor, batch, result):
            output = original(processor, batch, result)
            controller().finish_pending()
            return output
        module.SchedulerBatchResultProcessor.process_batch_result_decode = process
    elif module.__name__.endswith("decode_refresh"):
        original = module.PVDDecodeRefresher.cleanup_finished

        def cleanup(refresher):
            output = original(refresher)
            # Normal nondeferred owner release occurs in this next-iteration
            # cleanup, after the previous result processor freed formal rows.
            # Replay finishes before the scheduler admits the following case.
            controller().finish_pending()
            return output
        module.PVDDecodeRefresher.cleanup_finished = cleanup


class HookLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original
    def create_module(self, spec):
        return self.original.create_module(spec)
    def exec_module(self, module):
        self.original.exec_module(module)
        patch_module(module)


class HookFinder(importlib.abc.MetaPathFinder):
    names = {"sglang.srt.disaggregation.pvd.oasis_startup", "sglang.srt.disaggregation.pvd.oasis_scheduler",
             "sglang.srt.disaggregation.pvd.decode_refresh",
             "sglang.srt.managers.scheduler_components.batch_result_processor"}
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.names:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None:
            spec.loader = HookLoader(spec.loader)
        return spec


def install():
    global _INSTALLED
    if not os.environ.get("PVD_OASIS_REPLAY_CONFIG") or _INSTALLED:
        return
    for name in HookFinder.names:
        if name in sys.modules:
            raise RuntimeError("diagnostic import hook must precede serving Oasis imports")
    sys.meta_path.insert(0, HookFinder())
    _INSTALLED = True
    print(json.dumps(dict(schema=SCHEMA, hook_installed=True, pid=os.getpid(),
        config=os.environ["PVD_OASIS_REPLAY_CONFIG"])), file=sys.stderr, flush=True)
