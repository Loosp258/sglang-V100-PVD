"""Explicit TP1 experimental six-block Draft with learned target-Q output.

The target probe remains available for authoritative committed-position
recovery. Predicted tokens and HF KV are private and never enter target KV.
"""

import hashlib
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.disaggregation.pvd.cuda_probe_search import CUDAPredictionPipeline
from sglang.srt.disaggregation.pvd.cuda_target_probe import CUDAQwen2TargetProbe
from sglang.srt.disaggregation.pvd.prediction import (
    CommittedPrefix, DraftConfig, DraftPrediction, DraftProvider,
    PredictionConfigError, PredictionPipeline, ProbeConfig, QueryVectors,
)
from sglang.srt.disaggregation.pvd.transfer_lifecycle import TransferBudget

logger = logging.getLogger(__name__)
LATEST_SHA256 = "4b3fa87316c6b70d513fb32c3cdb79baf586e95046efbbfe3a51939cc8974215"
_FAILED_STARTUPS = []


class TargetQReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.empty(6, 896, 896))
        self.output = nn.Parameter(torch.empty(28, 896, 28 * 128))
        self.bias = nn.Parameter(torch.empty(28, 28 * 128))
        self.register_buffer("anchor_for_layer", torch.arange(28).mul(6).div(
            28, rounding_mode="floor"))
        self.anchor_logits = nn.Parameter(torch.empty(28, 6))

    def forward(self, features):
        anchors = F.gelu(torch.einsum("baw,awr->bar", features, self.anchor))
        selected = torch.einsum("bar,la->blr", anchors,
                                self.anchor_logits.softmax(dim=-1))
        return torch.einsum("blr,lrd->bld", selected, self.output) + self.bias


def apply_target_rope(query, position):
    freq = 1_000_000 ** (-torch.arange(
        0, 128, 2, device=query.device, dtype=torch.float32) / 128)
    phase = freq * position
    cosine = torch.cat((phase.cos(), phase.cos()))
    sine = torch.cat((phase.sin(), phase.sin()))
    rotated = torch.cat((-query[..., 64:], query[..., :64]), dim=-1)
    return query.float() * cosine + rotated.float() * sine


class JointDraftProvider(DraftProvider):
    def __init__(self, student, readout, config, scratch_budget, max_prefix,
                 checkpoint_sha256):
        self.student, self.readout, self.config = student, readout, config
        self.scratch_budget, self.max_prefix = scratch_budget, max_prefix
        self.checkpoint_sha256 = checkpoint_sha256
        self.factory = SimpleNamespace(prefix_cache_enabled=False)
        self.degraded = False
        self._active, self._owners = False, []

    def describe(self):
        return {"model": self.config.model_name_or_path,
                "revision": self.checkpoint_sha256,
                "query_source": "joint-six-layer-draft-target-q"}

    @contextmanager
    def branch(self):
        if self._active or self.degraded:
            raise PredictionConfigError("joint Draft branch active or quarantined")
        owner = f"pvd-joint-draft:{uuid.uuid4().hex}"
        # The configured scratch reservation covers eager FP32 attention,
        # private HF KV and Q/readout temporaries for the admitted context.
        self.scratch_budget.reserve(owner,
                                    self.scratch_budget.snapshot()["staging_bytes"], 1)
        self._active = True
        try:
            yield self
        finally:
            try:
                torch.cuda.current_stream(self.config.device).synchronize()
            except BaseException:
                self.degraded = True
                raise
            else:
                self._owners.clear()
                self._active = False
                self.scratch_budget.release(owner)

    def predict(self, prefix, max_tokens):
        prediction, _ = self.predict_q(prefix, max_tokens)
        return prediction

    @torch.inference_mode()
    def predict_q(self, prefix, max_tokens):
        if (not self._active or not isinstance(prefix, CommittedPrefix)
                or not 0 < len(prefix.tokens) <= self.max_prefix
                or not 1 <= max_tokens <= self.config.predict_tokens
                or any(not 0 <= token < self.student.config.vocab_size
                       for token in prefix.tokens)):
            raise PredictionConfigError("joint Draft input outside admitted bounds")
        from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

        device = torch.device(self.config.device)
        slots, handles = [None] * 6, []
        for slot, layer in enumerate(self.student.model.layers):
            def capture(module, inputs, output, slot=slot):
                state = output[0] if isinstance(output, tuple) else output
                slots[slot] = state[:, -1].detach()
            handles.append(layer.register_forward_hook(capture))
        self._owners.append(slots)
        started = time.perf_counter()
        try:
            ids = torch.tensor(prefix.tokens, device=device)[None]
            output = self.student(input_ids=ids, use_cache=True, logits_to_keep=1)
            self._owners.extend((output, ids))
            cache = output.past_key_values
            self._owners.append(cache)
            torch.cuda.current_stream(device).synchronize()
            prefill_done = time.perf_counter()
            penalty = RepetitionPenaltyLogitsProcessor(
                self.student.generation_config.repetition_penalty)
            token = penalty(ids, output.logits[:, -1].float()).argmax(-1, keepdim=True)
            tokens, rows = [], []
            self._owners.extend((tokens, rows))
            for step in range(max_tokens):
                tokens.append(token)
                output = self.student(input_ids=token, past_key_values=cache,
                                      use_cache=True, logits_to_keep=1)
                cache = output.past_key_values
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    raw = self.readout(torch.stack(slots, dim=1))
                rows.append(apply_target_rope(raw.reshape(28, 28, 128),
                                               len(prefix.tokens) + step))
                if step + 1 < max_tokens:
                    ids = torch.cat((ids, token), dim=-1)
                    token = penalty(ids, output.logits[:, -1].float()).argmax(
                        -1, keepdim=True)
            query = torch.stack(rows)
            self._owners.extend((output, cache, query))
            if not torch.isfinite(query).all().item():
                raise PredictionConfigError("joint Draft produced nonfinite target Q")
            future = tuple(torch.cat(tokens, dim=-1)[0].cpu().tolist())
            torch.cuda.current_stream(device).synchronize()
            logger.info("PVD joint Draft-Q: request=%s prefix_tokens=%d horizon=%d "
                        "prefill_seconds=%.6f rollout_q_seconds=%.6f target_forward_count=0",
                        prefix.request_id, len(prefix.tokens), max_tokens,
                        prefill_done - started, time.perf_counter() - prefill_done)
            return DraftPrediction(prefix.request_id, prefix.version, future,
                                   self.describe()), query
        finally:
            for handle in handles:
                handle.remove()


class JointDraftQPipeline(CUDAPredictionPipeline):
    def __init__(self, provider, probe, draft_config, probe_config, execution_lock):
        PredictionPipeline.__init__(self, provider, probe, draft_config, probe_config)
        self._lock = execution_lock
        self._rng_devices = [torch.device(draft_config.device).index]
        self._scope_active = self._quarantined = False
        self._worker_capture_lock = threading.Lock()
        self._worker_copy_retained = None
        self._worker_cpu_queries = False

    def run(self, prefix):
        if not self._scope_active:
            raise PredictionConfigError("joint Draft-Q requires its CUDA branch scope")
        prediction, query = self.provider.predict_q(
            prefix, self.draft_config.predict_tokens)
        positions = tuple(range(len(prefix.tokens),
                                len(prefix.tokens) + len(prediction.tokens)))
        queries = tuple(QueryVectors(
            vector_space=self.probe_config.target_model_id,
            version=prefix.version, layer=layer, head_start=0, head_count=28,
            positions=positions, valid_length=len(positions),
            vectors=query[:, layer], prefix_version=prefix.version,
            positional_encoding="rope_applied", request_id=prefix.request_id,
        ) for layer in range(28))
        return self._validate_queries(prefix, queries, positions)


def build_joint_startup(target_runner, *, checkpoint, draft_model_path,
                        target_model_id, placement, execution_lock, vocabulary,
                        max_prefix_tokens, predict_tokens, target_scratch_budget,
                        probe_transient_bytes_bound, prefix_budget):
    from transformers import AutoModelForCausalLM
    from sglang.srt.disaggregation.pvd.cuda_prediction_startup import CUDAPredictionStartup

    target = target_runner.model.config
    if (type(target_runner.model).__name__ != "Qwen2ForCausalLM"
            or (target.num_hidden_layers, target.num_attention_heads,
                target.num_key_value_heads, target.hidden_size) != (28, 28, 4, 3584)
            or float(target.rope_theta) != 1_000_000
            or max_prefix_tokens + predict_tokens > 2304):
        raise PredictionConfigError("joint checkpoint requires bounded Qwen2.5-7B TP1")
    if any(os.environ.get(name) == "1" for name in (
        "PVD_CONCURRENT_PREDICTION", "PVD_COOPERATIVE_PREDICTION",
        "PVD_SEED_PROBE_FROM_PROMPT_KV", "PVD_PRECOMPILE_QWEN_KERNELS")):
        raise PredictionConfigError("joint Draft-Q currently requires serialized unseeded prediction")
    with open(checkpoint, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != os.environ.get("PVD_JOINT_DRAFT_Q_SHA256", LATEST_SHA256):
        raise PredictionConfigError("joint Draft-Q checkpoint digest mismatch")
    # Two eager FP32 attention matrices dominate temporary storage. A
    # conservative additional 128 MiB covers cache, Q and smaller temporaries.
    required_scratch = (2 * 14 * (max_prefix_tokens + predict_tokens) ** 2 * 4
                        + 128 * 2**20)
    if placement.scratch_budget_bytes < required_scratch:
        raise PredictionConfigError("joint Draft-Q scratch budget is too small")
    retained = []
    try:
        with execution_lock, torch.cuda.device(placement.gpu_id), torch.random.fork_rng(
                devices=[placement.gpu_id]):
            device = torch.device(f"cuda:{placement.gpu_id}")
            student = AutoModelForCausalLM.from_pretrained(
                draft_model_path, dtype=torch.float32, attn_implementation="eager",
                local_files_only=True)
            retained.append(student)
            student.model.layers = nn.ModuleList(list(student.model.layers[:6]))
            student.model.config.num_hidden_layers = student.config.num_hidden_layers = 6
            readout = TargetQReadout()
            retained.append(readout)
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            student.load_state_dict(weights["student"], strict=True)
            readout.load_state_dict(weights["readout"], strict=True)
            del weights
            resident_bytes = sum(p.numel() * p.element_size()
                                 for model in (student, readout) for p in model.parameters())
            persistent = TransferBudget(placement.persistent_budget_bytes, 1)
            persistent.reserve(f"joint-resident:{digest}", resident_bytes, 1)
            student.to(device).eval()
            readout.to(device).eval()
            if max(vocabulary.allowed_ids) >= student.config.vocab_size:
                raise PredictionConfigError("joint Draft tokenizer exceeds embedding")
            config = DraftConfig(draft_model_path, revision=digest, device=str(device),
                                 dtype="float32", predict_tokens=predict_tokens)
            provider = JointDraftProvider(student, readout, config,
                TransferBudget(placement.scratch_budget_bytes, 1), max_prefix_tokens, digest)
            provider.persistent_budget = persistent
            retained.append(provider)
            probe_config = ProbeConfig(target_model_id, tuple(range(28)), head_count=28)
            probe = CUDAQwen2TargetProbe(target_runner, probe_config, device=device,
                execution_lock=execution_lock, target_model_id=target_model_id,
                max_tokens=max_prefix_tokens + predict_tokens,
                max_predict_tokens=predict_tokens,
                transient_bytes_bound=probe_transient_bytes_bound,
                budget=target_scratch_budget, prefix_budget=prefix_budget,
                vocabulary=vocabulary)
            pipeline = JointDraftQPipeline(provider, probe, config, probe_config, execution_lock)
            torch.cuda.synchronize(device)
            logger.info("PVD joint Draft-Q enabled: checkpoint_sha256=%s "
                        "student_layers=6 readout_rank=896 dtype=float32 resident_bytes=%d",
                        digest, resident_bytes)
            return CUDAPredictionStartup(pipeline, SimpleNamespace(model=student), None,
                vocabulary, target_model_id, resident_bytes, target_scratch_budget)
    except BaseException:
        _FAILED_STARTUPS.append(tuple(retained))
        raise
