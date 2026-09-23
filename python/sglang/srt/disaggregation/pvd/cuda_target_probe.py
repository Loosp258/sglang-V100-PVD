"""Explicit CUDA Llama/Qwen2 probes using private pools and target weights.

TP1/PP1, torch_native, FP16/FP32 only. The caller's target executor MUST use
the same execution lock. This is a synchronous baseline, not overlap evidence
or automatic production Scheduler activation. Native speculative output stays off.
"""

from contextlib import contextmanager

import torch
from sglang.srt.disaggregation.pvd.prediction import PredictionConfigError
from sglang.srt.disaggregation.pvd.target_probe import _LlamaTargetProbeCore


class CUDALlamaTargetProbe(_LlamaTargetProbeCore):
    def __init__(self, runner, config, *, device, execution_lock, **kwargs):
        selected = torch.device(device)
        if selected.type != "cuda" or selected.index is None:
            raise PredictionConfigError("explicit indexed CUDA probe device required")
        if not torch.cuda.is_available():
            raise PredictionConfigError("CUDA probe requested but CUDA is unavailable")
        if not all(
            callable(getattr(execution_lock, name, None))
            for name in ("acquire", "release")
        ):
            raise PredictionConfigError("shared target execution lock is required")
        self.device, self._execution_lock = str(selected), execution_lock
        self._execution_held = False
        super().__init__(runner, config, **kwargs)

    def _validate_placement(self, runner):
        requested = torch.device(self.device)
        runner_device = torch.device(runner.device)
        if (
            runner_device.type != "cuda"
            or (runner_device.index is not None and runner_device != requested)
            or getattr(runner, "gpu_id", requested.index) != requested.index
        ):
            raise PredictionConfigError("target runner and probe device differ")
        parameters = tuple(runner.model.parameters())
        dtypes = {p.dtype for p in parameters}
        if (
            not parameters
            or len(dtypes) != 1
            or not dtypes <= {torch.float16, torch.float32}
            or any(p.device != requested for p in parameters)
        ):
            raise PredictionConfigError(
                "probe needs uniform FP16/FP32 weights on its CUDA device"
            )
        self.dtype = next(iter(dtypes))

    def _drain_private(self):
        torch.cuda.synchronize(self.device)

    def quarantine_query_copy(self):
        """A consumer cannot prove completion of a read from captured Q.

        Keep the capture owner, its budget and the target execution lease.
        Only callable inside the active branch, before its cleanup can run.
        """
        self._require_main_thread()
        if not self._active or not self._execution_held:
            raise PredictionConfigError("query quarantine requires an active branch")
        self._quarantined = True

    def snapshot(self):
        self._require_main_thread()
        return {
            "device": self.device,
            "dtype": str(self.dtype),
            "active": self._active,
            "quarantined": self._quarantined,
            "target_execution_lease_retained": self._execution_held,
            "private_state_retained": self._private_state is not None,
            "reservation_bytes": self.reservation_bytes,
            "completion_policy": "device_synchronize",
        }

    @contextmanager
    def branch(self):
        self._require_main_thread()
        if self._active or self._quarantined or self._execution_held:
            raise PredictionConfigError("CUDA probe is active or quarantined")
        if not self._execution_lock.acquire(blocking=False):
            raise PredictionConfigError(
                "target execution is busy; probe cannot overlap"
            )
        self._execution_held = True
        try:
            # Pool constructors may create streams on the current device even
            # when their tensors receive an explicit device. Keep construction,
            # execution and both cleanup fences on the selected target device.
            with torch.cuda.device(self.device), super().branch():
                yield self
        finally:
            if not self._quarantined:
                self._execution_lock.release()
                self._execution_held = False


class CUDAQwen2TargetProbe(CUDALlamaTargetProbe):
    """Exact Qwen2 target-Q capture with Llama's CUDA ownership policy."""

    _model_architecture = "Qwen2ForCausalLM"
