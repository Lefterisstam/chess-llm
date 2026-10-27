from __future__ import annotations

import logging
import time
from collections import deque

import torch
from transformers import EarlyStoppingCallback, TrainerCallback

logger = logging.getLogger(__name__)


def make_early_stopping_callback(patience: int = 3, threshold: float = 0.0) -> EarlyStoppingCallback:
    return EarlyStoppingCallback(early_stopping_patience=patience,
                                 early_stopping_threshold=threshold)


class MemoryMonitorCallback(TrainerCallback):
    def __init__(self, critical_pct: float = 0.95):
        self.critical_pct = critical_pct

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = reserved / total

        if logs is not None:
            logs["gpu/allocated_gb"] = round(allocated, 2)
            logs["gpu/reserved_gb"] = round(reserved, 2)
            logs["gpu/utilisation_pct"] = round(used * 100, 1)

        if used >= self.critical_pct:
            logger.error("GPU memory at %.1f%% (%.1f of %.1f GB), stopping before an OOM",
                         used * 100, reserved, total)
            control.should_training_stop = True


class StepTimingCallback(TrainerCallback):
    def __init__(self):
        self.step_started = None
        self.step_times = deque(maxlen=100)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_started is not None:
            self.step_times.append(time.monotonic() - self.step_started)
            self.step_started = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.step_times or logs is None:
            return
        mean_step = sum(self.step_times) / len(self.step_times)
        remaining = max(0, state.max_steps - state.global_step)
        logs["timing/mean_step_sec"] = round(mean_step, 3)
        logs["timing/eta_hours"] = round(remaining * mean_step / 3600, 2)
