from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from model.config.training_config import (
    BASE_ADAPTER_PATH,
    BEST_HUB_REPO,
    TrainingConfig,
    make_continued_config,
    make_smoke_config,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _train_module_importable() -> bool:
    if not _HAS_TORCH:
        return False
    try:
        importlib.import_module("model.training.train")
        return True
    except Exception:
        return False


_TRAIN_IMPORTABLE = _train_module_importable()


class TestTrainingConfigDefaults:
    def test_default_hyperparams(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate == 1e-4
        assert cfg.per_device_train_batch_size == 24
        assert cfg.gradient_accumulation_steps == 1
        assert cfg.max_steps == 8000
        assert cfg.max_seq_length == 256
        assert cfg.warmup_ratio == 0.10
        assert cfg.lr_scheduler_type == "cosine"
        assert cfg.max_grad_norm == 1.0
        assert cfg.bf16 is True
        assert cfg.fp16 is False
        assert cfg.gradient_checkpointing is False
        assert cfg.early_stopping_patience == 5
        assert cfg.save_steps == cfg.eval_steps == 500
        assert cfg.val_subset_size == 5000

    def test_packing_off_by_default(self):
        assert TrainingConfig().packing is False


class TestSmokeConfig:
    def test_isolated_output_dir(self):
        base = TrainingConfig()
        smoke = make_smoke_config()
        assert smoke.output_dir != base.output_dir
        assert "smoke" in smoke.output_dir.lower()

    def test_no_hub_push(self):
        smoke = make_smoke_config()
        assert smoke.push_to_hub is False

    def test_drastically_reduced_steps(self):
        smoke = make_smoke_config()
        assert smoke.max_steps == 100
        assert smoke.save_steps == 50
        assert smoke.eval_steps == 50

    def test_inherits_stable_optimization(self):
        smoke = make_smoke_config()
        assert smoke.learning_rate == 1e-4
        assert smoke.bf16 is True
        assert smoke.max_grad_norm == 1.0


class TestContinuedConfig:
    def test_warm_starts_from_base_reproduces_v2(self):
        base = TrainingConfig()
        c = make_continued_config()
        assert c.start_adapter_path == BASE_ADAPTER_PATH
        assert c.output_dir.endswith("mistral7b-qlora-chess-v2")
        assert c.hub_model_id == BEST_HUB_REPO
        assert c.learning_rate < base.learning_rate

    def test_trains_full_epoch_no_step_cap(self):
        c = make_continued_config()
        assert c.max_steps <= 0
        assert c.num_train_epochs >= 1

    def test_early_stopping_disabled(self):
        base = TrainingConfig()
        c = make_continued_config()
        assert c.early_stopping_patience > base.early_stopping_patience
        assert c.early_stopping_patience >= 1000

    def test_keeps_eval_data_for_comparability(self):
        base = TrainingConfig()
        c = make_continued_config()
        assert c.val_jsonl == base.val_jsonl
        assert c.train_jsonl == base.train_jsonl

    def test_isolated_output_from_stage1_base(self):
        base = TrainingConfig()
        c = make_continued_config()
        assert c.output_dir != base.output_dir
        assert "v2" in c.output_dir
        assert c.hub_model_id != base.hub_model_id

    def test_inherits_stable_lora_hyperparams(self):
        base = TrainingConfig()
        c = make_continued_config()
        assert c.per_device_train_batch_size == base.per_device_train_batch_size
        assert c.bf16 is True
        assert c.max_grad_norm == base.max_grad_norm
        assert c.save_steps == c.eval_steps


@pytest.mark.skipif(
    not _TRAIN_IMPORTABLE,
    reason="train.py needs torch, trl and peft",
)
class TestTrainCliParser:
    @pytest.fixture
    def parser(self):
        from model.training.train import _build_parser
        return _build_parser()

    def test_no_preset_default(self, parser):
        args = parser.parse_args([])
        assert args.smoke is False
        assert args.continued is False

    def test_smoke_flag(self, parser):
        args = parser.parse_args(["--smoke"])
        assert args.smoke is True
        assert args.continued is False

    def test_continued_flag(self, parser):
        args = parser.parse_args(["--continued"])
        assert args.continued is True
        assert args.smoke is False

    def test_smoke_and_continued_mutually_exclusive(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--smoke", "--continued"])

    def test_packing_flag_default_false(self, parser):
        assert parser.parse_args([]).packing is False

    def test_packing_combinable_with_continued(self, parser):
        args = parser.parse_args(["--continued", "--packing"])
        assert args.continued is True
        assert args.packing is True

    def test_packing_override_applies_to_config(self, parser):
        from model.config.training_config import TrainingConfig
        from model.training.train import apply_overrides
        cfg = apply_overrides(TrainingConfig(), parser.parse_args(["--packing"]))
        assert cfg.packing is True
        assert apply_overrides(TrainingConfig(), parser.parse_args([])).packing is False

    def test_small_gpu_flag_default_false(self, parser):
        args = parser.parse_args([])
        assert args.small_gpu is False

    def test_small_gpu_combinable_with_continued(self, parser):
        args = parser.parse_args(["--continued", "--small-gpu"])
        assert args.continued is True
        assert args.small_gpu is True

    def test_small_gpu_combinable_with_smoke(self, parser):
        args = parser.parse_args(["--smoke", "--small-gpu"])
        assert args.smoke is True
        assert args.small_gpu is True

    def test_medium_gpu_flag_default_false(self, parser):
        args = parser.parse_args([])
        assert args.medium_gpu is False

    def test_medium_gpu_combinable_with_continued(self, parser):
        args = parser.parse_args(["--continued", "--medium-gpu"])
        assert args.continued is True
        assert args.medium_gpu is True

    def test_small_and_medium_mutually_exclusive(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--small-gpu", "--medium-gpu"])

    def test_gradient_checkpointing_flag_default_false(self, parser):
        args = parser.parse_args([])
        assert args.gradient_checkpointing is False
        assert args.no_gradient_checkpointing is False

    def test_gradient_checkpointing_combinable_with_continued(self, parser):
        args = parser.parse_args(["--continued", "--gradient-checkpointing"])
        assert args.continued is True
        assert args.gradient_checkpointing is True

    def test_gradient_checkpointing_combinable_with_medium_gpu(self, parser):
        args = parser.parse_args(["--medium-gpu", "--gradient-checkpointing"])
        assert args.medium_gpu is True
        assert args.gradient_checkpointing is True

    def test_gradient_checkpointing_pair_mutually_exclusive(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--gradient-checkpointing", "--no-gradient-checkpointing"])


@pytest.mark.skipif(
    not _TRAIN_IMPORTABLE,
    reason="train.py imports torch at module level",
)
class TestCheckpointIsStale:
    @pytest.fixture
    def fn(self):
        from model.training.train import checkpoint_is_stale
        return checkpoint_is_stale

    def test_step_past_max_is_stale(self, fn):
        assert fn(Path("/x/mistral7b-qlora-tactical/checkpoint-3000"), 1500) is True

    def test_step_equal_max_is_stale(self, fn):
        assert fn(Path("/x/checkpoint-1500"), 1500) is True

    def test_step_before_max_is_fresh(self, fn):
        assert fn(Path("/x/checkpoint-500"), 1500) is False

    def test_max_steps_zero_cannot_judge(self, fn):
        assert fn(Path("/x/checkpoint-3000"), 0) is False

    def test_malformed_name_is_not_stale(self, fn):
        assert fn(Path("/x/final_adapter"), 1500) is False


@pytest.mark.skipif(
    not _TRAIN_IMPORTABLE,
    reason="train.py needs torch, trl and peft",
)
class TestApplySmallGpuOverrides:
    @pytest.fixture
    def apply_fn(self):
        from model.training.train import apply_small_gpu_overrides
        return apply_small_gpu_overrides

    def test_reduces_batch_to_4(self, apply_fn):
        cfg = TrainingConfig()
        result = apply_fn(cfg)
        assert result.per_device_train_batch_size == 4
        assert result.per_device_eval_batch_size == 4

    def test_accum_keeps_effective_batch_24(self, apply_fn):
        cfg = TrainingConfig()
        result = apply_fn(cfg)
        effective = result.per_device_train_batch_size * result.gradient_accumulation_steps
        assert effective == 24

    def test_enables_gradient_checkpointing(self, apply_fn):
        cfg = TrainingConfig()
        result = apply_fn(cfg)
        assert result.gradient_checkpointing is True

    def test_uses_paged_optimizer(self, apply_fn):
        cfg = TrainingConfig()
        result = apply_fn(cfg)
        assert result.optim == "paged_adamw_8bit"

    def test_preserves_lr_and_precision(self, apply_fn):
        cfg = TrainingConfig()
        result = apply_fn(cfg)
        assert result.learning_rate == 1e-4
        assert result.bf16 is True
        assert result.max_grad_norm == 1.0
        assert result.warmup_ratio == 0.10

    def test_works_on_top_of_continued_preset(self, apply_fn):
        cfg = make_continued_config()
        result = apply_fn(cfg)
        assert result.per_device_train_batch_size == 4
        assert result.gradient_checkpointing is True
        assert "v2" in result.output_dir
        assert result.hub_model_id == BEST_HUB_REPO
        assert result.max_steps <= 0


@pytest.mark.skipif(
    not _TRAIN_IMPORTABLE,
    reason="train.py imports torch at module level",
)
class TestApplyMediumGpuOverrides:
    @pytest.fixture
    def apply_fn(self):
        from model.training.train import apply_medium_gpu_overrides
        return apply_medium_gpu_overrides

    def test_batch_12(self, apply_fn):
        cfg = TrainingConfig()
        result = apply_fn(cfg)
        assert result.per_device_train_batch_size == 12
        assert result.per_device_eval_batch_size == 12

    def test_effective_batch_24_matches_g4_base(self, apply_fn):
        cfg = TrainingConfig()
        result = apply_fn(cfg)
        effective = result.per_device_train_batch_size * result.gradient_accumulation_steps
        assert effective == 24

    def test_enables_gradient_checkpointing(self, apply_fn):
        cfg = TrainingConfig()
        result = apply_fn(cfg)
        assert result.gradient_checkpointing is True

    def test_keeps_fused_optimizer(self, apply_fn):
        cfg = TrainingConfig()
        result = apply_fn(cfg)
        assert result.optim == "adamw_torch_fused"

    def test_preserves_lr_and_precision(self, apply_fn):
        cfg = TrainingConfig()
        result = apply_fn(cfg)
        assert result.learning_rate == 1e-4
        assert result.bf16 is True
        assert result.max_grad_norm == 1.0
        assert result.warmup_ratio == 0.10

    def test_works_on_top_of_continued_preset(self, apply_fn):
        cfg = make_continued_config()
        result = apply_fn(cfg)
        assert result.per_device_train_batch_size == 12
        assert result.gradient_checkpointing is True
        assert "v2" in result.output_dir
        assert result.hub_model_id == BEST_HUB_REPO
        assert result.max_steps <= 0

    def test_larger_batch_than_small_gpu(self, apply_fn):
        from model.training.train import apply_small_gpu_overrides
        small = apply_small_gpu_overrides(TrainingConfig())
        medium = apply_fn(TrainingConfig())
        assert medium.per_device_train_batch_size > small.per_device_train_batch_size
        small_eff = small.per_device_train_batch_size * small.gradient_accumulation_steps
        medium_eff = medium.per_device_train_batch_size * medium.gradient_accumulation_steps
        assert small_eff == medium_eff == 24


@pytest.mark.skipif(
    not _TRAIN_IMPORTABLE,
    reason="train.py imports torch at module level",
)
class TestGradientCheckpointingOverride:
    @pytest.fixture
    def apply_overrides(self):
        from model.training.train import apply_overrides
        return apply_overrides

    @pytest.fixture
    def parser(self):
        from model.training.train import _build_parser
        return _build_parser()

    def test_enables_on_default(self, parser, apply_overrides):
        cfg = TrainingConfig()
        assert cfg.gradient_checkpointing is False
        args = parser.parse_args(["--gradient-checkpointing"])
        result = apply_overrides(cfg, args)
        assert result.gradient_checkpointing is True
        assert result.per_device_train_batch_size == 24
        assert result.gradient_accumulation_steps == 1

    def test_enables_on_continued_preset(self, parser, apply_overrides):
        cfg = make_continued_config()
        args = parser.parse_args(["--continued", "--gradient-checkpointing"])
        result = apply_overrides(cfg, args)
        assert result.gradient_checkpointing is True
        assert result.per_device_train_batch_size == 24
        assert "v2" in result.output_dir
        assert result.max_steps <= 0

    def test_disable_overrides_medium_gpu(self, parser, apply_overrides):
        cfg = TrainingConfig()
        args = parser.parse_args(["--medium-gpu", "--no-gradient-checkpointing"])
        result = apply_overrides(cfg, args)
        assert result.gradient_checkpointing is False
        assert result.per_device_train_batch_size == 12

    def test_no_flag_preserves_default(self, parser, apply_overrides):
        cfg = TrainingConfig()
        original = cfg.gradient_checkpointing
        args = parser.parse_args([])
        result = apply_overrides(cfg, args)
        assert result.gradient_checkpointing == original


@pytest.mark.skipif(
    not _TRAIN_IMPORTABLE,
    reason="train.py imports torch at module level",
)
class TestGradientAccumulationOverride:
    @pytest.fixture
    def apply_overrides(self):
        from model.training.train import apply_overrides
        return apply_overrides

    @pytest.fixture
    def parser(self):
        from model.training.train import _build_parser
        return _build_parser()

    def test_changes_accumulation_steps(self, parser, apply_overrides):
        cfg = TrainingConfig()
        assert cfg.gradient_accumulation_steps == 1
        args = parser.parse_args(["--gradient-accumulation-steps", "6"])
        result = apply_overrides(cfg, args)
        assert result.gradient_accumulation_steps == 6

    def test_custom_l4_safe_combo(self, parser, apply_overrides):
        cfg = make_continued_config()
        args = parser.parse_args([
            "--continued",
            "--per-device-train-batch-size", "4",
            "--gradient-accumulation-steps", "6",
            "--gradient-checkpointing",
        ])
        result = apply_overrides(cfg, args)
        assert result.per_device_train_batch_size == 4
        assert result.gradient_accumulation_steps == 6
        assert result.per_device_train_batch_size * result.gradient_accumulation_steps == 24
        assert result.gradient_checkpointing is True
        assert result.optim == "adamw_torch_fused"
        assert "v2" in result.output_dir
        assert result.max_steps <= 0

    def test_no_flag_preserves_default(self, parser, apply_overrides):
        cfg = TrainingConfig()
        args = parser.parse_args([])
        result = apply_overrides(cfg, args)
        assert result.gradient_accumulation_steps == 1
