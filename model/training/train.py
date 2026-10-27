from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from model.config.qlora_config import LoRAConfig, QuantizationConfig
from model.config.training_config import TrainingConfig, make_continued_config, make_smoke_config
from model.training.callbacks import (
    MemoryMonitorCallback,
    StepTimingCallback,
    make_early_stopping_callback,
)

logger = logging.getLogger(__name__)


def login_to_services() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    else:
        logger.warning("HF_TOKEN is not set, pushing to the Hub will fail")
    if wandb_key:
        import wandb
        wandb.login(key=wandb_key)
    else:
        logger.warning("WANDB_API_KEY is not set, training runs without W&B tracking")


def load_base_model_and_tokenizer(model_id: str, quant: QuantizationConfig,
                                  attn_implementation: str = "flash_attention_2"):
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except AttributeError:
        pass

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant.load_in_4bit,
        bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,
                                                     device_map="auto",
                                                     attn_implementation=attn_implementation)
    except (ImportError, ValueError) as e:
        logger.warning("%s is not available (%s), using sdpa", attn_implementation, e)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,
                                                     device_map="auto", attn_implementation="sdpa")
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def attach_lora(model, lora: LoRAConfig, use_gradient_checkpointing: bool = False):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    model = get_peft_model(model, LoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        target_modules=lora.target_modules,
        bias=lora.bias,
        task_type=lora.task_type,
    ))
    model.print_trainable_parameters()
    return model


def warm_start_lora(model, adapter_path: str, use_gradient_checkpointing: bool = False):
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    logger.info("Continuing training of the adapter in %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.print_trainable_parameters()
    return model


def load_datasets(train_path: str, val_path: str, train_subset_size: int | None = None,
                  val_subset_size: int | None = None) -> DatasetDict:
    train = load_dataset("json", data_files=train_path, split="train").select_columns(["text"])
    val = load_dataset("json", data_files=val_path, split="train").select_columns(["text"])
    if train_subset_size is not None:
        train = train.select(range(min(train_subset_size, len(train))))
    if val_subset_size is not None:
        val = val.select(range(min(val_subset_size, len(val))))
    logger.info("Train examples: %d, validation examples: %d", len(train), len(val))
    return DatasetDict({"train": train, "validation": val})


def build_sft_config(cfg: TrainingConfig) -> SFTConfig:
    kwargs = dict(
        output_dir=cfg.output_dir,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        max_length=cfg.max_seq_length,
        dataset_text_field="text",
        packing=cfg.packing,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.gradient_checkpointing else None,
        group_by_length=cfg.group_by_length,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_persistent_workers=cfg.dataloader_persistent_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        remove_unused_columns=cfg.remove_unused_columns,
        push_to_hub=cfg.push_to_hub,
        hub_model_id=cfg.hub_model_id,
        hub_strategy=cfg.hub_strategy,
        hub_private_repo=cfg.hub_private_repo,
        seed=cfg.seed,
    )
    accepted = inspect.signature(SFTConfig.__init__).parameters
    unsupported = sorted(k for k in kwargs if k not in accepted)
    if unsupported:
        logger.warning("This TRL version does not support %s, ignoring them", unsupported)
    return SFTConfig(**{k: v for k, v in kwargs.items() if k in accepted})


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))


def checkpoint_is_stale(checkpoint_path: Path, max_steps: int) -> bool:
    if max_steps <= 0:
        return False
    try:
        step = int(checkpoint_path.name.rsplit("-", 1)[-1])
    except ValueError:
        return False
    return step >= max_steps


def is_checkpoint_compatible(checkpoint_path: Path, lora: LoRAConfig) -> bool:
    config_file = checkpoint_path / "adapter_config.json"
    if not config_file.exists():
        return True
    try:
        saved = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    return (saved.get("r") == lora.r
            and saved.get("lora_alpha") == lora.lora_alpha
            and set(saved.get("target_modules", [])) == set(lora.target_modules))


def pick_resume_checkpoint(cfg: TrainingConfig, lora: LoRAConfig) -> Path | None:
    checkpoint = find_latest_checkpoint(Path(cfg.output_dir))
    if checkpoint is None:
        return None
    if not is_checkpoint_compatible(checkpoint, lora):
        logger.warning("%s was trained with a different LoRA setup, starting from scratch", checkpoint)
        return None
    if checkpoint_is_stale(checkpoint, cfg.max_steps):
        logger.warning("%s is already past max_steps (left over from another run), not resuming",
                       checkpoint)
        return None
    logger.info("Resuming from %s", checkpoint)
    return checkpoint


def run_training(cfg: TrainingConfig, lora: LoRAConfig, quant: QuantizationConfig) -> None:
    logger.info("Training %s, output in %s, effective batch %d", cfg.base_model, cfg.output_dir,
                cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    login_to_services()

    model, tokenizer = load_base_model_and_tokenizer(cfg.base_model, quant, cfg.attn_implementation)
    if cfg.start_adapter_path:
        model = warm_start_lora(model, cfg.start_adapter_path, cfg.gradient_checkpointing)
    else:
        model = attach_lora(model, lora, cfg.gradient_checkpointing)

    dataset = load_datasets(cfg.train_jsonl, cfg.val_jsonl, cfg.train_subset_size,
                            cfg.val_subset_size)
    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(cfg),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[
            make_early_stopping_callback(patience=cfg.early_stopping_patience),
            MemoryMonitorCallback(critical_pct=0.95),
            StepTimingCallback(),
        ],
    )

    resume = pick_resume_checkpoint(cfg, lora) if cfg.resume_from_checkpoint else None
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    final_path = Path(cfg.output_dir) / "final_adapter"
    trainer.save_model(str(final_path))
    logger.info("Saved the final adapter to %s", final_path)
    if cfg.push_to_hub:
        trainer.push_to_hub(commit_message="Final adapter after training")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning of Mistral 7B on chess moves.")
    preset = p.add_mutually_exclusive_group()
    preset.add_argument("--smoke", action="store_true", help="100-step test run")
    preset.add_argument("--continued", action="store_true",
                        help="continue the first adapter for one full epoch (this produced v2)")
    p.add_argument("--num-train-epochs", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--per-device-train-batch-size", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--from-adapter", type=str, default=None,
                   help="continue training this adapter instead of starting a new one")
    p.add_argument("--no-push-to-hub", action="store_true")
    gpu = p.add_mutually_exclusive_group()
    gpu.add_argument("--small-gpu", action="store_true", help="T4/L4: batch 4 x 6, 8-bit optimizer")
    gpu.add_argument("--medium-gpu", action="store_true", help="A100 40GB: batch 12 x 2")
    ckpt = p.add_mutually_exclusive_group()
    ckpt.add_argument("--gradient-checkpointing", action="store_true")
    ckpt.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--packing", action="store_true")
    precision = p.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true", help="for GPUs without bf16, e.g. T4")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def apply_small_gpu_overrides(cfg: TrainingConfig) -> TrainingConfig:
    cfg.per_device_train_batch_size = 4
    cfg.per_device_eval_batch_size = 4
    cfg.gradient_accumulation_steps = 6
    cfg.gradient_checkpointing = True
    cfg.optim = "paged_adamw_8bit"
    return cfg


def apply_medium_gpu_overrides(cfg: TrainingConfig) -> TrainingConfig:
    cfg.per_device_train_batch_size = 12
    cfg.per_device_eval_batch_size = 12
    cfg.gradient_accumulation_steps = 2
    cfg.gradient_checkpointing = True
    return cfg


def apply_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    overrides = {
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "start_adapter_path": args.from_adapter,
        "run_name": args.run_name,
        "output_dir": args.output_dir,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(cfg, name, value)

    if args.small_gpu:
        cfg = apply_small_gpu_overrides(cfg)
    elif args.medium_gpu:
        cfg = apply_medium_gpu_overrides(cfg)
    if args.gradient_checkpointing:
        cfg.gradient_checkpointing = True
    elif args.no_gradient_checkpointing:
        cfg.gradient_checkpointing = False
    if args.packing:
        cfg.packing = True
    if args.no_push_to_hub:
        cfg.push_to_hub = False
    if args.fp16:
        cfg.fp16, cfg.bf16 = True, False
    if args.bf16:
        cfg.fp16, cfg.bf16 = False, True
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    if args.smoke:
        cfg = make_smoke_config()
    elif args.continued:
        cfg = make_continued_config()
    else:
        cfg = TrainingConfig()
    run_training(apply_overrides(cfg, args), LoRAConfig(), QuantizationConfig())
    return 0


if __name__ == "__main__":
    sys.exit(main())
