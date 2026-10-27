from __future__ import annotations

from dataclasses import dataclass

CHECKPOINT_ROOT = "/content/drive/MyDrive/chess_llm_checkpoints"
DATA_ROOT = "/content/drive/MyDrive/chess_llm_data/processed"

BEST_ADAPTER_PATH = f"{CHECKPOINT_ROOT}/mistral7b-qlora-chess-v2/final_adapter"
BEST_HUB_REPO = "Leftstam/chess-llm-mistral-7b-v2-lora"
BASE_ADAPTER_PATH = f"{CHECKPOINT_ROOT}/mistral7b-qlora-chess/final_adapter"


@dataclass
class TrainingConfig:
    base_model: str = "mistralai/Mistral-7B-v0.1"
    start_adapter_path: str = ""
    train_jsonl: str = f"{DATA_ROOT}/train.jsonl"
    val_jsonl: str = f"{DATA_ROOT}/val.jsonl"
    output_dir: str = f"{CHECKPOINT_ROOT}/mistral7b-qlora-chess"

    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 24
    per_device_eval_batch_size: int = 24
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    max_steps: int = 8000
    max_seq_length: int = 256
    packing: bool = False
    warmup_ratio: float = 0.10
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch_fused"
    seed: int = 42

    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 5
    resume_from_checkpoint: bool = True

    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = False
    attn_implementation: str = "flash_attention_2"

    report_to: str = "wandb"
    wandb_project: str = "chess-llm-thesis"
    wandb_entity: str = "lefterisstamoulo-university-of-piraeus"
    run_name: str = "mistral7b-qlora-chess"

    push_to_hub: bool = True
    hub_model_id: str = "Leftstam/chess-llm-mistral-7b-lora"
    hub_strategy: str = "every_save"
    hub_private_repo: bool = True

    dataloader_num_workers: int = 4
    dataloader_persistent_workers: bool = True
    dataloader_pin_memory: bool = True
    group_by_length: bool = True
    remove_unused_columns: bool = False
    train_subset_size: int | None = None
    val_subset_size: int | None = 5000


def make_smoke_config() -> TrainingConfig:
    return TrainingConfig(
        max_steps=100,
        num_train_epochs=1,
        save_steps=50,
        eval_steps=50,
        logging_steps=10,
        save_total_limit=1,
        run_name="mistral7b-qlora-chess-smoke",
        output_dir=f"{CHECKPOINT_ROOT}/mistral7b-qlora-chess-smoke",
        push_to_hub=False,
        hub_model_id="Leftstam/chess-llm-mistral-7b-lora-smoke",
        train_subset_size=5000,
        val_subset_size=500,
    )


def make_continued_config() -> TrainingConfig:
    return TrainingConfig(
        output_dir=f"{CHECKPOINT_ROOT}/mistral7b-qlora-chess-v2",
        hub_model_id=BEST_HUB_REPO,
        run_name="mistral7b-qlora-chess-v2",
        start_adapter_path=BASE_ADAPTER_PATH,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        num_train_epochs=1,
        max_steps=-1,
        early_stopping_patience=10**9,
        save_steps=2000,
        eval_steps=2000,
    )
