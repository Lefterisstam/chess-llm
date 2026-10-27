from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.inference.legal_move_filter import extract_uci_candidate

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
NO_ADAPTER = "none"

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "You are a chess grandmaster. Analyze the current position and suggest "
    "the best next move. Respond ONLY with the move in UCI format "
    "(e.g., e2e4, g1f3, e1g1 for castling).\n\n"
    "Position (FEN): {fen}\n"
    "Move history (last {history_n} moves): {history}\n"
    "Side to move: {side}\n\n"
    "### Response:\n"
)

logger = logging.getLogger(__name__)


def format_prompt(fen: str, history: str, side: str, history_n: int = 5) -> str:
    return PROMPT_TEMPLATE.format(fen=fen, history=history, side=side, history_n=history_n)


def uses_adapter(adapter_path: str) -> bool:
    if not adapter_path or not str(adapter_path).strip():
        raise ValueError(f"Empty adapter path, give a path or {NO_ADAPTER!r} for the base model")
    return str(adapter_path).strip().lower() != NO_ADAPTER


class ChessMovePredictor:
    def __init__(self, adapter_path: str, base_model: str = BASE_MODEL, device_map: str = "auto",
                 load_in_4bit: bool = True):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        has_adapter = uses_adapter(adapter_path)
        logger.info("Loading %s %s", base_model, "with adapter" if has_adapter else "without adapter")
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        compute_dtype = torch.bfloat16 if native_bf16 else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config if load_in_4bit else None,
            device_map=device_map,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path) if has_adapter else base
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict_raw(self, fen: str, history: str, side: str, max_new_tokens: int = 10) -> str:
        import torch

        inputs = self.tokenizer(format_prompt(fen, history, side), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return extract_uci_candidate(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
