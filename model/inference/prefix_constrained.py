from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess

from model.inference.legal_move_filter import FilterResult
from model.inference.predict import ChessMovePredictor, format_prompt

DEFAULT_NUM_BEAMS = 5
RESPONSE_ANCHOR = "### Response:\n"

logger = logging.getLogger(__name__)


def tokenize_move_in_context(tokenizer, anchor: str, move_uci: str) -> list[int]:
    anchor_ids = list(tokenizer(anchor, add_special_tokens=False).input_ids)
    full_ids = list(tokenizer(anchor + move_uci, add_special_tokens=False).input_ids)
    if full_ids[:len(anchor_ids)] != anchor_ids:
        logger.warning("Token boundary changed for %r, tokenizing the move on its own", move_uci)
        return list(tokenizer(move_uci, add_special_tokens=False).input_ids)
    return full_ids[len(anchor_ids):]


def build_legal_move_token_sets(tokenizer, board: chess.Board) -> list[list[int]]:
    return [tokenize_move_in_context(tokenizer, RESPONSE_ANCHOR, m.uci()) for m in board.legal_moves]


def make_prefix_allowed_tokens_fn(legal_token_seqs: list[list[int]], prompt_length: int,
                                  eos_token_id: int):
    sequences = [tuple(seq) for seq in legal_token_seqs]
    complete = set(sequences)

    def prefix_fn(batch_id: int, input_ids) -> list[int]:
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        generated = tuple(ids[prompt_length:])
        if generated in complete:
            return [eos_token_id]
        n = len(generated)
        allowed = {seq[n] for seq in sequences if len(seq) > n and seq[:n] == generated}
        return list(allowed) if allowed else [eos_token_id]

    return prefix_fn


class PrefixConstrainedChessMovePredictor(ChessMovePredictor):
    def predict_prefix_constrained(self, fen: str, history: str, side: str,
                                   num_beams: int = DEFAULT_NUM_BEAMS,
                                   max_new_tokens: int = 10) -> FilterResult:
        import torch

        board = chess.Board(fen)
        if not list(board.legal_moves):
            raise ValueError("Position has 0 legal moves (checkmate or stalemate).")

        inputs = self.tokenizer(format_prompt(fen, history, side), return_tensors="pt").to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]
        prefix_fn = make_prefix_allowed_tokens_fn(build_legal_move_token_sets(self.tokenizer, board),
                                                  prompt_length, self.tokenizer.eos_token_id)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_fn,
            )

        text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).split()
        uci = text[0] if text else ""
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            move = None

        if move is None or move not in board.legal_moves:
            logger.warning("Constrained generation returned %r, using the first legal move", uci)
            fallback = min(board.legal_moves, key=lambda m: m.uci())
            return FilterResult(uci, fallback, was_legal=False, was_parseable=move is not None,
                                fallback_strategy="prefix_constrained_bug")
        return FilterResult(uci, move, was_legal=True, was_parseable=True,
                            fallback_strategy="prefix_constrained")
