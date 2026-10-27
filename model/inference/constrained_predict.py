from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess

from model.inference.legal_move_filter import (
    FilterResult,
    extract_uci_candidate,
    filter_to_legal,
    try_parse_uci,
)
from model.inference.predict import ChessMovePredictor, format_prompt

DEFAULT_NUM_BEAMS = 5


class ConstrainedChessMovePredictor(ChessMovePredictor):
    def predict_constrained(self, fen: str, history: str, side: str,
                            num_beams: int = DEFAULT_NUM_BEAMS,
                            max_new_tokens: int = 10) -> FilterResult:
        import torch

        board = chess.Board(fen)
        legal = {m.uci() for m in board.legal_moves}
        if not legal:
            raise ValueError("Position has 0 legal moves (checkmate or stalemate).")

        inputs = self.tokenizer(format_prompt(fen, history, side), return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        beams = [extract_uci_candidate(self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True))
                 for out in outputs]

        top = beams[0] if beams else ""
        for candidate in beams:
            if candidate in legal:
                return FilterResult(
                    raw=top,
                    filtered=chess.Move.from_uci(candidate),
                    was_legal=top in legal,
                    was_parseable=try_parse_uci(top) is not None,
                    fallback_strategy="top_beam_legal" if candidate == top else "lower_beam_legal",
                )

        result = filter_to_legal(top, board)
        result.fallback_strategy = f"post_hoc_filter:{result.fallback_strategy}"
        return result
