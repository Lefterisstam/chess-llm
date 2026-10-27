from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess

from model.inference.legal_move_filter import FilterResult
from model.inference.predict import format_prompt
from model.inference.prefix_constrained import (
    RESPONSE_ANCHOR,
    PrefixConstrainedChessMovePredictor,
    tokenize_move_in_context,
)

DEFAULT_BATCH_SIZE = 16
DEFAULT_INCLUDE_EOS = False


@dataclass
class ScoredMove:
    move: chess.Move
    logprob: float
    n_tokens: int
    rank: int = 0


def build_legal_move_token_map(tokenizer, board: chess.Board) -> list[tuple[chess.Move, list[int]]]:
    return [(m, tokenize_move_in_context(tokenizer, RESPONSE_ANCHOR, m.uci())) for m in board.legal_moves]


def sort_scored_moves(scored: list[ScoredMove]) -> list[ScoredMove]:
    ordered = sorted(scored, key=lambda s: (-s.logprob, s.move.uci()))
    for rank, s in enumerate(ordered, start=1):
        s.rank = rank
    return ordered


def legal_move_distribution(scored: list[ScoredMove]) -> list[tuple[chess.Move, float]]:
    if not scored:
        return []
    top = max(s.logprob for s in scored)
    weights = [math.exp(s.logprob - top) for s in scored]
    total = sum(weights)
    return [(s.move, w / total) for s, w in zip(scored, weights)]


def score_continuations(model, prompt_ids: list[int], continuations: list[list[int]],
                        pad_token_id: int, batch_size: int = DEFAULT_BATCH_SIZE) -> list[float]:
    import torch

    prompt_len = len(prompt_ids)
    scores = []
    for start in range(0, len(continuations), batch_size):
        batch = continuations[start:start + batch_size]
        longest = max(len(c) for c in batch)

        input_ids = torch.full((len(batch), prompt_len + longest), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, cont in enumerate(batch):
            seq = list(prompt_ids) + list(cont)
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :len(seq)] = 1

        with torch.no_grad():
            out = model(input_ids=input_ids.to(model.device),
                        attention_mask=attention_mask.to(model.device), use_cache=False)

        logits = out.logits[:, prompt_len - 1:prompt_len - 1 + longest, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for i, cont in enumerate(batch):
            targets = torch.tensor(cont, dtype=torch.long, device=logprobs.device)
            token_logprobs = logprobs[i, :len(cont), :].gather(1, targets.unsqueeze(1)).squeeze(1)
            scores.append(float(token_logprobs.sum().item()))
    return scores


def rank_legal_moves(model, tokenizer, prompt: str, board: chess.Board,
                     include_eos: bool = DEFAULT_INCLUDE_EOS,
                     batch_size: int = DEFAULT_BATCH_SIZE) -> list[ScoredMove]:
    token_map = build_legal_move_token_map(tokenizer, board)
    if not token_map:
        raise ValueError("Position has 0 legal moves (checkmate or stalemate).")

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else eos_id
    continuations = [list(tokens) + [eos_id] if include_eos else list(tokens) for _, tokens in token_map]
    logprobs = score_continuations(model, list(tokenizer(prompt).input_ids), continuations, pad_id,
                                   batch_size=batch_size)
    scored = [ScoredMove(move=move, logprob=lp, n_tokens=len(cont))
              for (move, _), cont, lp in zip(token_map, continuations, logprobs)]
    return sort_scored_moves(scored)


class ExactScoringChessMovePredictor(PrefixConstrainedChessMovePredictor):
    include_eos = DEFAULT_INCLUDE_EOS
    batch_size = DEFAULT_BATCH_SIZE

    def __init__(self, adapter_path: str, include_eos: bool = DEFAULT_INCLUDE_EOS,
                 batch_size: int = DEFAULT_BATCH_SIZE, **kwargs):
        super().__init__(adapter_path, **kwargs)
        self.include_eos = include_eos
        self.batch_size = batch_size

    def rank_legal_moves(self, fen: str, history: str, side: str) -> list[ScoredMove]:
        return rank_legal_moves(self.model, self.tokenizer, format_prompt(fen, history, side),
                                chess.Board(fen), include_eos=self.include_eos,
                                batch_size=self.batch_size)

    def predict_exact(self, fen: str, history: str, side: str) -> FilterResult:
        best = self.rank_legal_moves(fen, history, side)[0].move
        return FilterResult(best.uci(), best, was_legal=True, was_parseable=True,
                            fallback_strategy="exact_scoring")
