from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.inference.stockfish_rerank import (
    DEFAULT_CANDIDATE_SOURCE,
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_STOCKFISH_DEPTH,
)

MODES = ("greedy", "constrained", "prefix-constrained", "exact", "stockfish-rerank")


def build_predictor(adapter_path: str, mode: str, *, stockfish_path: str | None = None,
                    rerank_depth: int = DEFAULT_STOCKFISH_DEPTH,
                    rerank_k: int = DEFAULT_NUM_CANDIDATES,
                    rerank_candidates: str = DEFAULT_CANDIDATE_SOURCE):
    if mode == "stockfish-rerank":
        from model.inference.stockfish_rerank import StockfishRerankPredictor
        return StockfishRerankPredictor(adapter_path=adapter_path, stockfish_path=stockfish_path,
                                        stockfish_depth=rerank_depth, num_candidates=rerank_k,
                                        candidate_source=rerank_candidates)
    if mode == "exact":
        from model.inference.exact_scoring import ExactScoringChessMovePredictor
        return ExactScoringChessMovePredictor(adapter_path=adapter_path)
    if mode == "prefix-constrained":
        from model.inference.prefix_constrained import PrefixConstrainedChessMovePredictor
        return PrefixConstrainedChessMovePredictor(adapter_path=adapter_path)
    if mode == "constrained":
        from model.inference.constrained_predict import ConstrainedChessMovePredictor
        return ConstrainedChessMovePredictor(adapter_path=adapter_path)
    if mode == "greedy":
        from model.inference.predict import ChessMovePredictor
        return ChessMovePredictor(adapter_path=adapter_path)
    raise ValueError(f"Unknown inference mode {mode!r}, choose one of {MODES}")
