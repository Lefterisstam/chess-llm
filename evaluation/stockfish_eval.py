from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import chess.engine
from tqdm.auto import tqdm

from evaluation.metrics import (
    extract_fen_from_text,
    extract_ground_truth_from_text,
    extract_history_from_text,
    extract_side_from_text,
    load_test_examples,
    resolve_mode,
)
from model.inference.factory import build_predictor
from model.inference.legal_move_filter import filter_to_legal
from model.inference.stockfish_rerank import CANDIDATE_SOURCES, DEFAULT_CANDIDATE_SOURCE, stockfish_version

MATE_SCORE = 10000
BUCKET_EXCELLENT_MAX = 10
BUCKET_GOOD_MAX = 50
BUCKET_ACCEPTABLE_MAX = 200
BUCKETS = ("excellent", "good", "acceptable", "blunder")

logger = logging.getLogger(__name__)


def compute_cpl(eval_best_cp: int, eval_ours_cp: int) -> int:
    return max(0, eval_best_cp - eval_ours_cp)


def classify_cpl(cpl: int) -> str:
    if cpl <= BUCKET_EXCELLENT_MAX:
        return "excellent"
    if cpl <= BUCKET_GOOD_MAX:
        return "good"
    if cpl <= BUCKET_ACCEPTABLE_MAX:
        return "acceptable"
    return "blunder"


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    rank = (p / 100.0) * (len(s) - 1)
    lower = int(rank)
    if lower + 1 < len(s):
        return s[lower] + (rank - lower) * (s[lower + 1] - s[lower])
    return float(s[lower])


@dataclass
class CplPositionResult:
    fen: str
    ground_truth_uci: str
    predicted_uci: str
    best_move_uci: str
    eval_best_cp: int
    eval_ours_cp: int
    cpl: int
    matched_best: bool


@dataclass
class CplStats:
    n_positions: int = 0
    cpl_values: list[int] = field(default_factory=list)
    n_matched_best: int = 0
    adapter_path: str = ""
    test_file: str = ""
    stockfish_depth: int = 12
    mode: str = ""
    config: dict = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def mean_cpl(self) -> float:
        return sum(self.cpl_values) / len(self.cpl_values) if self.cpl_values else 0.0

    @property
    def median_cpl(self) -> float:
        return self.percentile(50)

    @property
    def best_match_rate(self) -> float:
        return self.n_matched_best / self.n_positions if self.n_positions else 0.0

    def percentile(self, p: float) -> float:
        return percentile(self.cpl_values, p)

    @property
    def bucket_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(BUCKETS, 0)
        for cpl in self.cpl_values:
            counts[classify_cpl(cpl)] += 1
        return counts

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean_cpl"] = round(self.mean_cpl, 2)
        d["median_cpl"] = round(self.median_cpl, 2)
        d["p25_cpl"] = round(self.percentile(25), 2)
        d["p75_cpl"] = round(self.percentile(75), 2)
        d["p95_cpl"] = round(self.percentile(95), 2)
        d["best_match_rate"] = round(self.best_match_rate, 4)
        d["bucket_counts"] = self.bucket_counts
        return d

    def log_summary(self) -> None:
        logger.info("Positions: %d (Stockfish depth %d, mode %s)", self.n_positions,
                    self.stockfish_depth, self.mode)
        logger.info("Same move as Stockfish: %.1f%%", 100 * self.best_match_rate)
        logger.info("CPL mean %.1f, median %.1f, p25/p75/p95 %.1f/%.1f/%.1f", self.mean_cpl,
                    self.median_cpl, self.percentile(25), self.percentile(75), self.percentile(95))
        logger.info("Buckets: %s", self.bucket_counts)
        logger.info("Took %.1f s", self.duration_seconds)


def score_for(score: chess.engine.PovScore, color: chess.Color) -> int:
    cp = score.pov(color).score(mate_score=MATE_SCORE)
    if cp is None:
        cp = MATE_SCORE if score.pov(color).is_mate() else 0
    return cp


def compute_position_cpl(engine, fen: str, our_move: chess.Move, depth: int = 12) -> CplPositionResult | None:
    board = chess.Board(fen)
    if not list(board.legal_moves):
        return None
    if our_move not in board.legal_moves:
        logger.warning("%s is not legal in %s, skipping", our_move.uci(), fen)
        return None

    limit = chess.engine.Limit(depth=depth)
    before = engine.analyse(board, limit)
    best_move = before["pv"][0]
    eval_best = score_for(before["score"], board.turn)

    after = board.copy()
    after.push(our_move)
    if after.is_game_over():
        eval_ours = MATE_SCORE if after.is_checkmate() else 0
    else:
        eval_ours = -score_for(engine.analyse(after, limit)["score"], after.turn)

    return CplPositionResult(
        fen=fen,
        ground_truth_uci="",
        predicted_uci=our_move.uci(),
        best_move_uci=best_move.uci(),
        eval_best_cp=eval_best,
        eval_ours_cp=eval_ours,
        cpl=compute_cpl(eval_best, eval_ours),
        matched_best=our_move == best_move,
    )


def predict_move(predictor, mode: str, fen: str, history: str, side: str) -> chess.Move:
    if mode == "stockfish-rerank":
        return predictor.predict_reranked(fen, history, side).move
    if mode == "prefix-constrained":
        return predictor.predict_prefix_constrained(fen, history, side).filtered
    if mode == "exact":
        return predictor.predict_exact(fen, history, side).filtered
    if mode == "constrained":
        return predictor.predict_constrained(fen, history, side).filtered
    return filter_to_legal(predictor.predict_raw(fen, history, side), chess.Board(fen)).filtered


def evaluate_cpl(predictor, test_examples: list[dict], stockfish_path: str, depth: int = 12,
                 mode: str = "greedy") -> tuple[CplStats, list[CplPositionResult]]:
    stats = CplStats(stockfish_depth=depth, mode=mode)
    results = []
    started = time.monotonic()

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        for example in tqdm(test_examples, desc="CPL eval", unit="position"):
            text = example.get("text", "")
            fen = extract_fen_from_text(text)
            truth = extract_ground_truth_from_text(text)
            if not fen or not truth:
                continue
            try:
                chess.Board(fen)
                move = predict_move(predictor, mode, fen, extract_history_from_text(text),
                                    extract_side_from_text(text))
            except ValueError:
                continue

            result = compute_position_cpl(engine, fen, move, depth=depth)
            if result is None:
                continue
            result.ground_truth_uci = truth
            results.append(result)
            stats.n_positions += 1
            stats.cpl_values.append(result.cpl)
            if result.matched_best:
                stats.n_matched_best += 1
    finally:
        engine.quit()

    stats.duration_seconds = time.monotonic() - started
    return stats, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Centipawn loss of the model's moves, measured with Stockfish.")
    parser.add_argument("--adapter", required=True, help="adapter folder or Hub repo, or 'none' for the base model")
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=200, help="random subset size, 0 = all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stockfish", default="/usr/games/stockfish")
    parser.add_argument("--depth", type=int, default=12, help="search depth used to measure the loss")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-per-position", action="store_true",
                        help="also write every position to <output>.positions.jsonl")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--constrained", action="store_true")
    modes.add_argument("--prefix-constrained", action="store_true")
    modes.add_argument("--stockfish-rerank", action="store_true")
    modes.add_argument("--exact", action="store_true")
    parser.add_argument("--stockfish-rerank-depth", type=int, default=6,
                        help="depth used by the reranker (not the measuring depth)")
    parser.add_argument("--stockfish-rerank-k", type=int, default=10)
    parser.add_argument("--rerank-candidates", choices=list(CANDIDATE_SOURCES), default=DEFAULT_CANDIDATE_SOURCE)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    mode = resolve_mode(constrained=args.constrained, prefix_constrained=args.prefix_constrained,
                        stockfish_rerank=args.stockfish_rerank, exact=args.exact)
    examples = load_test_examples(args.test_jsonl, args.n_samples if args.n_samples > 0 else None, args.seed)
    predictor = build_predictor(args.adapter, mode, stockfish_path=args.stockfish,
                                rerank_depth=args.stockfish_rerank_depth, rerank_k=args.stockfish_rerank_k,
                                rerank_candidates=args.rerank_candidates)
    try:
        stats, results = evaluate_cpl(predictor, examples, stockfish_path=args.stockfish,
                                      depth=args.depth, mode=mode)
    finally:
        if args.stockfish_rerank:
            predictor.close()

    rerank = args.stockfish_rerank
    stats.adapter_path = str(args.adapter)
    stats.test_file = str(args.test_jsonl)
    stats.config = {
        "mode": mode,
        "seed": args.seed,
        "n_samples_requested": args.n_samples,
        "cpl_depth": args.depth,
        "rerank_depth": args.stockfish_rerank_depth if rerank else None,
        "rerank_k": args.stockfish_rerank_k if rerank else None,
        "rerank_candidates": args.rerank_candidates if rerank else None,
        "stockfish_version": stockfish_version(args.stockfish),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stats.log_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved results to %s", args.output)

    if args.save_per_position:
        positions_path = args.output.with_suffix(args.output.suffix + ".positions.jsonl")
        with positions_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
