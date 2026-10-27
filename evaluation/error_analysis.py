from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess

from evaluation.metrics import load_test_examples, resolve_mode
from evaluation.stockfish_eval import BUCKETS, CplPositionResult, classify_cpl, evaluate_cpl, percentile
from model.inference.factory import build_predictor
from model.inference.stockfish_rerank import CANDIDATE_SOURCES, DEFAULT_CANDIDATE_SOURCE, stockfish_version

PIECE_VALUES = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
ENDGAME_MATERIAL_MAX = 13
OPENING_FULLMOVE_MAX = 12
PHASES = ("opening", "middlegame", "endgame")

logger = logging.getLogger(__name__)


def non_pawn_material(board: chess.Board, color: chess.Color) -> int:
    return sum(value * len(board.pieces(piece, color)) for piece, value in PIECE_VALUES.items())


def classify_phase(board: chess.Board) -> str:
    if (non_pawn_material(board, chess.WHITE) <= ENDGAME_MATERIAL_MAX
            and non_pawn_material(board, chess.BLACK) <= ENDGAME_MATERIAL_MAX):
        return "endgame"
    if board.fullmove_number <= OPENING_FULLMOVE_MAX:
        return "opening"
    return "middlegame"


@dataclass
class PhaseRecord:
    phase: str
    top1: bool
    matched_best: bool
    cpl: int
    is_blunder: bool


def build_records(cpl_results: list[CplPositionResult], blunder_cp: int = 200) -> list[PhaseRecord]:
    records = []
    for r in cpl_results:
        try:
            board = chess.Board(r.fen)
        except ValueError:
            continue
        records.append(PhaseRecord(
            phase=classify_phase(board),
            top1=r.predicted_uci == r.ground_truth_uci,
            matched_best=r.matched_best,
            cpl=r.cpl,
            is_blunder=r.cpl > blunder_cp,
        ))
    return records


@dataclass
class PhaseAggregate:
    n: int = 0
    n_top1: int = 0
    n_matched_best: int = 0
    n_blunder: int = 0
    cpl_values: list[int] = field(default_factory=list)
    bucket_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(BUCKETS, 0))

    def add(self, record: PhaseRecord) -> None:
        self.n += 1
        if record.top1:
            self.n_top1 += 1
        if record.matched_best:
            self.n_matched_best += 1
        if record.is_blunder:
            self.n_blunder += 1
        self.cpl_values.append(record.cpl)
        self.bucket_counts[classify_cpl(record.cpl)] += 1

    @property
    def top1_rate(self) -> float:
        return self.n_top1 / self.n if self.n else 0.0

    @property
    def best_match_rate(self) -> float:
        return self.n_matched_best / self.n if self.n else 0.0

    @property
    def blunder_rate(self) -> float:
        return self.n_blunder / self.n if self.n else 0.0

    @property
    def mean_cpl(self) -> float:
        return sum(self.cpl_values) / len(self.cpl_values) if self.cpl_values else 0.0

    @property
    def median_cpl(self) -> float:
        return percentile(self.cpl_values, 50)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "top1_rate": round(self.top1_rate, 4),
            "best_match_rate": round(self.best_match_rate, 4),
            "blunder_rate": round(self.blunder_rate, 4),
            "mean_cpl": round(self.mean_cpl, 2),
            "median_cpl": round(self.median_cpl, 2),
            "bucket_counts": dict(self.bucket_counts),
        }


def aggregate_by_phase(records: list[PhaseRecord]) -> dict:
    overall = PhaseAggregate()
    by_phase = {phase: PhaseAggregate() for phase in PHASES}
    for record in records:
        overall.add(record)
        by_phase.setdefault(record.phase, PhaseAggregate()).add(record)
    return {"overall": overall.to_dict(), "by_phase": {p: by_phase[p].to_dict() for p in PHASES}}


def log_report(report: dict, *, mode: str, blunder_cp: int) -> None:
    logger.info("Error analysis by game phase (mode %s, blunder = more than %d cp)", mode, blunder_cp)
    logger.info("%-12s %6s %8s %10s %10s %10s", "phase", "n", "top-1", "blunder", "median cpl", "best")
    rows = [(phase, report["by_phase"][phase]) for phase in PHASES] + [("overall", report["overall"])]
    for name, s in rows:
        logger.info("%-12s %6d %7.1f%% %9.1f%% %10.1f %9.1f%%", name, s["n"], 100 * s["top1_rate"],
                    100 * s["blunder_rate"], s["median_cpl"], 100 * s["best_match_rate"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Top-1, centipawn loss and blunders for each game phase.")
    parser.add_argument("--adapter", required=True, help="adapter folder or Hub repo, or 'none' for the base model")
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=500, help="endgames are rare, use at least 500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stockfish", default="/usr/games/stockfish")
    parser.add_argument("--depth", type=int, default=12, help="search depth used to measure the loss")
    parser.add_argument("--blunder-cp", type=int, default=200)
    parser.add_argument("--output", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--constrained", action="store_true")
    modes.add_argument("--prefix-constrained", action="store_true")
    modes.add_argument("--stockfish-rerank", action="store_true")
    modes.add_argument("--exact", action="store_true")
    parser.add_argument("--stockfish-rerank-depth", type=int, default=6)
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
        _, cpl_results = evaluate_cpl(predictor, examples, stockfish_path=args.stockfish,
                                      depth=args.depth, mode=mode)
    finally:
        if mode == "stockfish-rerank":
            predictor.close()

    rerank = args.stockfish_rerank
    report = aggregate_by_phase(build_records(cpl_results, blunder_cp=args.blunder_cp))
    report.update({
        "mode": mode,
        "adapter": str(args.adapter),
        "test_file": str(args.test_jsonl),
        "stockfish_depth": args.depth,
        "blunder_cp": args.blunder_cp,
        "n_positions": report["overall"]["n"],
        "config": {
            "mode": mode,
            "seed": args.seed,
            "n_samples_requested": args.n_samples,
            "cpl_depth": args.depth,
            "rerank_depth": args.stockfish_rerank_depth if rerank else None,
            "rerank_k": args.stockfish_rerank_k if rerank else None,
            "rerank_candidates": args.rerank_candidates if rerank else None,
            "stockfish_version": stockfish_version(args.stockfish),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "phase_definition": {
            "endgame_material_max": ENDGAME_MATERIAL_MAX,
            "opening_fullmove_max": OPENING_FULLMOVE_MAX,
            "piece_values": {chess.piece_name(p): v for p, v in PIECE_VALUES.items()},
        },
    })
    log_report(report, mode=mode, blunder_cp=args.blunder_cp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Saved the report to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
