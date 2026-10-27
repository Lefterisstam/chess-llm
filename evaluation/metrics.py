from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
from tqdm.auto import tqdm

from model.inference.factory import build_predictor
from model.inference.legal_move_filter import FilterResult, filter_to_legal
from model.inference.stockfish_rerank import CANDIDATE_SOURCES, DEFAULT_CANDIDATE_SOURCE, stockfish_version

logger = logging.getLogger(__name__)


@dataclass
class EvalStats:
    n_total: int = 0
    n_legal_raw: int = 0
    n_parseable_raw: int = 0
    n_correct_raw: int = 0
    n_correct_filtered: int = 0
    n_stockfish_overrode: int = 0
    total_candidates_evaluated: int = 0
    n_gt_in_candidates: int = 0
    n_gt_in_top5: int = 0
    n_gt_in_top10: int = 0
    sum_reciprocal_rank: float = 0.0
    fallback_strategies: dict[str, int] = field(default_factory=dict)
    adapter_path: str = ""
    test_file: str = ""
    config: dict = field(default_factory=dict)
    n_samples_used: int = 0
    duration_seconds: float = 0.0
    is_stockfish_rerank: bool = False
    is_exact: bool = False

    def _rate(self, count: float) -> float:
        return count / self.n_total if self.n_total else 0.0

    @property
    def lmr(self) -> float:
        return self._rate(self.n_legal_raw)

    @property
    def parse_rate(self) -> float:
        return self._rate(self.n_parseable_raw)

    @property
    def top1_raw(self) -> float:
        return self._rate(self.n_correct_raw)

    @property
    def top1_filtered(self) -> float:
        return self._rate(self.n_correct_filtered)

    @property
    def stockfish_override_rate(self) -> float:
        return self._rate(self.n_stockfish_overrode)

    @property
    def mean_candidates_per_position(self) -> float:
        return self._rate(self.total_candidates_evaluated)

    @property
    def candidate_recall(self) -> float:
        return self._rate(self.n_gt_in_candidates)

    @property
    def recall_at_5(self) -> float:
        return self._rate(self.n_gt_in_top5)

    @property
    def recall_at_10(self) -> float:
        return self._rate(self.n_gt_in_top10)

    @property
    def mrr(self) -> float:
        return self._rate(self.sum_reciprocal_rank)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lmr"] = round(self.lmr, 4)
        d["parse_rate"] = round(self.parse_rate, 4)
        d["top1_raw"] = round(self.top1_raw, 4)
        d["top1_filtered"] = round(self.top1_filtered, 4)
        if self.is_stockfish_rerank:
            d["stockfish_override_rate"] = round(self.stockfish_override_rate, 4)
            d["mean_candidates_per_position"] = round(self.mean_candidates_per_position, 2)
            d["candidate_recall"] = round(self.candidate_recall, 4)
        if self.is_exact:
            d["recall_at_5"] = round(self.recall_at_5, 4)
            d["recall_at_10"] = round(self.recall_at_10, 4)
            d["mrr"] = round(self.mrr, 4)
        return d

    def log_summary(self) -> None:
        logger.info("Positions: %d", self.n_total)
        logger.info("Legal move rate: %.1f%%, parse rate: %.1f%%", 100 * self.lmr, 100 * self.parse_rate)
        logger.info("Top-1 raw: %.1f%%, after filter: %.1f%%", 100 * self.top1_raw, 100 * self.top1_filtered)
        if self.is_stockfish_rerank:
            logger.info("Stockfish changed the LLM's pick: %.1f%%, candidates per position: %.1f, "
                        "human move among candidates: %.1f%%", 100 * self.stockfish_override_rate,
                        self.mean_candidates_per_position, 100 * self.candidate_recall)
        if self.is_exact:
            logger.info("Recall@5: %.1f%%, recall@10: %.1f%%, MRR: %.3f",
                        100 * self.recall_at_5, 100 * self.recall_at_10, self.mrr)
        if self.fallback_strategies:
            logger.info("Fallbacks: %s", self.fallback_strategies)
        logger.info("Took %.1f s", self.duration_seconds)


def extract_ground_truth_from_text(text: str) -> str:
    if "### Response:" not in text:
        return ""
    answer = text.split("### Response:", 1)[1].split()
    return answer[0].lower() if answer else ""


def extract_fen_from_text(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Position (FEN):"):
            return line.split(":", 1)[1].strip()
    return ""


def extract_history_from_text(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Move history"):
            return line.split(":", 1)[1].strip() if ":" in line else ""
    return ""


def extract_side_from_text(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("Side to move:"):
            return line.split(":", 1)[1].strip()
    return "White"


def resolve_mode(*, constrained: bool = False, prefix_constrained: bool = False,
                 stockfish_rerank: bool = False, exact: bool = False) -> str:
    if stockfish_rerank:
        return "stockfish-rerank"
    if exact:
        return "exact"
    if prefix_constrained:
        return "prefix-constrained"
    if constrained:
        return "constrained"
    return "greedy"


def evaluate_predictions(predictor, test_examples: list[dict], constrained: bool = False,
                         prefix_constrained: bool = False, stockfish_rerank: bool = False,
                         exact: bool = False, num_beams: int = 5) -> EvalStats:
    if sum([constrained, prefix_constrained, stockfish_rerank, exact]) > 1:
        raise ValueError("Use only one of constrained / prefix_constrained / stockfish_rerank / exact")

    stats = EvalStats(is_stockfish_rerank=stockfish_rerank, is_exact=exact)
    fallbacks = Counter()
    started = time.monotonic()

    for i, example in enumerate(tqdm(test_examples, desc="Evaluating", unit="example")):
        text = example.get("text", "")
        fen = extract_fen_from_text(text)
        history = extract_history_from_text(text)
        side = extract_side_from_text(text)
        truth = extract_ground_truth_from_text(text)
        if not fen or not truth:
            logger.warning("Example %d has no FEN or answer, skipping", i)
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            logger.warning("Example %d has an invalid FEN, skipping", i)
            continue

        try:
            if stockfish_rerank:
                reranked = predictor.predict_reranked(fen, history, side)
                result = FilterResult(reranked.move.uci(), reranked.move, True, True, "stockfish_rerank")
                stats.total_candidates_evaluated += reranked.num_candidates_legal
                if not reranked.selected_was_llm_top:
                    stats.n_stockfish_overrode += 1
                if any(c.uci() == truth for c in reranked.all_candidates):
                    stats.n_gt_in_candidates += 1
            elif exact:
                ranked = predictor.rank_legal_moves(fen, history, side)
                top = ranked[0].move
                result = FilterResult(top.uci(), top, True, True, "exact_scoring")
                truth_rank = next((s.rank for s in ranked if s.move.uci() == truth), None)
                if truth_rank is not None:
                    stats.sum_reciprocal_rank += 1.0 / truth_rank
                    if truth_rank <= 5:
                        stats.n_gt_in_top5 += 1
                    if truth_rank <= 10:
                        stats.n_gt_in_top10 += 1
            elif prefix_constrained:
                result = predictor.predict_prefix_constrained(fen, history, side, num_beams=num_beams)
            elif constrained:
                result = predictor.predict_constrained(fen, history, side, num_beams=num_beams)
            else:
                result = filter_to_legal(predictor.predict_raw(fen, history, side), board)
        except ValueError:
            continue

        stats.n_total += 1
        if result.was_parseable:
            stats.n_parseable_raw += 1
        if result.was_legal:
            stats.n_legal_raw += 1
            if result.raw == truth:
                stats.n_correct_raw += 1
        else:
            fallbacks[result.fallback_strategy] += 1
        if result.filtered.uci() == truth:
            stats.n_correct_filtered += 1

    stats.duration_seconds = time.monotonic() - started
    stats.fallback_strategies = dict(fallbacks)
    return stats


def load_test_examples(test_jsonl: Path, n_samples: int | None, seed: int = 42) -> list[dict]:
    examples = []
    with test_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    logger.info("Loaded %d examples from %s", len(examples), test_jsonl)
    if n_samples is not None and n_samples < len(examples):
        examples = random.Random(seed).sample(examples, n_samples)
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure legal move rate and top-1 accuracy on the test set.")
    parser.add_argument("--adapter", required=True, help="adapter folder or Hub repo, or 'none' for the base model")
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=1000, help="random subset size, 0 = all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path, help="where to write the results JSON")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--constrained", action="store_true", help="beam search, then filter")
    modes.add_argument("--prefix-constrained", action="store_true", help="beam search over legal moves only")
    modes.add_argument("--stockfish-rerank", action="store_true", help="model candidates reranked by Stockfish")
    modes.add_argument("--exact", action="store_true", help="score every legal move and take the best")
    parser.add_argument("--stockfish", help="Stockfish binary, needed for --stockfish-rerank")
    parser.add_argument("--stockfish-rerank-depth", type=int, default=6)
    parser.add_argument("--stockfish-rerank-k", type=int, default=10)
    parser.add_argument("--rerank-candidates", choices=list(CANDIDATE_SOURCES), default=DEFAULT_CANDIDATE_SOURCE)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    if args.stockfish_rerank and not args.stockfish:
        parser.error("--stockfish-rerank requires --stockfish PATH")

    mode = resolve_mode(constrained=args.constrained, prefix_constrained=args.prefix_constrained,
                        stockfish_rerank=args.stockfish_rerank, exact=args.exact)
    examples = load_test_examples(args.test_jsonl, args.n_samples if args.n_samples > 0 else None, args.seed)
    predictor = build_predictor(args.adapter, mode, stockfish_path=args.stockfish,
                                rerank_depth=args.stockfish_rerank_depth, rerank_k=args.stockfish_rerank_k,
                                rerank_candidates=args.rerank_candidates)
    try:
        stats = evaluate_predictions(predictor, examples, constrained=args.constrained,
                                     prefix_constrained=args.prefix_constrained,
                                     stockfish_rerank=args.stockfish_rerank, exact=args.exact,
                                     num_beams=args.num_beams)
    finally:
        if args.stockfish_rerank:
            predictor.close()

    rerank = args.stockfish_rerank
    stats.adapter_path = str(args.adapter)
    stats.test_file = str(args.test_jsonl)
    stats.n_samples_used = len(examples)
    stats.config = {
        "mode": mode,
        "seed": args.seed,
        "n_samples_requested": args.n_samples,
        "num_beams": args.num_beams if mode in ("constrained", "prefix-constrained") else None,
        "rerank_depth": args.stockfish_rerank_depth if rerank else None,
        "rerank_k": args.stockfish_rerank_k if rerank else None,
        "rerank_candidates": args.rerank_candidates if rerank else None,
        "stockfish_version": stockfish_version(args.stockfish) if rerank else None,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stats.log_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved results to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
