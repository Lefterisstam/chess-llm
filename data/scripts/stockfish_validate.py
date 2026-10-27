from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.metrics import extract_fen_from_text, extract_ground_truth_from_text

DEFAULT_DEPTH = 12
DEFAULT_MAX_CP_LOSS = 50
DEFAULT_STOCKFISH_THREADS = 1
DEFAULT_STOCKFISH_HASH_MB = 128
DEFAULT_CHECKPOINT_EVERY = 1000
MATE_SCORE = 10_000

REJECT_COUNTERS = {
    "invalid_json": "rejected_invalid_fen",
    "missing_fen": "rejected_missing_fen",
    "missing_target": "rejected_missing_target",
    "invalid_fen": "rejected_invalid_fen",
    "illegal_target": "rejected_illegal_target",
    "engine_error": "rejected_engine_error",
    "cp_loss": "rejected_cp_loss",
}

logger = logging.getLogger(__name__)


def compute_cp_loss(best_cp: int, human_cp: int) -> int:
    return max(0, best_cp - human_cp)


def open_engine(stockfish_path: str, threads: int = DEFAULT_STOCKFISH_THREADS,
                hash_mb: int = DEFAULT_STOCKFISH_HASH_MB) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    options = {}
    if threads > 1:
        options["Threads"] = threads
    if hash_mb > 0:
        options["Hash"] = hash_mb
    if options:
        engine.configure(options)
    return engine


def score_to_cp(score: chess.engine.PovScore) -> int:
    cp = score.relative.score(mate_score=MATE_SCORE)
    return int(cp) if cp is not None else 0


def evaluate_position_and_move(engine, board: chess.Board, human_move: chess.Move,
                               depth: int) -> tuple[int, str, int, int]:
    limit = chess.engine.Limit(depth=depth)
    info = engine.analyse(board, limit)
    if not info.get("pv"):
        raise RuntimeError("Stockfish returned no principal variation")
    best_move = info["pv"][0]
    best_cp = score_to_cp(info["score"])
    if human_move == best_move:
        return 0, best_move.uci(), best_cp, best_cp

    board.push(human_move)
    try:
        human_cp = -score_to_cp(engine.analyse(board, limit)["score"])
    finally:
        board.pop()
    return compute_cp_loss(best_cp, human_cp), best_move.uci(), best_cp, human_cp


@dataclass
class ValidationStats:
    examples_seen: int = 0
    examples_kept: int = 0
    rejected_missing_fen: int = 0
    rejected_missing_target: int = 0
    rejected_invalid_fen: int = 0
    rejected_illegal_target: int = 0
    rejected_engine_error: int = 0
    rejected_cp_loss: int = 0
    sum_cp_loss_kept: int = 0
    input_file: str = ""
    output_file: str = ""
    stockfish_path: str = ""
    depth: int = DEFAULT_DEPTH
    max_cp_loss: int = DEFAULT_MAX_CP_LOSS
    threads: int = DEFAULT_STOCKFISH_THREADS
    duration_seconds: float = 0.0
    examples_per_second: float = 0.0
    skipped_resume: int = 0

    @property
    def examples_rejected(self) -> int:
        return max(0, self.examples_seen - self.skipped_resume - self.examples_kept)

    @property
    def avg_cp_loss_kept(self) -> float:
        return self.sum_cp_loss_kept / self.examples_kept if self.examples_kept else 0.0

    @property
    def keep_rate(self) -> float:
        processed = self.examples_seen - self.skipped_resume
        return self.examples_kept / processed if processed else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["examples_rejected"] = self.examples_rejected
        d["avg_cp_loss_kept"] = round(self.avg_cp_loss_kept, 2)
        d["keep_rate"] = round(self.keep_rate, 4)
        return d

    def update_speed(self, started: float) -> None:
        self.duration_seconds = time.monotonic() - started
        if self.duration_seconds > 0:
            self.examples_per_second = (self.examples_seen - self.skipped_resume) / self.duration_seconds

    def save(self, output_path: Path) -> None:
        path = output_path.with_suffix(output_path.suffix + ".stats.json")
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def log_summary(self) -> None:
        logger.info("Seen %d examples (%d skipped on resume), kept %d (%.1f%%), avg cp loss %.1f",
                    self.examples_seen, self.skipped_resume, self.examples_kept,
                    100 * self.keep_rate, self.avg_cp_loss_kept)
        logger.info("Rejected: cp_loss>%d=%d missing_fen=%d missing_target=%d invalid_fen=%d "
                    "illegal_target=%d engine_error=%d",
                    self.max_cp_loss, self.rejected_cp_loss, self.rejected_missing_fen,
                    self.rejected_missing_target, self.rejected_invalid_fen,
                    self.rejected_illegal_target, self.rejected_engine_error)
        logger.info("Took %.1f s (%.2f examples/s)", self.duration_seconds, self.examples_per_second)


def classify_line(raw_line: str, engine, depth: int, max_cp_loss: int) -> tuple[str, dict | None, int]:
    line = raw_line.strip()
    if not line:
        return "invalid_json", None, 0
    try:
        example = json.loads(line)
    except json.JSONDecodeError:
        return "invalid_json", None, 0

    text = example.get("text", "")
    fen = extract_fen_from_text(text)
    target = extract_ground_truth_from_text(text)
    if not fen:
        return "missing_fen", None, 0
    if not target:
        return "missing_target", None, 0

    try:
        board = chess.Board(fen)
    except ValueError:
        return "invalid_fen", None, 0
    try:
        human_move = chess.Move.from_uci(target)
    except ValueError:
        return "illegal_target", None, 0
    if human_move not in board.legal_moves:
        return "illegal_target", None, 0

    try:
        cp_loss, best_uci, best_cp, human_cp = evaluate_position_and_move(engine, board, human_move, depth)
    except (chess.engine.EngineError, chess.engine.EngineTerminatedError, RuntimeError):
        return "engine_error", None, 0
    if cp_loss > max_cp_loss:
        return "cp_loss", None, cp_loss

    meta = example.setdefault("metadata", {})
    meta["sf_best_uci"] = best_uci
    meta["sf_best_cp"] = best_cp
    meta["sf_human_cp"] = human_cp
    meta["sf_cp_loss"] = cp_loss
    meta["sf_depth"] = depth
    return "kept", example, cp_loss


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def record(stats: ValidationStats, status: str, example: dict | None, cp_loss: int, out,
           output_path: Path, started: float, checkpoint_every: int, write_stats: bool) -> None:
    if status != "kept":
        counter = REJECT_COUNTERS[status]
        setattr(stats, counter, getattr(stats, counter) + 1)
        return

    out.write(json.dumps(example, ensure_ascii=False) + "\n")
    stats.examples_kept += 1
    stats.sum_cp_loss_kept += cp_loss
    if stats.examples_kept % checkpoint_every == 0:
        out.flush()
        if write_stats:
            stats.update_speed(started)
            stats.save(output_path)


def validate_jsonl(input_path: Path, output_path: Path, engine, depth: int = DEFAULT_DEPTH,
                   max_cp_loss: int = DEFAULT_MAX_CP_LOSS, limit: int | None = None,
                   checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
                   write_stats: bool = True) -> ValidationStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = count_lines(output_path)
    if already_done:
        logger.info("Resuming: output already has %d lines", already_done)
    stats = ValidationStats(input_file=str(input_path), output_file=str(output_path),
                            depth=depth, max_cp_loss=max_cp_loss)
    started = time.monotonic()

    with input_path.open("r", encoding="utf-8") as src, \
            output_path.open("a" if already_done else "w", encoding="utf-8") as out:
        for raw_line in tqdm(src, desc="Validating", unit="ex"):
            line = raw_line.strip()
            if not line:
                continue
            stats.examples_seen += 1
            if stats.examples_seen <= already_done:
                stats.skipped_resume += 1
                continue
            if limit is not None and stats.examples_seen - already_done > limit:
                break

            status, example, cp_loss = classify_line(line, engine, depth, max_cp_loss)
            if status == "engine_error":
                logger.warning("Engine error on line %d", stats.examples_seen)
            record(stats, status, example, cp_loss, out, output_path, started,
                   checkpoint_every, write_stats)

    stats.update_speed(started)
    if write_stats:
        stats.save(output_path)
    stats.log_summary()
    return stats


_worker_engine = None


def _init_worker_engine(stockfish_path: str, threads: int, hash_mb: int) -> None:
    global _worker_engine
    _worker_engine = open_engine(stockfish_path, threads=threads, hash_mb=hash_mb)


def _validate_one_line_worker(args: tuple) -> tuple[str, dict | None, int]:
    raw_line, depth, max_cp_loss = args
    return classify_line(raw_line, _worker_engine, depth, max_cp_loss)


def validate_jsonl_parallel(input_path: Path, output_path: Path, stockfish_path: str, workers: int,
                            depth: int = DEFAULT_DEPTH, max_cp_loss: int = DEFAULT_MAX_CP_LOSS,
                            sf_threads: int = 1, hash_mb: int = DEFAULT_STOCKFISH_HASH_MB,
                            limit: int | None = None,
                            checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
                            chunksize: int = 32, write_stats: bool = True) -> ValidationStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = count_lines(output_path)
    if already_done:
        logger.info("Resuming: output already has %d lines", already_done)
    stats = ValidationStats(input_file=str(input_path), output_file=str(output_path),
                            stockfish_path=stockfish_path, depth=depth, max_cp_loss=max_cp_loss,
                            threads=sf_threads, skipped_resume=already_done)
    started = time.monotonic()

    def jobs(src):
        skipped = 0
        sent = 0
        for raw_line in src:
            if not raw_line.strip():
                continue
            if skipped < already_done:
                skipped += 1
                continue
            if limit is not None and sent >= limit:
                return
            sent += 1
            yield raw_line, depth, max_cp_loss

    with Pool(processes=workers, initializer=_init_worker_engine,
              initargs=(stockfish_path, sf_threads, hash_mb)) as pool, \
            input_path.open("r", encoding="utf-8") as src, \
            output_path.open("a" if already_done else "w", encoding="utf-8") as out, \
            tqdm(desc=f"Validating ({workers} engines)", unit="ex") as bar:
        for status, example, cp_loss in pool.imap(_validate_one_line_worker, jobs(src),
                                                  chunksize=chunksize):
            stats.examples_seen += 1
            bar.update(1)
            record(stats, status, example, cp_loss, out, output_path, started,
                   checkpoint_every, write_stats)

    stats.examples_seen += already_done
    stats.update_speed(started)
    if write_stats:
        stats.save(output_path)
    stats.log_summary()
    return stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Drop training examples where the human move "
                                            "lost too much according to Stockfish.")
    p.add_argument("--input", required=True, type=Path, help="JSONL file from pgn_to_training.py")
    p.add_argument("--output", required=True, type=Path, help="JSONL with the kept examples")
    p.add_argument("--stockfish", required=True, help="path to the Stockfish binary")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--max-cp-loss", type=int, default=DEFAULT_MAX_CP_LOSS)
    p.add_argument("--threads", type=int, default=DEFAULT_STOCKFISH_THREADS,
                   help="threads per engine; keep 1 when using several workers")
    p.add_argument("--workers", type=int, default=0,
                   help="number of engines in parallel; 0 = one per CPU")
    p.add_argument("--hash-mb", type=int, default=DEFAULT_STOCKFISH_HASH_MB)
    p.add_argument("--limit", type=int, default=None, help="stop after this many new examples")
    p.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                   help="save stats every N kept examples")
    p.add_argument("--no-stats", action="store_true", help="don't write the .stats.json file")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    if args.max_cp_loss < 0:
        logger.error("--max-cp-loss must be >= 0")
        return 2
    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        return 2

    workers = args.workers if args.workers > 0 else os.cpu_count() or 1
    logger.info("Stockfish %s, depth %d, max cp loss %d, %d workers x %d threads",
                args.stockfish, args.depth, args.max_cp_loss, workers, args.threads)

    if workers > 1:
        stats = validate_jsonl_parallel(
            args.input, args.output,
            stockfish_path=args.stockfish,
            workers=workers,
            depth=args.depth,
            max_cp_loss=args.max_cp_loss,
            sf_threads=args.threads,
            hash_mb=args.hash_mb,
            limit=args.limit,
            checkpoint_every=args.checkpoint_every,
            write_stats=not args.no_stats,
        )
    else:
        engine = open_engine(args.stockfish, threads=args.threads, hash_mb=args.hash_mb)
        try:
            stats = validate_jsonl(
                args.input, args.output, engine,
                depth=args.depth,
                max_cp_loss=args.max_cp_loss,
                limit=args.limit,
                checkpoint_every=args.checkpoint_every,
                write_stats=not args.no_stats,
            )
        finally:
            engine.quit()
    return 0 if stats.examples_kept > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
