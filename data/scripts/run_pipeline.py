from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MIN_ELO = 2200
DEFAULT_MIN_BASE_SECONDS = 480
DEFAULT_MIN_FULL_MOVES = 20
DEFAULT_MAX_FULL_MOVES = 150
DEFAULT_STOCKFISH_DEPTH = 12
DEFAULT_MAX_CP_LOSS = 50
DEFAULT_VAL_PCT = 10
DEFAULT_TEST_PCT = 10

logger = logging.getLogger(__name__)


def run_script(name: str, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / name), *args]
    logger.info("Running: %s", " ".join(cmd))
    started = time.monotonic()
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")
    logger.info("%s finished in %.1f s", name, time.monotonic() - started)


def is_done(*paths: Path) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def month_name(path: Path) -> str:
    name = path.name.removeprefix("lichess_db_standard_rated_")
    for suffix in (".pgn.zst", ".pgn"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def step_filter(raw_dir: Path, filtered_dir: Path, args) -> list[Path]:
    raw_files = sorted(list(raw_dir.glob("*.pgn.zst")) + list(raw_dir.glob("*.pgn")))
    if not raw_files:
        raise SystemExit(f"No .pgn or .pgn.zst files in {raw_dir}")
    filtered_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for raw in raw_files:
        out = filtered_dir / f"{month_name(raw)}.pgn"
        if raw.resolve() == out.resolve():
            raise SystemExit(f"Input and output are the same file ({raw}), use another --work-dir")
        outputs.append(out)
        if is_done(out) and not args.force:
            logger.info("Skipping filter, %s already exists", out.name)
            continue
        run_script("filter_games.py",
                   "--input", str(raw),
                   "--output", str(out),
                   "--min-elo", str(args.min_elo),
                   "--min-base-seconds", str(args.min_base_seconds),
                   "--min-moves", str(args.min_full_moves),
                   "--max-moves", str(args.max_full_moves),
                   "--workers", "0")
    return outputs


def _resolve_step_toggles(args) -> tuple[bool, bool, bool, bool]:
    only = (args.only_filter, args.only_convert, args.only_stockfish, args.only_split)
    if any(only):
        return only
    return (not args.skip_filter, not args.skip_convert, not args.skip_stockfish,
            not args.skip_split)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the whole data pipeline: filter, convert, "
                                            "validate with Stockfish, split.")
    p.add_argument("--raw-dir", required=True, type=Path, help="folder with the Lichess dumps")
    p.add_argument("--work-dir", required=True, type=Path,
                   help="filtered/ and processed/ are created here")
    p.add_argument("--stockfish", default="/usr/games/stockfish")
    p.add_argument("--min-elo", type=int, default=DEFAULT_MIN_ELO)
    p.add_argument("--min-base-seconds", type=int, default=DEFAULT_MIN_BASE_SECONDS)
    p.add_argument("--min-moves", type=int, default=DEFAULT_MIN_FULL_MOVES, dest="min_full_moves")
    p.add_argument("--max-moves", type=int, default=DEFAULT_MAX_FULL_MOVES, dest="max_full_moves")
    p.add_argument("--depth", type=int, default=DEFAULT_STOCKFISH_DEPTH)
    p.add_argument("--max-cp-loss", type=int, default=DEFAULT_MAX_CP_LOSS)
    p.add_argument("--val-pct", type=int, default=DEFAULT_VAL_PCT)
    p.add_argument("--test-pct", type=int, default=DEFAULT_TEST_PCT)
    for step in ("filter", "convert", "stockfish", "split"):
        p.add_argument(f"--skip-{step}", action="store_true")
    only = p.add_mutually_exclusive_group()
    for step in ("filter", "convert", "stockfish", "split"):
        only.add_argument(f"--only-{step}", action="store_true")
    p.add_argument("--force", action="store_true", help="redo every step")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    filtered_dir = args.work_dir / "filtered"
    processed_dir = args.work_dir / "processed"
    training_jsonl = processed_dir / "training.jsonl"
    validated_jsonl = processed_dir / "training_sf_validated.jsonl"
    splits = [processed_dir / f"{name}.jsonl" for name in ("train", "val", "test")]
    do_filter, do_convert, do_stockfish, do_split = _resolve_step_toggles(args)
    started = time.monotonic()

    if do_filter:
        filtered = step_filter(args.raw_dir, filtered_dir, args)
    else:
        filtered = sorted(filtered_dir.glob("*.pgn"))
        if do_convert and not filtered:
            raise SystemExit(f"No filtered .pgn files in {filtered_dir}")

    if do_convert:
        if is_done(training_jsonl) and not args.force:
            logger.info("Skipping convert, %s already exists", training_jsonl.name)
        else:
            processed_dir.mkdir(parents=True, exist_ok=True)
            run_script("pgn_to_training.py",
                       "--inputs", *[str(p) for p in filtered],
                       "--output", str(training_jsonl))

    if do_stockfish:
        if not training_jsonl.exists():
            raise SystemExit(f"{training_jsonl} not found, run the convert step first")
        if args.force and validated_jsonl.exists():
            validated_jsonl.unlink()
        run_script("stockfish_validate.py",
                   "--input", str(training_jsonl),
                   "--output", str(validated_jsonl),
                   "--stockfish", args.stockfish,
                   "--depth", str(args.depth),
                   "--max-cp-loss", str(args.max_cp_loss),
                   "--workers", "0",
                   "--threads", "1")

    if do_split:
        if not validated_jsonl.exists():
            raise SystemExit(f"{validated_jsonl} not found, run the stockfish step first")
        if is_done(*splits) and not args.force:
            logger.info("Skipping split, train/val/test already exist")
        else:
            run_script("split_dataset.py",
                       "--input", str(validated_jsonl),
                       "--output-dir", str(processed_dir),
                       "--val-pct", str(args.val_pct),
                       "--test-pct", str(args.test_pct))

    logger.info("Pipeline finished in %.1f min, splits are in %s",
                (time.monotonic() - started) / 60, processed_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
