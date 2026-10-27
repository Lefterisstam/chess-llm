from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm.auto import tqdm

DEFAULT_VAL_PCT = 10
DEFAULT_TEST_PCT = 10
DEFAULT_SEED = 42
SPLITS = ("train", "val", "test")

logger = logging.getLogger(__name__)


def assign_split(identifier: str, seed: int = DEFAULT_SEED, val_pct: int = DEFAULT_VAL_PCT,
                 test_pct: int = DEFAULT_TEST_PCT) -> str:
    if val_pct < 0 or test_pct < 0 or val_pct + test_pct >= 100:
        raise ValueError(f"Invalid percentages: val={val_pct}, test={test_pct} (sum must be < 100)")
    digest = hashlib.sha1(f"{identifier}|{seed}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 100 - val_pct - test_pct:
        return "train"
    if bucket < 100 - test_pct:
        return "val"
    return "test"


def extract_game_id(metadata: dict, fallback_counter: int) -> str:
    site = (metadata or {}).get("site", "")
    if not site or site == "?":
        return f"__missing_site_{fallback_counter}__"
    return site


@dataclass
class SplitStats:
    examples_total: int = 0
    examples_train: int = 0
    examples_val: int = 0
    examples_test: int = 0
    games_total: int = 0
    games_train: int = 0
    games_val: int = 0
    games_test: int = 0
    missing_site_count: int = 0
    input_file: str = ""
    output_dir: str = ""
    val_pct: int = DEFAULT_VAL_PCT
    test_pct: int = DEFAULT_TEST_PCT
    seed: int = DEFAULT_SEED
    duration_seconds: float = 0.0


def split_jsonl_file(input_path: Path, output_dir: Path, val_pct: int = DEFAULT_VAL_PCT,
                     test_pct: int = DEFAULT_TEST_PCT, seed: int = DEFAULT_SEED,
                     write_stats: bool = True) -> SplitStats:
    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = SplitStats(input_file=str(input_path), output_dir=str(output_dir),
                       val_pct=val_pct, test_pct=test_pct, seed=seed)
    games = {name: set() for name in SPLITS}
    started = time.monotonic()

    outputs = {name: (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") for name in SPLITS}
    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Splitting", unit="example"):
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping a line that is not valid JSON")
                    continue

                game_id = extract_game_id(example.get("metadata"), stats.missing_site_count)
                if game_id.startswith("__missing_site_"):
                    stats.missing_site_count += 1

                split = assign_split(game_id, seed, val_pct, test_pct)
                outputs[split].write(line if line.endswith("\n") else line + "\n")
                games[split].add(game_id)
                stats.examples_total += 1
                setattr(stats, f"examples_{split}", getattr(stats, f"examples_{split}") + 1)
    finally:
        for out in outputs.values():
            out.close()

    if games["train"] & games["val"] or games["train"] & games["test"] or games["val"] & games["test"]:
        raise AssertionError("A game ended up in more than one split")

    stats.games_train = len(games["train"])
    stats.games_val = len(games["val"])
    stats.games_test = len(games["test"])
    stats.games_total = stats.games_train + stats.games_val + stats.games_test
    stats.duration_seconds = time.monotonic() - started

    logger.info("Games train/val/test: %d / %d / %d", stats.games_train, stats.games_val,
                stats.games_test)
    logger.info("Examples train/val/test: %d / %d / %d", stats.examples_train, stats.examples_val,
                stats.examples_test)
    if stats.missing_site_count:
        logger.warning("%d examples had no game id and were split one by one",
                       stats.missing_site_count)

    if write_stats:
        (output_dir / "split.stats.json").write_text(json.dumps(asdict(stats), indent=2),
                                                     encoding="utf-8")
    return stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split examples into train/val/test by game.")
    p.add_argument("--input", required=True, type=Path, help="JSONL file from pgn_to_training.py")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--val-pct", type=int, default=DEFAULT_VAL_PCT)
    p.add_argument("--test-pct", type=int, default=DEFAULT_TEST_PCT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--no-stats", action="store_true", help="don't write split.stats.json")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    stats = split_jsonl_file(args.input, args.output_dir, val_pct=args.val_pct,
                             test_pct=args.test_pct, seed=args.seed,
                             write_stats=not args.no_stats)
    return 0 if stats.examples_total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
