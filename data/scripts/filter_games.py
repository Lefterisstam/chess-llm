from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import IO

import chess.pgn
import zstandard as zstd
from tqdm.auto import tqdm

DEFAULT_MIN_ELO = 2200
DEFAULT_MIN_FULL_MOVES = 20
DEFAULT_MAX_FULL_MOVES = 150
LICHESS_RAPID_MIN_SECONDS = 480
DEFAULT_MIN_BASE_SECONDS = LICHESS_RAPID_MIN_SECONDS
VALID_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})
KEEP_TERMINATION = "Normal"
RESULT_TOKENS = ("1-0", "0-1", "1/2-1/2", "*")

MOVE_NUMBER_RE = re.compile(r"(?:^|\s)(\d+)\.")
COMMENT_RE = re.compile(r"\{[^}]*\}")

logger = logging.getLogger(__name__)


def parse_time_control(tc: str | None) -> tuple[int, int] | None:
    if not tc:
        return None
    tc = tc.strip()
    if tc == "-":
        return (-1, 0)
    try:
        if "+" in tc:
            base, inc = tc.split("+", 1)
            return (int(base), int(inc))
        return (int(tc), 0)
    except ValueError:
        return None


def is_below_min_time(tc: str | None, min_base_seconds: int) -> bool:
    parsed = parse_time_control(tc)
    if parsed is None or parsed[0] < 0:
        return False
    return parsed[0] < min_base_seconds


def extract_elo(headers, side: str) -> int | None:
    raw = headers.get(f"{side}Elo", "")
    if not raw or raw == "?":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def passes_elo_filter(headers, min_elo: int) -> bool:
    white = extract_elo(headers, "White")
    black = extract_elo(headers, "Black")
    return white is not None and black is not None and white >= min_elo and black >= min_elo


def passes_termination_filter(headers) -> bool:
    return headers.get("Termination", "") == KEEP_TERMINATION


def passes_result_filter(headers) -> bool:
    return headers.get("Result", "") in VALID_RESULTS


def passes_timecontrol_filter(headers, min_base_seconds: int = DEFAULT_MIN_BASE_SECONDS) -> bool:
    tc = headers.get("TimeControl", "")
    return parse_time_control(tc) is not None and not is_below_min_time(tc, min_base_seconds)


def count_full_moves(game: chess.pgn.Game) -> int:
    plies = sum(1 for _ in game.mainline_moves())
    return (plies + 1) // 2


def passes_length_filter(game: chess.pgn.Game, min_full: int, max_full: int) -> bool:
    return min_full <= count_full_moves(game) <= max_full


def game_signature(game: chess.pgn.Game) -> bytes:
    h = hashlib.sha1()
    for move in game.mainline_moves():
        h.update(move.uci().encode("ascii") + b" ")
    return h.digest()


@dataclass
class FilterStats:
    total_seen: int = 0
    kept: int = 0
    rejected_elo: int = 0
    rejected_timecontrol: int = 0
    rejected_termination: int = 0
    rejected_result: int = 0
    rejected_length: int = 0
    rejected_duplicate: int = 0
    rejected_malformed: int = 0
    input_file: str = ""
    output_file: str = ""
    min_elo: int = DEFAULT_MIN_ELO
    min_full_moves: int = DEFAULT_MIN_FULL_MOVES
    max_full_moves: int = DEFAULT_MAX_FULL_MOVES
    min_base_seconds: int = DEFAULT_MIN_BASE_SECONDS
    duration_seconds: float = 0.0
    games_per_second: float = 0.0
    yield_percent: float = 0.0

    def finish(self, started: float) -> None:
        self.duration_seconds = time.monotonic() - started
        if self.duration_seconds > 0:
            self.games_per_second = self.total_seen / self.duration_seconds
        if self.total_seen > 0:
            self.yield_percent = 100.0 * self.kept / self.total_seen

    def log_summary(self) -> None:
        logger.info("Seen %d games, kept %d (%.2f%%)", self.total_seen, self.kept, self.yield_percent)
        logger.info("Rejected: elo=%d timecontrol=%d termination=%d result=%d length=%d "
                    "duplicate=%d malformed=%d",
                    self.rejected_elo, self.rejected_timecontrol, self.rejected_termination,
                    self.rejected_result, self.rejected_length, self.rejected_duplicate,
                    self.rejected_malformed)
        logger.info("Took %.1f s (%.0f games/s)", self.duration_seconds, self.games_per_second)


def open_pgn(path: Path) -> IO[str]:
    if path.name.endswith(".zst"):
        reader = zstd.ZstdDecompressor().stream_reader(path.open("rb"), read_across_frames=True)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def filter_pgn_stream(input_stream: IO[str], output_stream: IO[str],
                      min_elo: int = DEFAULT_MIN_ELO,
                      min_full_moves: int = DEFAULT_MIN_FULL_MOVES,
                      max_full_moves: int = DEFAULT_MAX_FULL_MOVES,
                      min_base_seconds: int = DEFAULT_MIN_BASE_SECONDS,
                      limit: int | None = None,
                      progress: bool = True) -> FilterStats:
    stats = FilterStats(min_elo=min_elo, min_full_moves=min_full_moves,
                        max_full_moves=max_full_moves, min_base_seconds=min_base_seconds)
    seen = set()
    exporter = chess.pgn.FileExporter(output_stream)
    started = time.monotonic()

    with tqdm(desc="Filtering", unit="game", disable=not progress) as bar:
        while limit is None or stats.total_seen < limit:
            try:
                game = chess.pgn.read_game(input_stream)
            except (ValueError, RuntimeError):
                stats.rejected_malformed += 1
                continue
            if game is None:
                break
            stats.total_seen += 1
            bar.update(1)

            h = game.headers
            if not passes_result_filter(h):
                stats.rejected_result += 1
                continue
            if not passes_termination_filter(h):
                stats.rejected_termination += 1
                continue
            if not passes_elo_filter(h, min_elo):
                stats.rejected_elo += 1
                continue
            if not passes_timecontrol_filter(h, min_base_seconds):
                stats.rejected_timecontrol += 1
                continue
            if not passes_length_filter(game, min_full_moves, max_full_moves):
                stats.rejected_length += 1
                continue

            signature = game_signature(game)
            if signature in seen:
                stats.rejected_duplicate += 1
                continue
            seen.add(signature)
            game.accept(exporter)
            stats.kept += 1

    stats.finish(started)
    return stats


def _iter_pgn_game_texts(stream: IO[str]):
    buffer = []
    just_saw_result = False
    for line in stream:
        if just_saw_result and line.startswith("[Event "):
            yield "".join(buffer)
            buffer = []
            just_saw_result = False
        buffer.append(line)
        if line.rstrip().endswith(RESULT_TOKENS):
            just_saw_result = True
    if buffer:
        yield "".join(buffer)


def parse_headers(game_text: str) -> dict[str, str]:
    headers = {}
    for line in game_text.splitlines():
        line = line.strip()
        if not line:
            break
        quote = line.find(' "')
        if line.startswith("[") and line.endswith('"]') and quote > 0:
            headers[line[1:quote]] = line[quote + 2:-2]
    return headers


def check_game_text(args: tuple) -> tuple[str, str | None, bytes | None]:
    game_text, min_elo, min_base_seconds, min_full_moves, max_full_moves = args
    h = parse_headers(game_text)

    if not passes_result_filter(h):
        return "rejected_result", None, None
    if not passes_termination_filter(h):
        return "rejected_termination", None, None
    if not passes_elo_filter(h, min_elo):
        return "rejected_elo", None, None
    if not passes_timecontrol_filter(h, min_base_seconds):
        return "rejected_timecontrol", None, None

    parts = game_text.split("\n\n", 1)
    movetext = parts[1] if len(parts) > 1 else ""
    move_numbers = MOVE_NUMBER_RE.findall(movetext)
    if not move_numbers:
        return "rejected_length", None, None
    full_moves = max(int(n) for n in move_numbers)
    if not min_full_moves <= full_moves <= max_full_moves:
        return "rejected_length", None, None

    normalized = COMMENT_RE.sub("", movetext).strip()
    signature = hashlib.sha1(normalized.encode("utf-8", errors="replace")).digest()
    return "kept", game_text, signature


def filter_pgn_stream_parallel(input_stream: IO[str], output_stream: IO[str],
                               min_elo: int = DEFAULT_MIN_ELO,
                               min_full_moves: int = DEFAULT_MIN_FULL_MOVES,
                               max_full_moves: int = DEFAULT_MAX_FULL_MOVES,
                               min_base_seconds: int = DEFAULT_MIN_BASE_SECONDS,
                               limit: int | None = None,
                               workers: int = 4,
                               chunksize: int = 1024,
                               progress: bool = True) -> FilterStats:
    stats = FilterStats(min_elo=min_elo, min_full_moves=min_full_moves,
                        max_full_moves=max_full_moves, min_base_seconds=min_base_seconds)
    seen = set()
    started = time.monotonic()

    def jobs():
        for i, game_text in enumerate(_iter_pgn_game_texts(input_stream)):
            if limit is not None and i >= limit:
                return
            yield game_text, min_elo, min_base_seconds, min_full_moves, max_full_moves

    with Pool(processes=workers) as pool, \
            tqdm(desc=f"Filtering ({workers} workers)", unit="game", disable=not progress) as bar:
        for status, game_text, signature in pool.imap_unordered(check_game_text, jobs(),
                                                                chunksize=chunksize):
            stats.total_seen += 1
            bar.update(1)
            if status != "kept":
                setattr(stats, status, getattr(stats, status) + 1)
            elif signature in seen:
                stats.rejected_duplicate += 1
            else:
                seen.add(signature)
                output_stream.write(game_text if game_text.endswith("\n") else game_text + "\n")
                stats.kept += 1

    stats.finish(started)
    return stats


def filter_pgn_file(input_path: Path, output_path: Path,
                    min_elo: int = DEFAULT_MIN_ELO,
                    min_full_moves: int = DEFAULT_MIN_FULL_MOVES,
                    max_full_moves: int = DEFAULT_MAX_FULL_MOVES,
                    min_base_seconds: int = DEFAULT_MIN_BASE_SECONDS,
                    limit: int | None = None,
                    write_stats: bool = True,
                    workers: int = 1) -> FilterStats:
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Filtering %s -> %s (min elo %d, min base %ds, moves %d-%d, %d workers)",
                input_path, output_path, min_elo, min_base_seconds, min_full_moves,
                max_full_moves, workers)

    options = dict(min_elo=min_elo, min_full_moves=min_full_moves, max_full_moves=max_full_moves,
                   min_base_seconds=min_base_seconds, limit=limit)
    with open_pgn(input_path) as src, output_path.open("w", encoding="utf-8") as dst:
        if workers > 1:
            stats = filter_pgn_stream_parallel(src, dst, workers=workers, **options)
        else:
            stats = filter_pgn_stream(src, dst, **options)

    stats.input_file = str(input_path)
    stats.output_file = str(output_path)
    stats.log_summary()

    if write_stats:
        stats_path = output_path.with_suffix(output_path.suffix + ".stats.json")
        stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    return stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Keep only high-quality Lichess games.")
    p.add_argument("--input", required=True, type=Path, help=".pgn or .pgn.zst file")
    p.add_argument("--output", required=True, type=Path, help="filtered .pgn file")
    p.add_argument("--min-elo", type=int, default=DEFAULT_MIN_ELO,
                   help="minimum rating for both players")
    p.add_argument("--min-base-seconds", type=int, default=DEFAULT_MIN_BASE_SECONDS,
                   help="minimum base time; 480 keeps rapid and classical only")
    p.add_argument("--min-moves", type=int, default=DEFAULT_MIN_FULL_MOVES, dest="min_full_moves")
    p.add_argument("--max-moves", type=int, default=DEFAULT_MAX_FULL_MOVES, dest="max_full_moves")
    p.add_argument("--limit", type=int, default=None, help="stop after this many input games")
    p.add_argument("--workers", type=int, default=0, help="0 = one per CPU, 1 = no multiprocessing")
    p.add_argument("--no-stats", action="store_true", help="don't write the .stats.json file")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    stats = filter_pgn_file(
        args.input, args.output,
        min_elo=args.min_elo,
        min_full_moves=args.min_full_moves,
        max_full_moves=args.max_full_moves,
        min_base_seconds=args.min_base_seconds,
        limit=args.limit,
        write_stats=not args.no_stats,
        workers=args.workers if args.workers > 0 else os.cpu_count() or 1,
    )
    return 0 if stats.kept > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
