from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import chess
import chess.pgn
from tqdm.auto import tqdm

DEFAULT_POSITIONS_PER_GAME = 10
DEFAULT_HISTORY_MOVES = 5
DEFAULT_SKIP_OPENING_PLY = 6
DEFAULT_SEED = 42

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "You are a chess grandmaster. Analyze the current position and suggest "
    "the best next move. Respond ONLY with the move in UCI format "
    "(e.g., e2e4, g1f3, e1g1 for castling).\n\n"
    "Position (FEN): {fen}\n"
    "Move history (last {history_n} moves): {history}\n"
    "Side to move: {side}\n\n"
    "### Response:\n"
    "{target_move}"
)

logger = logging.getLogger(__name__)


def sample_ply_indices(total_ply: int, n_samples: int, skip_opening_ply: int,
                       rng: random.Random) -> list[int]:
    candidates = list(range(skip_opening_ply, total_ply))
    if n_samples >= len(candidates):
        return candidates
    return sorted(rng.sample(candidates, n_samples))


def position_at_ply(moves: list[chess.Move], ply_index: int) -> tuple[str, str, bool, list[str]]:
    board = chess.Board()
    for move in moves[:ply_index]:
        board.push(move)
    history = [move.uci() for move in moves[:ply_index]]
    return board.fen(), moves[ply_index].uci(), board.turn == chess.WHITE, history


def trim_history(history_uci: list[str], history_full_moves: int) -> str:
    plies = history_full_moves * 2
    tail = history_uci[-plies:] if plies > 0 else []
    return " ".join(tail) if tail else "(none — opening)"


def format_training_example(fen: str, history_uci: list[str], side_white: bool, target_uci: str,
                            history_full_moves: int = DEFAULT_HISTORY_MOVES) -> str:
    return PROMPT_TEMPLATE.format(
        fen=fen,
        history_n=history_full_moves,
        history=trim_history(history_uci, history_full_moves),
        side="White" if side_white else "Black",
        target_move=target_uci,
    )


def _safe_int(s: str | None) -> int | None:
    try:
        return int(s) if s and s != "?" else None
    except (ValueError, TypeError):
        return None


def extract_examples_from_game(game: chess.pgn.Game, n_positions: int, history_full_moves: int,
                               skip_opening_ply: int, rng: random.Random):
    moves = list(game.mainline_moves())
    headers = game.headers
    game_info = {
        "white_elo": _safe_int(headers.get("WhiteElo")),
        "black_elo": _safe_int(headers.get("BlackElo")),
        "result": headers.get("Result", ""),
        "time_control": headers.get("TimeControl", ""),
        "site": headers.get("Site", ""),
    }
    for ply in sample_ply_indices(len(moves), n_positions, skip_opening_ply, rng):
        fen, target, side_white, history = position_at_ply(moves, ply)
        yield {
            "text": format_training_example(fen, history, side_white, target, history_full_moves),
            "metadata": {
                **game_info,
                "ply": ply,
                "side_to_move": "white" if side_white else "black",
            },
        }


@dataclass
class ConversionStats:
    games_seen: int = 0
    games_used: int = 0
    games_too_short: int = 0
    examples_written: int = 0
    inputs: list[str] = field(default_factory=list)
    output: str = ""
    positions_per_game: int = DEFAULT_POSITIONS_PER_GAME
    history_full_moves: int = DEFAULT_HISTORY_MOVES
    skip_opening_ply: int = DEFAULT_SKIP_OPENING_PLY
    seed: int = DEFAULT_SEED
    duration_seconds: float = 0.0
    side_to_move_white: int = 0
    side_to_move_black: int = 0


def read_games(paths: list[Path]):
    for path in paths:
        logger.info("Reading %s", path)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            while True:
                try:
                    game = chess.pgn.read_game(f)
                except (ValueError, RuntimeError):
                    continue
                if game is None:
                    break
                yield game


def convert_pgn_files(inputs: list[Path], output: Path,
                      positions_per_game: int = DEFAULT_POSITIONS_PER_GAME,
                      history_full_moves: int = DEFAULT_HISTORY_MOVES,
                      skip_opening_ply: int = DEFAULT_SKIP_OPENING_PLY,
                      seed: int = DEFAULT_SEED,
                      max_examples: int | None = None,
                      write_stats: bool = True) -> ConversionStats:
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    stats = ConversionStats(inputs=[str(p) for p in inputs], output=str(output),
                            positions_per_game=positions_per_game,
                            history_full_moves=history_full_moves,
                            skip_opening_ply=skip_opening_ply, seed=seed)
    started = time.monotonic()

    with output.open("w", encoding="utf-8") as out, tqdm(desc="Converting", unit="game") as bar:
        for game in read_games(inputs):
            stats.games_seen += 1
            bar.update(1)

            produced = 0
            for example in extract_examples_from_game(game, positions_per_game, history_full_moves,
                                                      skip_opening_ply, rng):
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                stats.examples_written += 1
                if example["metadata"]["side_to_move"] == "white":
                    stats.side_to_move_white += 1
                else:
                    stats.side_to_move_black += 1
                produced += 1
                if max_examples is not None and stats.examples_written >= max_examples:
                    break
            if max_examples is not None and stats.examples_written >= max_examples:
                break

            if produced:
                stats.games_used += 1
            else:
                stats.games_too_short += 1

    stats.duration_seconds = time.monotonic() - started
    logger.info("Games seen %d, used %d, too short %d; wrote %d examples (white %d, black %d)",
                stats.games_seen, stats.games_used, stats.games_too_short, stats.examples_written,
                stats.side_to_move_white, stats.side_to_move_black)

    if write_stats:
        stats_path = output.with_suffix(output.suffix + ".stats.json")
        stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    return stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Turn filtered PGN games into JSONL training examples.")
    p.add_argument("--inputs", nargs="+", required=True, type=Path, help="filtered .pgn files")
    p.add_argument("--output", required=True, type=Path, help="output .jsonl file")
    p.add_argument("--positions-per-game", type=int, default=DEFAULT_POSITIONS_PER_GAME)
    p.add_argument("--history-moves", type=int, default=DEFAULT_HISTORY_MOVES,
                   help="full moves of history in the prompt (5 = 10 plies)")
    p.add_argument("--skip-opening-ply", type=int, default=DEFAULT_SKIP_OPENING_PLY)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-examples", type=int, default=None, help="stop after this many examples")
    p.add_argument("--no-stats", action="store_true", help="don't write the .stats.json file")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    stats = convert_pgn_files(
        args.inputs, args.output,
        positions_per_game=args.positions_per_game,
        history_full_moves=args.history_moves,
        skip_opening_ply=args.skip_opening_ply,
        seed=args.seed,
        max_examples=args.max_examples,
        write_stats=not args.no_stats,
    )
    return 0 if stats.examples_written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
