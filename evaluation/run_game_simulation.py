from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluation.game_simulation as gs
from model.inference.stockfish_rerank import CANDIDATE_SOURCES, DEFAULT_CANDIDATE_SOURCE

DEFAULT_OUR_AGENT = "llm:stockfish-rerank"
DEFAULT_OPPONENTS = ["random", "stockfish:1", "zero-shot:gpt2", "zero-shot:mistralai/Mistral-7B-v0.1"]

logger = logging.getLogger(__name__)


def spec_tag(spec: str) -> str:
    return spec.replace("zero-shot:", "zeroshot-").replace(":", "-").replace("/", "-")


def output_path_for(output_dir: Path, our_spec: str, opponent: str) -> Path:
    return output_dir / f"game_sim_{spec_tag(our_spec)}_vs_{spec_tag(opponent)}.json"


def build_matchup_argv(our_spec: str, opponent: str, *, adapter: str | None, stockfish: str,
                       games: int, max_plies: int, rerank_depth: int, rerank_k: int, seed: int,
                       output: Path, rerank_candidates: str = DEFAULT_CANDIDATE_SOURCE) -> list[str]:
    argv = [
        "--agent-a", our_spec,
        "--agent-b", opponent,
        "--games", str(games),
        "--max-plies", str(max_plies),
        "--stockfish", stockfish,
        "--rerank-depth", str(rerank_depth),
        "--rerank-k", str(rerank_k),
        "--rerank-candidates", rerank_candidates,
        "--seed", str(seed),
        "--output", str(output),
    ]
    if adapter:
        argv += ["--adapter", adapter]
    return argv


def summarize(results: list[tuple[str, dict]]) -> list[dict]:
    return [{
        "opponent": opponent,
        "n_games": d.get("n_games", 0),
        "wins": d.get("wins_a", 0),
        "draws": d.get("draws", 0),
        "losses": d.get("wins_b", 0),
        "win_rate": d.get("win_rate_a", 0.0),
        "score": d.get("score_a", 0.0),
    } for opponent, d in results]


def format_table(our_spec: str, rows: list[dict]) -> str:
    lines = [
        f"Games played by {our_spec}",
        f"{'opponent':<40}{'W':>4}{'D':>4}{'L':>4}{'win%':>8}{'score%':>8}",
        "-" * 68,
    ]
    for r in rows:
        lines.append(f"{r['opponent']:<40}{r['wins']:>4}{r['draws']:>4}{r['losses']:>4}"
                     f"{100 * r['win_rate']:>7.1f}{100 * r['score']:>8.1f}")
    return "\n".join(lines)


def run_matrix(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for opponent in args.opponents:
        out_path = output_path_for(output_dir, args.our_agent, opponent)
        logger.info("%s vs %s", args.our_agent, opponent)
        argv = build_matchup_argv(args.our_agent, opponent, adapter=args.adapter,
                                  stockfish=args.stockfish, games=args.games,
                                  max_plies=args.max_plies, rerank_depth=args.rerank_depth,
                                  rerank_k=args.rerank_k, seed=args.seed, output=out_path,
                                  rerank_candidates=args.rerank_candidates)
        try:
            gs.main(argv)
            results.append((opponent, json.loads(out_path.read_text(encoding="utf-8"))))
        except Exception as e:
            logger.error("%s vs %s failed: %s", args.our_agent, opponent, e)
            results.append((opponent, {"error": str(e)}))

    rows = summarize(results)
    logger.info("\n%s", format_table(args.our_agent, rows))
    (output_dir / "game_simulation_summary.json").write_text(
        json.dumps({"our_agent": args.our_agent, "rows": rows}, indent=2), encoding="utf-8")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play our model against every baseline.")
    p.add_argument("--adapter", default=None)
    p.add_argument("--our-agent", default=DEFAULT_OUR_AGENT)
    p.add_argument("--opponents", nargs="+", default=DEFAULT_OPPONENTS)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--max-plies", type=int, default=gs.DEFAULT_MAX_PLIES)
    p.add_argument("--stockfish", default="/usr/games/stockfish")
    p.add_argument("--rerank-depth", type=int, default=6)
    p.add_argument("--rerank-k", type=int, default=10)
    p.add_argument("--rerank-candidates", choices=list(CANDIDATE_SOURCES), default=DEFAULT_CANDIDATE_SOURCE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True, help="where the game_sim_*.json files go")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    return run_matrix(args)


if __name__ == "__main__":
    sys.exit(main())
