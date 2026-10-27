from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess

from model.config.training_config import BEST_HUB_REPO
from model.inference.exact_scoring import legal_move_distribution
from model.inference.factory import build_predictor
from model.inference.legal_move_filter import filter_to_legal
from model.inference.stockfish_rerank import CANDIDATE_SOURCES, DEFAULT_CANDIDATE_SOURCE

MODE_STOCKFISH_RERANK = "stockfish-rerank"
MODE_PREFIX_CONSTRAINED = "prefix-constrained"
MODE_EXACT = "exact"
MODE_GREEDY = "greedy"
ALL_MODES = [MODE_STOCKFISH_RERANK, MODE_PREFIX_CONSTRAINED, MODE_EXACT, MODE_GREEDY]
DEFAULT_MODE = MODE_STOCKFISH_RERANK


def format_eval_pawns(cp: int) -> str:
    return f"{cp / 100.0:+.2f}"


def classify_eval(cp: int) -> str:
    size = abs(cp)
    if size < 50:
        return "near equal"
    if size < 150:
        return "slight advantage" if cp > 0 else "slight disadvantage"
    if size < 500:
        return "clear advantage" if cp > 0 else "clear disadvantage"
    return "winning" if cp > 0 else "losing"


def _san(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except (ValueError, AssertionError):
        return move.uci()


def _header(board: chess.Board) -> list[str]:
    return [
        f"Position: {board.fen()}",
        f"Side to move: {'White' if board.turn == chess.WHITE else 'Black'}",
        f"Legal moves available: {board.legal_moves.count()}",
        "",
    ]


def format_stockfish_rerank_output(result, board: chess.Board, show_candidates: bool = False) -> str:
    move = result.move
    lines = _header(board)
    lines.append(f"Recommended move: {move.uci()} ({_san(board, move)})")
    lines.append(f"  Stockfish eval after the move: {format_eval_pawns(result.stockfish_eval_cp)} "
                 f"pawns ({classify_eval(result.stockfish_eval_cp)})")

    if result.selected_was_llm_top:
        lines.append("  LLM ranked this candidate #1 and Stockfish agreed")
    else:
        rank = f" (#{result.all_candidates.index(move) + 1})" if move in result.all_candidates else ""
        lines.append(f"  Stockfish overrode the LLM's #1 pick{rank}")
        lines.append(f"  The LLM's top pick was {result.llm_top_pick.uci()} "
                     f"({_san(board, result.llm_top_pick)})")

    if show_candidates and result.all_candidates:
        lines.append("")
        lines.append(f"Top {len(result.all_candidates)} candidates scored by Stockfish:")
        by_score = sorted(enumerate(zip(result.all_candidates, result.all_scores_cp)),
                          key=lambda item: -item[1][1])
        for sf_rank, (llm_rank, (candidate, score)) in enumerate(by_score, start=1):
            marker = "  <- SELECTED" if candidate == move else ""
            lines.append(f"  {sf_rank}. {candidate.uci()} ({_san(board, candidate)}) "
                         f"{format_eval_pawns(score)} pawns (LLM #{llm_rank + 1}){marker}")
    return "\n".join(lines)


def format_basic_output(move: chess.Move, board: chess.Board, mode_label: str,
                        extra_lines: list[str] | None = None) -> str:
    lines = _header(board)
    lines.append(f"Recommended move: {move.uci()} ({_san(board, move)})")
    lines.append(f"  Mode: {mode_label}")
    lines.extend(extra_lines or [])
    return "\n".join(lines)


def format_exact_output(ranked, board: chess.Board, show_candidates: bool = False,
                        top_n: int = 10) -> str:
    probs = {m.uci(): p for m, p in legal_move_distribution(ranked)}
    best = ranked[0]
    lines = _header(board)
    lines.append(f"Recommended move: {best.move.uci()} ({_san(board, best.move)})")
    lines.append(f"  Mode: exact scoring of all {len(ranked)} legal moves")
    lines.append(f"  Model confidence: {100 * probs[best.move.uci()]:.1f}% of the probability "
                 f"over legal moves")

    if show_candidates:
        lines.append("")
        lines.append(f"Top {min(top_n, len(ranked))} moves by model probability:")
        for s in ranked[:top_n]:
            marker = "  <- SELECTED" if s.rank == 1 else ""
            lines.append(f"  {s.rank}. {s.move.uci()} ({_san(board, s.move)}) "
                         f"{100 * probs[s.move.uci()]:.1f}% (logp {s.logprob:.2f}){marker}")
    return "\n".join(lines)


def _construct_predictor(mode: str, args: argparse.Namespace):
    return build_predictor(args.adapter, mode, stockfish_path=args.stockfish,
                           rerank_depth=args.stockfish_depth, rerank_k=args.num_candidates,
                           rerank_candidates=args.rerank_candidates)


def _run_inference(predictor, mode: str, fen: str, history: str, side: str):
    if mode == MODE_STOCKFISH_RERANK:
        return predictor.predict_reranked(fen, history, side)
    if mode == MODE_PREFIX_CONSTRAINED:
        return predictor.predict_prefix_constrained(fen, history, side)
    if mode == MODE_EXACT:
        return predictor.rank_legal_moves(fen, history, side)
    return filter_to_legal(predictor.predict_raw(fen, history, side), chess.Board(fen))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Suggest a move for one position.")
    p.add_argument("--adapter", default=BEST_HUB_REPO, help="Hugging Face repo or local folder")
    p.add_argument("--fen", required=True)
    p.add_argument("--history", default="(none — opening)",
                   help="the last moves in UCI, space separated; the model was trained with "
                        "6-10 plies of history, so pass it whenever you have it")
    p.add_argument("--side", choices=["White", "Black"], help="read from the FEN if not given")
    p.add_argument("--mode", choices=ALL_MODES, default=DEFAULT_MODE)
    p.add_argument("--show-candidates", action="store_true")
    p.add_argument("--rerank-candidates", choices=CANDIDATE_SOURCES, default=DEFAULT_CANDIDATE_SOURCE)
    p.add_argument("--stockfish", default="/usr/games/stockfish")
    p.add_argument("--stockfish-depth", type=int, default=6)
    p.add_argument("--num-candidates", type=int, default=10)
    p.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    try:
        board = chess.Board(args.fen)
    except ValueError as e:
        print(f"Invalid FEN: {e}", file=sys.stderr)
        return 1
    if board.legal_moves.count() == 0:
        print("The position has no legal moves (checkmate or stalemate).", file=sys.stderr)
        return 1
    side = args.side or ("White" if board.turn == chess.WHITE else "Black")

    print("Loading the model, this takes a minute the first time...", file=sys.stderr)
    predictor = _construct_predictor(args.mode, args)
    try:
        result = _run_inference(predictor, args.mode, args.fen, args.history, side)
    finally:
        if hasattr(predictor, "close"):
            predictor.close()

    if args.mode == MODE_STOCKFISH_RERANK:
        print(format_stockfish_rerank_output(result, board, args.show_candidates))
    elif args.mode == MODE_EXACT:
        print(format_exact_output(result, board, args.show_candidates))
    elif args.mode == MODE_PREFIX_CONSTRAINED:
        print(format_basic_output(result.filtered, board, "prefix-constrained (token-level legality)"))
    else:
        print(format_basic_output(result.filtered, board, "greedy + post-hoc filter"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
