from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
from flask import Flask, jsonify, request, send_from_directory

from model.config.training_config import BEST_HUB_REPO
from model.inference.stockfish_rerank import CANDIDATE_SOURCES, DEFAULT_CANDIDATE_SOURCE

HISTORY_PLIES = 10
NO_HISTORY = "(none — opening)"

UCI_TOKEN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbnQRBN]?$")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
MOVE_NUMBER_PREFIX = re.compile(r"^\d+\.+")
BARE_MOVE_NUMBER = re.compile(r"^\d+\.*$")

logger = logging.getLogger(__name__)


def san_or_uci(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except (ValueError, AssertionError):
        return move.uci()


def serialize_rerank_result(result, board: chess.Board) -> dict:
    candidates = [{
        "uci": move.uci(),
        "san": san_or_uci(board, move),
        "eval_cp": int(score),
        "eval_pawns": round(score / 100.0, 2),
        "selected": move == result.move,
    } for move, score in zip(result.all_candidates, result.all_scores_cp)]

    return {
        "move_uci": result.move.uci(),
        "move_san": san_or_uci(board, result.move),
        "eval_cp": int(result.stockfish_eval_cp),
        "eval_pawns": round(result.stockfish_eval_cp / 100.0, 2),
        "llm_top_uci": result.llm_top_pick.uci(),
        "selected_was_llm_top": result.selected_was_llm_top,
        "num_candidates_legal": result.num_candidates_legal,
        "candidates": candidates,
    }


def history_string(uci_moves: list[str]) -> str:
    tail = list(uci_moves)[-HISTORY_PLIES:]
    return " ".join(tail) if tail else NO_HISTORY


def looks_like_fen(text: str) -> bool:
    s = (text or "").strip()
    if not s or "\n" in s:
        return False
    parts = s.split()
    return 2 <= len(parts) <= 6 and parts[0].count("/") == 7


def strip_pgn_noise(text: str) -> str:
    text = re.sub(r"^\s*\[[^\]\n]*\]\s*$", " ", text, flags=re.MULTILINE)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\{[^{}]*\}", " ", text)
        text = re.sub(r"\([^()]*\)", " ", text)
    return re.sub(r"\$\d+", " ", text)


def parse_moves_text(text: str, start_fen: str | None = None) -> tuple[chess.Board, list[str], list[str]]:
    try:
        board = chess.Board(start_fen) if start_fen else chess.Board()
    except ValueError as e:
        raise ValueError(f"Άκυρο αρχικό FEN: {e}") from e

    uci_moves, san_moves = [], []
    for raw in strip_pgn_noise(text or "").split():
        if raw in RESULT_TOKENS:
            continue
        token = MOVE_NUMBER_PREFIX.sub("", raw)
        if not token or BARE_MOVE_NUMBER.match(token):
            continue
        token = token.rstrip("!?")
        if not token:
            continue

        move = None
        if UCI_TOKEN.match(token):
            try:
                candidate = chess.Move.from_uci(token.lower())
            except ValueError:
                candidate = None
            if candidate in board.legal_moves:
                move = candidate
        if move is None:
            try:
                move = board.parse_san(token)
            except ValueError:
                raise ValueError(f"Η κίνηση #{len(uci_moves) + 1} «{raw}» δεν είναι νόμιμη/αναγνωρίσιμη "
                                 f"στη θέση {board.fen()}") from None

        san_moves.append(board.san(move))
        uci_moves.append(move.uci())
        board.push(move)
    return board, uci_moves, san_moves


def game_over_reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_seventyfive_moves():
        return "75-move rule"
    if board.is_fivefold_repetition():
        return "fivefold repetition"
    return "game over"


def create_app(predictor=None, meta: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="/static")
    app.config["PREDICTOR"] = predictor
    app.config["SERVER_META"] = meta or {}

    def run_prediction(board: chess.Board, history: str):
        if not list(board.legal_moves):
            return None, (jsonify({"error": f"no legal moves ({game_over_reason(board)})"}), 400)
        side = "White" if board.turn == chess.WHITE else "Black"
        started = time.monotonic()
        try:
            result = app.config["PREDICTOR"].predict_reranked(board.fen(), history, side)
        except Exception as e:
            logger.exception("Prediction failed")
            return None, (jsonify({"error": str(e)}), 500)
        payload = serialize_rerank_result(result, board)
        payload["inference_ms"] = int((time.monotonic() - started) * 1000)
        return payload, None

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "predictor_loaded": app.config["PREDICTOR"] is not None,
                        **app.config["SERVER_META"]})

    @app.route("/api/predict", methods=["POST"])
    def predict():
        if app.config["PREDICTOR"] is None:
            return jsonify({"error": "predictor not loaded"}), 503
        data = request.get_json(silent=True) or {}
        fen = data.get("fen")
        if not fen:
            return jsonify({"error": "fen required"}), 400
        try:
            board = chess.Board(fen)
        except ValueError as e:
            return jsonify({"error": f"invalid FEN: {e}"}), 400

        payload, error = run_prediction(board, data.get("history", NO_HISTORY))
        return error or jsonify(payload)

    @app.route("/api/advise", methods=["POST"])
    def advise():
        if app.config["PREDICTOR"] is None:
            return jsonify({"error": "predictor not loaded"}), 503
        data = request.get_json(silent=True) or {}
        moves_text = (data.get("moves") or "").strip()
        start_fen = (data.get("start_fen") or "").strip() or None
        if not moves_text and not start_fen:
            return jsonify({"error": "moves required (κινήσεις της παρτίδας, SAN ή UCI)"}), 400

        warning = None
        try:
            if moves_text and start_fen is None and looks_like_fen(moves_text):
                board = chess.Board(moves_text)
                uci_moves, san_moves = [], []
                warning = ("Δόθηκε μόνο FEN χωρίς ιστορικό κινήσεων. Το μοντέλο εκπαιδεύτηκε "
                           "πάντα με τις προηγούμενες κινήσεις στο prompt — η πρόταση μπορεί να "
                           "είναι λιγότερο αξιόπιστη. Προτίμησε να επικολλήσεις τις κινήσεις.")
            else:
                board, uci_moves, san_moves = parse_moves_text(moves_text, start_fen)
                if not uci_moves:
                    warning = ("Δεν δόθηκαν κινήσεις — η πρόταση αφορά την αρχική θέση χωρίς "
                               "ιστορικό (εκτός κατανομής για το μοντέλο).")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        if board.is_game_over():
            return jsonify({"error": f"Η παρτίδα έχει τελειώσει ({game_over_reason(board)}) — "
                                     "καμία κίνηση."}), 400

        history = history_string(uci_moves)
        payload, error = run_prediction(board, history)
        if error:
            return error
        payload.update({
            "fen": board.fen(),
            "side_to_move": "White" if board.turn == chess.WHITE else "Black",
            "ply": len(uci_moves),
            "san_moves": san_moves,
            "uci_moves": uci_moves,
            "history_used": history,
            "history_provided": bool(uci_moves),
            "warning": warning,
            "start_fen": start_fen,
        })
        return jsonify(payload)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Web app: play against the model or ask it for a move.")
    parser.add_argument("--adapter", default=BEST_HUB_REPO, help="Hugging Face repo or local folder")
    parser.add_argument("--stockfish", default="/usr/games/stockfish")
    parser.add_argument("--stockfish-depth", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--rerank-candidates", choices=list(CANDIDATE_SOURCES), default=DEFAULT_CANDIDATE_SOURCE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    import torch
    from model.inference.stockfish_rerank import StockfishRerankPredictor

    print(f"Loading the model with {args.adapter}...", file=sys.stderr)
    predictor = StockfishRerankPredictor(adapter_path=args.adapter, stockfish_path=args.stockfish,
                                         stockfish_depth=args.stockfish_depth,
                                         num_candidates=args.num_candidates,
                                         candidate_source=args.rerank_candidates)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Running on CPU, every move can take minutes. Use a GPU (e.g. Colab).", file=sys.stderr)
    meta = {
        "device": device,
        "adapter": args.adapter,
        "num_candidates": args.num_candidates,
        "stockfish_depth": args.stockfish_depth,
        "candidate_source": args.rerank_candidates,
    }

    print(f"Open http://{args.host}:{args.port} in your browser", file=sys.stderr)
    try:
        create_app(predictor=predictor, meta=meta).run(host=args.host, port=args.port,
                                                       debug=False, use_reloader=False)
    finally:
        predictor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
