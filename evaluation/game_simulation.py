from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import chess.engine
import chess.pgn

from model.inference.factory import build_predictor
from model.inference.legal_move_filter import extract_uci_candidate, filter_to_legal
from model.inference.predict import format_prompt
from model.inference.stockfish_rerank import CANDIDATE_SOURCES, DEFAULT_CANDIDATE_SOURCE, stockfish_version

DEFAULT_MAX_PLIES = 300
HISTORY_PLIES = 10

logger = logging.getLogger(__name__)


def side_name(board: chess.Board) -> str:
    return "White" if board.turn == chess.WHITE else "Black"


def history_string(history_uci: list[str]) -> str:
    tail = history_uci[-HISTORY_PLIES:]
    return " ".join(tail) if tail else "(none — opening)"


class Agent:
    name = "agent"

    def select_move(self, board: chess.Board, history_uci: list[str]) -> chess.Move:
        raise NotImplementedError

    def close(self) -> None:
        pass


class RandomAgent(Agent):
    def __init__(self, seed: int = 42):
        self.name = "random"
        self.rng = random.Random(seed)

    def select_move(self, board: chess.Board, history_uci: list[str]) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("No legal moves")
        return self.rng.choice(moves)


class StockfishAgent(Agent):
    def __init__(self, stockfish_path: str, depth: int):
        self.name = f"stockfish-d{depth}"
        self.stockfish_path = stockfish_path
        self.depth = depth
        self._engine = None

    def select_move(self, board: chess.Board, history_uci: list[str]) -> chess.Move:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        return self._engine.play(board, chess.engine.Limit(depth=self.depth)).move

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                logger.warning("Stockfish did not quit cleanly")
            self._engine = None


class LLMAgent(Agent):
    VALID_MODES = ("greedy", "prefix-constrained", "exact", "stockfish-rerank")

    def __init__(self, predictor, mode: str):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown LLM mode {mode!r}, choose one of {self.VALID_MODES}")
        self.name = f"llm-{mode}"
        self.predictor = predictor
        self.mode = mode

    def select_move(self, board: chess.Board, history_uci: list[str]) -> chess.Move:
        fen, history, side = board.fen(), history_string(history_uci), side_name(board)
        if self.mode == "stockfish-rerank":
            return self.predictor.predict_reranked(fen, history, side).move
        if self.mode == "prefix-constrained":
            return self.predictor.predict_prefix_constrained(fen, history, side).filtered
        if self.mode == "exact":
            return self.predictor.predict_exact(fen, history, side).filtered
        return filter_to_legal(self.predictor.predict_raw(fen, history, side), board).filtered

    def close(self) -> None:
        if hasattr(self.predictor, "close"):
            self.predictor.close()


class ZeroShotAgent(Agent):
    def __init__(self, model_id: str):
        self.name = f"zero-shot-{model_id.split('/')[-1]}"
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    def _load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        kwargs = {"device_map": "auto"}
        if "gpt2" not in self.model_id.lower():
            bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def select_move(self, board: chess.Board, history_uci: list[str]) -> chess.Move:
        import torch

        if self._model is None:
            self._load()
        prompt = format_prompt(board.fen(), history_string(history_uci), side_name(board))
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=1024).to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                          pad_token_id=self._tokenizer.eos_token_id)
        text = self._tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return filter_to_legal(extract_uci_candidate(text), board).filtered


@dataclass
class GameResult:
    white_name: str
    black_name: str
    result: str
    termination: str
    n_plies: int
    final_fen: str
    moves_uci: list[str] = field(default_factory=list)


def termination_reason(board: chess.Board, capped: bool) -> str:
    if capped:
        return "max_plies"
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_seventyfive_moves():
        return "seventyfive_moves"
    if board.is_fivefold_repetition():
        return "fivefold_repetition"
    return "other"


def play_game(white: Agent, black: Agent, max_plies: int = DEFAULT_MAX_PLIES) -> GameResult:
    board = chess.Board()
    history = []
    while not board.is_game_over() and len(history) < max_plies:
        agent = white if board.turn == chess.WHITE else black
        move = agent.select_move(board, history)
        if move not in board.legal_moves:
            raise ValueError(f"{agent.name} played the illegal move {move.uci()} in {board.fen()}")
        board.push(move)
        history.append(move.uci())

    capped = not board.is_game_over()
    return GameResult(
        white_name=white.name,
        black_name=black.name,
        result="1/2-1/2" if capped else board.result(),
        termination=termination_reason(board, capped),
        n_plies=len(history),
        final_fen=board.fen(),
        moves_uci=history,
    )


@dataclass
class MatchStats:
    agent_a: str = ""
    agent_b: str = ""
    n_games: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    wins_a_as_white: int = 0
    wins_a_as_black: int = 0
    max_plies: int = DEFAULT_MAX_PLIES
    duration_seconds: float = 0.0
    config: dict = field(default_factory=dict)
    games: list[dict] = field(default_factory=list)

    @property
    def win_rate_a(self) -> float:
        return self.wins_a / self.n_games if self.n_games else 0.0

    @property
    def score_a(self) -> float:
        return (self.wins_a + 0.5 * self.draws) / self.n_games if self.n_games else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["win_rate_a"] = round(self.win_rate_a, 4)
        d["score_a"] = round(self.score_a, 4)
        return d

    def log_summary(self) -> None:
        logger.info("%s vs %s, %d games: %d wins (%d as White, %d as Black), %d draws, %d losses",
                    self.agent_a, self.agent_b, self.n_games, self.wins_a, self.wins_a_as_white,
                    self.wins_a_as_black, self.draws, self.wins_b)
        logger.info("Win rate %.1f%%, score %.1f%%, took %.1f s", 100 * self.win_rate_a,
                    100 * self.score_a, self.duration_seconds)


def play_match(agent_a: Agent, agent_b: Agent, n_games: int, max_plies: int = DEFAULT_MAX_PLIES,
               alternate_colors: bool = True) -> MatchStats:
    stats = MatchStats(agent_a=agent_a.name, agent_b=agent_b.name, max_plies=max_plies)
    started = time.monotonic()

    for i in range(n_games):
        a_is_white = i % 2 == 0 if alternate_colors else True
        white, black = (agent_a, agent_b) if a_is_white else (agent_b, agent_a)
        game = play_game(white, black, max_plies=max_plies)
        stats.n_games += 1
        stats.games.append(asdict(game))

        if game.result == "1/2-1/2":
            stats.draws += 1
        elif (game.result == "1-0") == a_is_white:
            stats.wins_a += 1
            if a_is_white:
                stats.wins_a_as_white += 1
            else:
                stats.wins_a_as_black += 1
        else:
            stats.wins_b += 1
        logger.info("Game %d/%d: %s with %s as White (%s, %d plies)", i + 1, n_games, game.result,
                    white.name, game.termination, game.n_plies)

    stats.duration_seconds = time.monotonic() - started
    return stats


def parse_agent_spec(spec: str, args: argparse.Namespace, shared: dict) -> Agent:
    if spec == "random":
        return RandomAgent(seed=args.seed)

    if spec.startswith("stockfish:"):
        depth = int(spec.split(":", 1)[1])
        if depth > 10:
            raise ValueError("Stockfish depth > 10 is not allowed as a baseline")
        return StockfishAgent(args.stockfish, depth)

    if spec.startswith("llm:"):
        mode = spec.split(":", 1)[1]
        if not args.adapter:
            raise ValueError("llm:* agents need --adapter")
        if mode not in LLMAgent.VALID_MODES:
            raise ValueError(f"Unknown LLM mode {mode!r}, choose one of {LLMAgent.VALID_MODES}")
        if mode not in shared:
            shared[mode] = build_predictor(args.adapter, mode, stockfish_path=args.stockfish,
                                           rerank_depth=args.rerank_depth, rerank_k=args.rerank_k,
                                           rerank_candidates=args.rerank_candidates)
        return LLMAgent(shared[mode], mode)

    if spec.startswith("zero-shot:"):
        return ZeroShotAgent(spec.split(":", 1)[1])

    raise ValueError(f"Unknown agent spec {spec!r}")


def export_pgn(stats: MatchStats, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, g in enumerate(stats.games):
            game = chess.pgn.Game()
            game.headers["Event"] = f"{stats.agent_a} vs {stats.agent_b}"
            game.headers["Round"] = str(i + 1)
            game.headers["White"] = g["white_name"]
            game.headers["Black"] = g["black_name"]
            game.headers["Result"] = g["result"]
            game.headers["Termination"] = g["termination"]
            node = game
            for uci in g["moves_uci"]:
                node = node.add_variation(chess.Move.from_uci(uci))
            print(game, file=f, end="\n\n")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Play full games between two agents.")
    p.add_argument("--agent-a", required=True,
                   help="random | stockfish:<depth> | llm:<mode> | zero-shot:<model id>")
    p.add_argument("--agent-b", required=True)
    p.add_argument("--games", type=int, default=20, help="colours alternate every game")
    p.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES,
                   help="games that reach this length count as draws")
    p.add_argument("--adapter", default=None, help="needed for llm:* agents")
    p.add_argument("--stockfish", default="/usr/games/stockfish")
    p.add_argument("--rerank-depth", type=int, default=6)
    p.add_argument("--rerank-k", type=int, default=10)
    p.add_argument("--rerank-candidates", choices=list(CANDIDATE_SOURCES), default=DEFAULT_CANDIDATE_SOURCE)
    p.add_argument("--seed", type=int, default=42, help="seed of the random agent")
    p.add_argument("--output", required=True, type=Path, help="results JSON, e.g. game_sim_<name>.json")
    p.add_argument("--save-pgn", type=Path, default=None)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    shared = {}
    agent_a = parse_agent_spec(args.agent_a, args, shared)
    agent_b = parse_agent_spec(args.agent_b, args, shared)
    try:
        stats = play_match(agent_a, agent_b, n_games=args.games, max_plies=args.max_plies)
    finally:
        agent_a.close()
        agent_b.close()

    stats.config = {
        "agent_a_spec": args.agent_a,
        "agent_b_spec": args.agent_b,
        "adapter": args.adapter,
        "stockfish": args.stockfish,
        "stockfish_version": stockfish_version(args.stockfish),
        "rerank_depth": args.rerank_depth,
        "rerank_k": args.rerank_k,
        "rerank_candidates": args.rerank_candidates,
        "seed": args.seed,
        "max_plies": args.max_plies,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    stats.log_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    if args.save_pgn:
        export_pgn(stats, args.save_pgn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
