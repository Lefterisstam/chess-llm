from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chess
import chess.engine

from model.inference.exact_scoring import ExactScoringChessMovePredictor, rank_legal_moves
from model.inference.legal_move_filter import extract_uci_candidate
from model.inference.predict import format_prompt
from model.inference.prefix_constrained import build_legal_move_token_sets, make_prefix_allowed_tokens_fn

MATE_SCORE = 10000
DEFAULT_NUM_CANDIDATES = 10
DEFAULT_STOCKFISH_DEPTH = 6
DEFAULT_MAX_NEW_TOKENS = 10
DEFAULT_STOCKFISH_PATH = "/usr/games/stockfish"
CANDIDATE_SOURCES = ("beam", "exact")
DEFAULT_CANDIDATE_SOURCE = "exact"

logger = logging.getLogger(__name__)


def validate_candidate_source(source: str) -> str:
    if source not in CANDIDATE_SOURCES:
        raise ValueError(f"Unknown candidate_source {source!r}, choose one of {CANDIDATE_SOURCES}")
    return source


def stockfish_version(stockfish_path: str | None) -> str:
    if not stockfish_path:
        return ""
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            return str(engine.id.get("name", ""))
    except Exception as e:
        logger.warning("Could not read the Stockfish version from %s: %s", stockfish_path, e)
        return ""


@dataclass
class StockfishRerankResult:
    move: chess.Move
    stockfish_eval_cp: int
    num_candidates_legal: int
    llm_top_pick: chess.Move
    selected_was_llm_top: bool
    all_candidates: list[chess.Move] = field(default_factory=list)
    all_scores_cp: list[int] = field(default_factory=list)


class StockfishRerankPredictor(ExactScoringChessMovePredictor):
    candidate_source = DEFAULT_CANDIDATE_SOURCE
    _engine = None

    def __init__(self, adapter_path: str, stockfish_path: str = DEFAULT_STOCKFISH_PATH,
                 stockfish_depth: int = DEFAULT_STOCKFISH_DEPTH,
                 num_candidates: int = DEFAULT_NUM_CANDIDATES,
                 candidate_source: str = DEFAULT_CANDIDATE_SOURCE, **kwargs):
        super().__init__(adapter_path, **kwargs)
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.num_candidates = num_candidates
        self.candidate_source = validate_candidate_source(candidate_source)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                logger.warning("Stockfish did not quit cleanly")
            self._engine = None

    def _get_engine(self):
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        return self._engine

    def predict_reranked(self, fen: str, history: str, side: str,
                         max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> StockfishRerankResult:
        board = chess.Board(fen)
        if not list(board.legal_moves):
            raise ValueError("Position has 0 legal moves (checkmate or stalemate).")

        candidates = self._generate_candidates(board, fen, history, side, max_new_tokens)
        if not candidates:
            logger.warning("The model returned no candidates, using the first legal move")
            candidates = [min(board.legal_moves, key=lambda m: m.uci())]

        engine = self._get_engine()
        scores = [self._score_move(board, move, engine) for move in candidates]
        best = scores.index(max(scores))
        return StockfishRerankResult(
            move=candidates[best],
            stockfish_eval_cp=scores[best],
            num_candidates_legal=len(candidates),
            llm_top_pick=candidates[0],
            selected_was_llm_top=candidates[best] == candidates[0],
            all_candidates=candidates,
            all_scores_cp=scores,
        )

    def _generate_candidates(self, board: chess.Board, fen: str, history: str, side: str,
                             max_new_tokens: int) -> list[chess.Move]:
        prompt = format_prompt(fen, history, side)
        if self.candidate_source == "exact":
            ranked = rank_legal_moves(self.model, self.tokenizer, prompt, board,
                                      include_eos=self.include_eos, batch_size=self.batch_size)
            return [s.move for s in ranked[:self.num_candidates]]
        return self._beam_candidates(board, prompt, max_new_tokens)

    def _beam_candidates(self, board: chess.Board, prompt: str, max_new_tokens: int) -> list[chess.Move]:
        import torch

        legal_token_seqs = build_legal_move_token_sets(self.tokenizer, board)
        k = min(self.num_candidates, len(legal_token_seqs))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]
        prefix_fn = make_prefix_allowed_tokens_fn(legal_token_seqs, prompt_length,
                                                  self.tokenizer.eos_token_id)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=k,
                num_return_sequences=k,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_fn,
            )

        candidates = []
        for output in outputs:
            uci = extract_uci_candidate(self.tokenizer.decode(output[prompt_length:], skip_special_tokens=True))
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if move in board.legal_moves and move not in candidates:
                candidates.append(move)
        return candidates

    def _score_move(self, board: chess.Board, move: chess.Move, engine) -> int:
        after = board.copy(stack=False)
        after.push(move)
        if after.is_checkmate():
            return MATE_SCORE
        if after.is_stalemate() or after.is_insufficient_material():
            return 0

        info = engine.analyse(after, chess.engine.Limit(depth=self.stockfish_depth))
        opponent_score = info["score"].pov(after.turn)
        opponent_cp = opponent_score.score(mate_score=MATE_SCORE)
        if opponent_cp is None:
            opponent_cp = MATE_SCORE if opponent_score.is_mate() else 0
        return -opponent_cp
