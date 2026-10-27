from __future__ import annotations

import importlib
import importlib.util

import chess
import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestStockfishVersion:
    def test_missing_binary_returns_empty(self, tmp_path):
        from model.inference.stockfish_rerank import stockfish_version
        assert stockfish_version(str(tmp_path / "missing-binary")) == ""

    def test_empty_path_returns_empty(self):
        from model.inference.stockfish_rerank import stockfish_version
        assert stockfish_version("") == ""
        assert stockfish_version(None) == ""


class _FakeInfo(dict):
    pass


class _FakeScore:
    def __init__(self, opponent_eval_cp: int):
        self._opponent_eval = opponent_eval_cp

    def pov(self, color):
        class _PovWrapper:
            def __init__(self, val):
                self._val = val
            def score(self, mate_score=10000):
                return self._val
            def is_mate(self):
                return False
        return _PovWrapper(self._opponent_eval)


class FakeEngine:
    def __init__(self, opp_evals: dict[str, int]):
        self.opp_evals = opp_evals
        self.last_move_played = None
        self._closed = False

    def analyse(self, board, limit):
        last_uci = board.move_stack[-1].uci() if board.move_stack else ""
        opp_eval = self.opp_evals.get(last_uci, 0)
        return _FakeInfo(score=_FakeScore(opp_eval))

    def quit(self):
        self._closed = True


def _bare_rerank_predictor():
    from model.inference.stockfish_rerank import StockfishRerankPredictor
    cls = StockfishRerankPredictor
    return cls.__new__(cls)


class TestStockfishRerankResult:
    def test_minimal_construction(self):
        from model.inference.stockfish_rerank import StockfishRerankResult
        m = chess.Move.from_uci("e2e4")
        result = StockfishRerankResult(
            move=m,
            stockfish_eval_cp=30,
            num_candidates_legal=5,
            llm_top_pick=m,
            selected_was_llm_top=True,
        )
        assert result.move.uci() == "e2e4"
        assert result.stockfish_eval_cp == 30
        assert result.all_candidates == []
        assert result.all_scores_cp == []

    def test_full_construction(self):
        from model.inference.stockfish_rerank import StockfishRerankResult
        cands = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
        scores = [50, 30]
        result = StockfishRerankResult(
            move=cands[0],
            stockfish_eval_cp=50,
            num_candidates_legal=2,
            llm_top_pick=cands[0],
            selected_was_llm_top=True,
            all_candidates=cands,
            all_scores_cp=scores,
        )
        assert len(result.all_candidates) == 2
        assert result.all_scores_cp == [50, 30]


def _make_predictor_for_score_tests():
    if not _HAS_TORCH:
        pytest.skip("needs torch")
    mod = importlib.import_module("model.inference.stockfish_rerank")
    p = mod.StockfishRerankPredictor.__new__(mod.StockfishRerankPredictor)
    p.stockfish_depth = 6
    return p


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
class TestScoreMoveEdgeCases:
    def test_checkmate_by_us_returns_mate_score(self):
        from model.inference.stockfish_rerank import MATE_SCORE
        p = _make_predictor_for_score_tests()

        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2")
        move = chess.Move.from_uci("d8h4")
        engine = FakeEngine(opp_evals={})
        score = p._score_move(board, move, engine)
        assert score == MATE_SCORE

    def test_stalemate_returns_zero(self):
        p = _make_predictor_for_score_tests()
        board = chess.Board("8/8/8/4k3/8/8/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("e1e2")
        engine = FakeEngine(opp_evals={"e1e2": 100})
        score = p._score_move(board, move, engine)
        assert score == 0


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
class TestScoreMoveNormalPosition:
    def test_normal_move_with_positive_player_eval(self):
        p = _make_predictor_for_score_tests()
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        engine = FakeEngine(opp_evals={"e2e4": -30})
        score = p._score_move(board, move, engine)
        assert score == 30

    def test_normal_move_with_negative_player_eval(self):
        p = _make_predictor_for_score_tests()
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        engine = FakeEngine(opp_evals={"e2e4": 100})
        score = p._score_move(board, move, engine)
        assert score == -100


class TestSelectionLogic:
    @pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
    def test_picks_highest_score(self):
        cands = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "g1f3")]
        scores = [10, 50, 30]

        p = _bare_rerank_predictor()
        p._generate_candidates = lambda *a, **kw: cands
        p._score_move = lambda board, mv, engine: scores[cands.index(mv)]
        p._get_engine = lambda: object()
        p.num_candidates = 10
        p.stockfish_depth = 6

        result = p.predict_reranked(chess.STARTING_FEN, "", "White")
        assert result.move.uci() == "d2d4"
        assert result.stockfish_eval_cp == 50
        assert result.llm_top_pick.uci() == "e2e4"
        assert result.selected_was_llm_top is False

    @pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
    def test_tie_break_prefers_llm_top(self):
        cands = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "g1f3")]
        scores = [50, 50, 30]

        p = _bare_rerank_predictor()
        p._generate_candidates = lambda *a, **kw: cands
        p._score_move = lambda board, mv, engine: scores[cands.index(mv)]
        p._get_engine = lambda: object()
        p.num_candidates = 10
        p.stockfish_depth = 6

        result = p.predict_reranked(chess.STARTING_FEN, "", "White")
        assert result.move.uci() == "e2e4"
        assert result.selected_was_llm_top is True

    @pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
    def test_kept_llm_top_when_it_scores_highest(self):
        cands = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "g1f3")]
        scores = [100, 50, 30]

        p = _bare_rerank_predictor()
        p._generate_candidates = lambda *a, **kw: cands
        p._score_move = lambda board, mv, engine: scores[cands.index(mv)]
        p._get_engine = lambda: object()
        p.num_candidates = 10
        p.stockfish_depth = 6

        result = p.predict_reranked(chess.STARTING_FEN, "", "White")
        assert result.move.uci() == "e2e4"
        assert result.selected_was_llm_top is True


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
class TestNoLegalMoves:
    def test_raises_on_checkmate_position(self):
        p = _bare_rerank_predictor()
        mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        with pytest.raises(ValueError, match="0 legal moves"):
            p.predict_reranked(mate_fen, "", "White")


@pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")
class TestLifecycle:
    def test_close_when_engine_not_open(self):
        p = _bare_rerank_predictor()
        p._engine = None
        p.close()
        assert p._engine is None

    def test_close_quits_engine(self):
        p = _bare_rerank_predictor()
        fake = FakeEngine(opp_evals={})
        p._engine = fake
        p.close()
        assert fake._closed is True
        assert p._engine is None

    def test_close_is_idempotent(self):
        p = _bare_rerank_predictor()
        fake = FakeEngine(opp_evals={})
        p._engine = fake
        p.close()
        p.close()
        assert p._engine is None

    def test_context_manager_calls_close(self):
        p = _bare_rerank_predictor()
        fake = FakeEngine(opp_evals={})
        p._engine = fake
        with p as ctx:
            assert ctx is p
        assert fake._closed is True
