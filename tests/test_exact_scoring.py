from __future__ import annotations

import importlib.util
import math
import types

import chess
import pytest

from model.inference.exact_scoring import (
    DEFAULT_BATCH_SIZE,
    ScoredMove,
    build_legal_move_token_map,
    legal_move_distribution,
    sort_scored_moves,
)
from model.inference.predict import format_prompt
from model.inference.stockfish_rerank import DEFAULT_CANDIDATE_SOURCE

_HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="needs torch")

VOCAB = 128
BOOST = 2.0
FOOLS_MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
OPENING_HISTORY = "(none — opening)"


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class FakeCharTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return _Enc([ord(c) for c in text])


def _lp_pref() -> float:
    return BOOST - math.log(math.exp(BOOST) + VOCAB - 1)


def _lp_non() -> float:
    return -math.log(math.exp(BOOST) + VOCAB - 1)


def _lp_unif() -> float:
    return -math.log(VOCAB)


class FakeBigramModel:
    device = "cpu"

    def __init__(self, prefs: dict[int, int]):
        self.prefs = prefs
        self.calls: list[tuple[int, int]] = []

    def __call__(self, input_ids, attention_mask=None, use_cache=False):
        import torch
        self.calls.append(tuple(input_ids.shape))
        n_rows, n_cols = input_ids.shape
        logits = torch.zeros(n_rows, n_cols, VOCAB)
        for b in range(n_rows):
            for t in range(n_cols):
                nxt = self.prefs.get(int(input_ids[b, t]))
                if nxt is not None:
                    logits[b, t, nxt] = BOOST
        return types.SimpleNamespace(logits=logits)


def _bigram_prefs(chain: str) -> dict[int, int]:
    return {ord(a): ord(b) for a, b in zip(chain, chain[1:])}


def _sm(uci: str, lp: float) -> ScoredMove:
    return ScoredMove(move=chess.Move.from_uci(uci), logprob=lp, n_tokens=4)


class TestBuildLegalMoveTokenMap:
    def test_starting_position(self):
        board = chess.Board()
        token_map = build_legal_move_token_map(FakeCharTokenizer(), board)
        assert len(token_map) == 20
        assert [mv for mv, _ in token_map] == list(board.legal_moves)
        toks = {mv.uci(): t for mv, t in token_map}
        assert toks["e2e4"] == [ord(c) for c in "e2e4"]

    def test_promotion_has_five_tokens(self):
        board = chess.Board("7K/4P3/8/8/8/8/8/k7 w - - 0 1")
        toks = {mv.uci(): t for mv, t in build_legal_move_token_map(FakeCharTokenizer(), board)}
        assert toks["e7e8q"] == [ord(c) for c in "e7e8q"]

    def test_empty_on_checkmate(self):
        assert build_legal_move_token_map(FakeCharTokenizer(), chess.Board(FOOLS_MATE_FEN)) == []


class TestSortScoredMoves:
    def test_sort_desc_with_ranks(self):
        out = sort_scored_moves([_sm("e2e4", -2.0), _sm("d2d4", -1.0), _sm("g1f3", -3.0)])
        assert [x.move.uci() for x in out] == ["d2d4", "e2e4", "g1f3"]
        assert [x.rank for x in out] == [1, 2, 3]

    def test_tie_break_is_alphabetical(self):
        out = sort_scored_moves([_sm("g1f3", -1.0), _sm("a2a3", -1.0)])
        assert [x.move.uci() for x in out] == ["a2a3", "g1f3"]

    def test_empty(self):
        assert sort_scored_moves([]) == []


class TestLegalMoveDistribution:
    def test_sums_to_one_and_monotone(self):
        dist = legal_move_distribution([_sm("e2e4", -1.0), _sm("d2d4", -2.0), _sm("g1f3", -5.0)])
        probs = [p for _, p in dist]
        assert abs(sum(probs) - 1.0) < 1e-9
        assert probs[0] > probs[1] > probs[2]
        z = math.exp(-1.0) + math.exp(-2.0) + math.exp(-5.0)
        assert abs(probs[0] - math.exp(-1.0) / z) < 1e-9

    def test_preserves_input_order(self):
        dist = legal_move_distribution([_sm("g1f3", -5.0), _sm("e2e4", -1.0)])
        assert [m.uci() for m, _ in dist] == ["g1f3", "e2e4"]

    def test_empty(self):
        assert legal_move_distribution([]) == []

    def test_numerically_stable_for_very_negative_logprobs(self):
        dist = legal_move_distribution([_sm("e2e4", -1000.0), _sm("d2d4", -1001.0)])
        assert abs(sum(p for _, p in dist) - 1.0) < 1e-9


@needs_torch
class TestScoreContinuations:
    def test_exact_values_match_hand_computation(self):
        from model.inference.exact_scoring import score_continuations
        model = FakeBigramModel(_bigram_prefs("Xab"))
        prompt = [ord("X")]
        conts = [[ord("a"), ord("b")], [ord("a"), ord("c")], [ord("c"), ord("d")]]
        got = score_continuations(model, prompt, conts, pad_token_id=0)
        expected = [2 * _lp_pref(), _lp_pref() + _lp_non(), _lp_non() + _lp_unif()]
        for g, e in zip(got, expected):
            assert abs(g - e) < 1e-5
        assert got[0] > got[1] > got[2]

    def test_padding_invariance(self):
        from model.inference.exact_scoring import score_continuations
        model = FakeBigramModel(_bigram_prefs("Xabcd"))
        prompt = [ord("X")]
        short = [ord("a"), ord("b")]
        long_ = [ord("a"), ord("b"), ord("c"), ord("d")]
        alone = score_continuations(model, prompt, [short], 0)[0]
        batched = score_continuations(model, prompt, [long_, short], 0)[1]
        assert abs(alone - batched) < 1e-6
        assert abs(alone - 2 * _lp_pref()) < 1e-5
        assert abs(score_continuations(model, prompt, [long_], 0)[0] - 4 * _lp_pref()) < 1e-5

    def test_batch_size_invariance_and_chunking(self):
        from model.inference.exact_scoring import score_continuations
        model = FakeBigramModel(_bigram_prefs("Xab"))
        prompt = [ord("X")]
        conts = [[ord("a"), ord(c)] for c in "bcdefgh"]
        one_by_one = score_continuations(model, prompt, conts, 0, batch_size=1)
        assert len(model.calls) == 7
        model.calls.clear()
        chunked = score_continuations(model, prompt, conts, 0, batch_size=16)
        assert model.calls == [(7, 3)]
        for a, b in zip(one_by_one, chunked):
            assert abs(a - b) < 1e-6

    def test_no_length_normalization(self):
        from model.inference.exact_scoring import score_continuations
        model = FakeBigramModel(_bigram_prefs("Xabc"))
        prompt = [ord("X")]
        two, three = score_continuations(model, prompt, [[ord("a"), ord("b")], [ord("a"), ord("b"), ord("c")]], 0)
        assert three < two

    def test_empty_input(self):
        from model.inference.exact_scoring import score_continuations
        assert score_continuations(FakeBigramModel({}), [1], [], 0) == []


@needs_torch
class TestRankLegalMoves:
    def _prompt(self) -> str:
        return format_prompt(chess.STARTING_FEN, OPENING_HISTORY, "White")

    def test_prefers_bigram_path_g1f3(self):
        from model.inference.exact_scoring import rank_legal_moves
        model = FakeBigramModel(_bigram_prefs("\ng1f3"))
        ranked = rank_legal_moves(model, FakeCharTokenizer(), self._prompt(), chess.Board())
        assert len(ranked) == 20
        assert ranked[0].move.uci() == "g1f3"
        assert ranked[1].move.uci() == "g1h3"
        assert [s.rank for s in ranked] == list(range(1, 21))
        assert all(s.n_tokens == 4 for s in ranked)
        assert abs(ranked[0].logprob - 4 * _lp_pref()) < 1e-5

    def test_all_moves_legal_and_distinct(self):
        from model.inference.exact_scoring import rank_legal_moves
        board = chess.Board()
        ranked = rank_legal_moves(FakeBigramModel({}), FakeCharTokenizer(), self._prompt(), board)
        assert len({s.move.uci() for s in ranked}) == 20
        assert all(s.move in board.legal_moves for s in ranked)

    def test_include_eos_adds_one_token(self):
        from model.inference.exact_scoring import rank_legal_moves
        ranked = rank_legal_moves(
            FakeBigramModel({}), FakeCharTokenizer(), self._prompt(), chess.Board(), include_eos=True,
        )
        assert all(s.n_tokens == 5 for s in ranked)

    def test_raises_on_no_legal_moves(self):
        from model.inference.exact_scoring import rank_legal_moves
        with pytest.raises(ValueError, match="0 legal moves"):
            rank_legal_moves(FakeBigramModel({}), FakeCharTokenizer(), "x", chess.Board(FOOLS_MATE_FEN))


@needs_torch
class TestExactScoringPredictor:
    def _bare(self, prefs):
        from model.inference.exact_scoring import ExactScoringChessMovePredictor as Cls
        p = Cls.__new__(Cls)
        p.model = FakeBigramModel(prefs)
        p.tokenizer = FakeCharTokenizer()
        p.include_eos = False
        p.batch_size = DEFAULT_BATCH_SIZE
        return p

    def test_predict_exact_returns_legal_top_move(self):
        r = self._bare(_bigram_prefs("\ng1f3")).predict_exact(chess.STARTING_FEN, OPENING_HISTORY, "White")
        assert r.filtered.uci() == "g1f3"
        assert r.raw == "g1f3"
        assert r.was_legal and r.was_parseable
        assert r.fallback_strategy == "exact_scoring"

    def test_rank_legal_moves_full_ranking(self):
        ranked = self._bare(_bigram_prefs("\ng1f3")).rank_legal_moves(chess.STARTING_FEN, OPENING_HISTORY, "White")
        assert len(ranked) == 20
        assert ranked[0].rank == 1 and ranked[0].move.uci() == "g1f3"


@needs_torch
class TestRerankCandidateSourceExact:
    def _bare(self, prefs, k):
        from model.inference.stockfish_rerank import StockfishRerankPredictor as Cls
        p = Cls.__new__(Cls)
        p.model = FakeBigramModel(prefs)
        p.tokenizer = FakeCharTokenizer()
        p.candidate_source = "exact"
        p.num_candidates = k
        return p

    def test_exact_candidates_top_k_in_llm_order(self):
        p = self._bare(_bigram_prefs("\ng1f3"), k=5)
        board = chess.Board()
        cands = p._generate_candidates(board, chess.STARTING_FEN, OPENING_HISTORY, "White", 10)
        assert len(cands) == 5
        assert cands[0].uci() == "g1f3"
        assert len({c.uci() for c in cands}) == 5
        assert all(c in board.legal_moves for c in cands)

    def test_exact_candidates_capped_by_legal_moves(self):
        p = self._bare({}, k=10)
        board = chess.Board("7k/8/8/8/8/8/8/K7 w - - 0 1")
        cands = p._generate_candidates(board, board.fen(), OPENING_HISTORY, "White", 10)
        assert len(cands) == 3

    def test_predict_reranked_end_to_end_with_exact_candidates(self):
        p = self._bare(_bigram_prefs("\ng1f3"), k=3)
        p.stockfish_depth = 6
        scores = {"g1f3": 10, "g1h3": 50}
        p._score_move = lambda board, mv, engine: scores.get(mv.uci(), 0)
        p._get_engine = lambda: object()
        result = p.predict_reranked(chess.STARTING_FEN, OPENING_HISTORY, "White")
        assert result.llm_top_pick.uci() == "g1f3"
        assert result.move.uci() == "g1h3"
        assert result.selected_was_llm_top is False
        assert result.num_candidates_legal == 3


class TestCandidateSourceValidation:
    def test_valid_sources(self):
        from model.inference.stockfish_rerank import (
            CANDIDATE_SOURCES,
            validate_candidate_source,
        )
        assert set(CANDIDATE_SOURCES) == {"beam", "exact"}
        assert validate_candidate_source("beam") == "beam"
        assert validate_candidate_source("exact") == "exact"

    def test_invalid_source_rejected(self):
        from model.inference.stockfish_rerank import validate_candidate_source
        with pytest.raises(ValueError, match="candidate_source"):
            validate_candidate_source("bogus")

    def test_default_is_exact_after_ab(self):
        assert DEFAULT_CANDIDATE_SOURCE == "exact"


def _example(gt: str) -> dict:
    return {"text": format_prompt(chess.STARTING_FEN, OPENING_HISTORY, "White") + gt}


class _StubExactPredictor:
    def __init__(self, order: list[str]):
        self.order = order

    def rank_legal_moves(self, fen, history, side):
        return sort_scored_moves([
            ScoredMove(move=chess.Move.from_uci(u), logprob=-float(i), n_tokens=4)
            for i, u in enumerate(self.order)
        ])


class TestMetricsExactMode:
    def test_top1_and_recall_stats(self):
        from evaluation.metrics import evaluate_predictions
        stub = _StubExactPredictor(["e2e4", "d2d4", "g1f3", "c2c4"])
        stats = evaluate_predictions(stub, [_example("g1f3"), _example("e2e4")], exact=True)
        assert stats.n_total == 2
        assert stats.is_exact
        assert stats.lmr == 1.0 and stats.parse_rate == 1.0
        assert stats.n_correct_filtered == 1
        assert stats.n_correct_raw == 1
        assert stats.n_gt_in_top5 == 2 and stats.n_gt_in_top10 == 2
        assert abs(stats.mrr - (1 / 3 + 1 / 1) / 2) < 1e-9
        d = stats.to_dict()
        assert d["recall_at_5"] == 1.0 and d["recall_at_10"] == 1.0 and "mrr" in d

    def test_gt_outside_ranking_counts_zero(self):
        from evaluation.metrics import evaluate_predictions
        stats = evaluate_predictions(_StubExactPredictor(["e2e4"]), [_example("g1f3")], exact=True)
        assert stats.n_gt_in_top5 == 0 and stats.n_gt_in_top10 == 0 and stats.mrr == 0.0

    def test_exact_is_mutually_exclusive(self):
        from evaluation.metrics import evaluate_predictions
        with pytest.raises(ValueError):
            evaluate_predictions(_StubExactPredictor(["e2e4"]), [], exact=True, prefix_constrained=True)

    def test_recall_fields_absent_when_not_exact(self):
        from evaluation.metrics import EvalStats
        s = EvalStats(n_total=10)
        assert s.recall_at_5 == 0.0 and s.recall_at_10 == 0.0 and s.mrr == 0.0
        d = s.to_dict()
        assert "recall_at_5" not in d and "mrr" not in d and "candidate_recall" not in d


class TestMetricsCandidateRecall:
    def test_candidate_recall_counted_in_rerank_mode(self):
        from evaluation.metrics import evaluate_predictions

        class _RerankResult:
            def __init__(self, cands):
                self.move = cands[0]
                self.llm_top_pick = cands[0]
                self.selected_was_llm_top = True
                self.num_candidates_legal = len(cands)
                self.all_candidates = cands

        class _Stub:
            def predict_reranked(self, fen, history, side):
                return _RerankResult([chess.Move.from_uci(u) for u in ("e2e4", "g1f3")])

        stats = evaluate_predictions(_Stub(), [_example("g1f3"), _example("d2d4")], stockfish_rerank=True)
        assert stats.n_gt_in_candidates == 1
        assert stats.candidate_recall == 0.5
        assert stats.to_dict()["candidate_recall"] == 0.5


class TestResolveModeExact:
    def test_exact_flag(self):
        from evaluation.stockfish_eval import resolve_mode
        assert resolve_mode(exact=True) == "exact"
        assert resolve_mode() == "greedy"
        assert resolve_mode(exact=True, stockfish_rerank=True) == "stockfish-rerank"


class TestPlayExactMode:
    def test_mode_registered(self):
        from model.inference.play import ALL_MODES, MODE_EXACT
        assert MODE_EXACT == "exact" and MODE_EXACT in ALL_MODES

    def test_parser_accepts_exact_and_candidate_source(self):
        from model.inference.play import _build_parser
        base = ["--adapter", "a", "--fen", chess.STARTING_FEN]
        args = _build_parser().parse_args(base + ["--mode", "exact"])
        assert args.mode == "exact" and args.rerank_candidates == DEFAULT_CANDIDATE_SOURCE
        args = _build_parser().parse_args(base + ["--rerank-candidates", "beam"])
        assert args.rerank_candidates == "beam"
        with pytest.raises(SystemExit):
            _build_parser().parse_args(base + ["--rerank-candidates", "bogus"])

    def test_format_exact_output(self):
        from model.inference.play import format_exact_output
        board = chess.Board()
        ranked = sort_scored_moves([_sm("g1f3", -1.0), _sm("e2e4", -2.0)])
        out = format_exact_output(ranked, board, show_candidates=True)
        assert "Recommended move: g1f3 (Nf3)" in out
        assert "Model confidence" in out
        assert "Legal moves available: 20" in out
        assert "1. g1f3" in out and "SELECTED" in out and "2. e2e4" in out
        brief = format_exact_output(ranked, board, show_candidates=False)
        assert "2. e2e4" not in brief


class TestGameSimulationExact:
    def test_llm_agent_exact_dispatch(self):
        from evaluation.game_simulation import LLMAgent
        from model.inference.legal_move_filter import FilterResult

        class _Stub:
            def predict_exact(self, fen, history, side):
                m = chess.Move.from_uci("e2e4")
                return FilterResult(raw="e2e4", filtered=m, was_legal=True,
                                    was_parseable=True, fallback_strategy="exact_scoring")

        agent = LLMAgent(_Stub(), "exact")
        assert agent.name == "llm-exact"
        assert agent.select_move(chess.Board(), []).uci() == "e2e4"

    def test_parser_rerank_candidates(self):
        from evaluation.game_simulation import _build_parser
        base = ["--agent-a", "random", "--agent-b", "random", "--output", "x.json"]
        assert _build_parser().parse_args(base).rerank_candidates == DEFAULT_CANDIDATE_SOURCE
        assert _build_parser().parse_args(base + ["--rerank-candidates", "beam"]).rerank_candidates == "beam"


class TestRunGameSimulationArgv:
    def _argv(self, tmp_path, **kw):
        from evaluation.run_game_simulation import build_matchup_argv
        return build_matchup_argv(
            "llm:stockfish-rerank", "random", adapter="/a", stockfish="sf",
            games=2, max_plies=10, rerank_depth=6, rerank_k=10, seed=1,
            output=tmp_path / "o.json", **kw,
        )

    def test_rerank_candidates_forwarded(self, tmp_path):
        argv = self._argv(tmp_path, rerank_candidates="beam")
        assert argv[argv.index("--rerank-candidates") + 1] == "beam"

    def test_default_follows_single_source_of_truth(self, tmp_path):
        argv = self._argv(tmp_path)
        assert argv[argv.index("--rerank-candidates") + 1] == DEFAULT_CANDIDATE_SOURCE

    def test_orchestrator_parser_has_flag(self):
        from evaluation.run_game_simulation import _build_parser
        assert _build_parser().parse_args(["--output-dir", "out"]).rerank_candidates == DEFAULT_CANDIDATE_SOURCE
        args = _build_parser().parse_args(["--output-dir", "out", "--rerank-candidates", "beam"])
        assert args.rerank_candidates == "beam"
