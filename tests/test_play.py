from __future__ import annotations

import chess
import pytest

from model.inference.play import (
    ALL_MODES,
    DEFAULT_MODE,
    _build_parser,
    classify_eval,
    format_basic_output,
    format_eval_pawns,
    format_stockfish_rerank_output,
)


class TestFormatEvalPawns:
    @pytest.mark.parametrize("cp, expected", [
        (0, "+0.00"),
        (50, "+0.50"),
        (100, "+1.00"),
        (-30, "-0.30"),
        (1500, "+15.00"),
    ])
    def test_formatting(self, cp, expected):
        assert format_eval_pawns(cp) == expected


class TestClassifyEval:
    @pytest.mark.parametrize("cp, expected", [
        (0, "near equal"),
        (40, "near equal"),
        (-40, "near equal"),
        (100, "slight advantage"),
        (-100, "slight disadvantage"),
        (300, "clear advantage"),
        (-300, "clear disadvantage"),
        (800, "winning"),
        (-800, "losing"),
    ])
    def test_classification(self, cp, expected):
        assert classify_eval(cp) == expected


class _FakeRerankResult:
    def __init__(self, move, eval_cp, llm_top_pick, all_candidates, all_scores_cp):
        self.move = move
        self.stockfish_eval_cp = eval_cp
        self.num_candidates_legal = len(all_candidates)
        self.llm_top_pick = llm_top_pick
        self.selected_was_llm_top = (move == llm_top_pick)
        self.all_candidates = all_candidates
        self.all_scores_cp = all_scores_cp


class TestFormatStockfishRerankOutput:
    def test_llm_agreed_with_stockfish(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = _FakeRerankResult(
            move=move, eval_cp=30, llm_top_pick=move,
            all_candidates=[move], all_scores_cp=[30],
        )
        output = format_stockfish_rerank_output(result, board)
        assert "e2e4" in output
        assert "+0.30" in output
        assert "LLM ranked this candidate #1" in output
        assert "overrode" not in output.lower()

    def test_stockfish_overrode_llm(self):
        board = chess.Board()
        llm_top = chess.Move.from_uci("e2e4")
        selected = chess.Move.from_uci("d2d4")
        result = _FakeRerankResult(
            move=selected, eval_cp=50, llm_top_pick=llm_top,
            all_candidates=[llm_top, selected],
            all_scores_cp=[10, 50],
        )
        output = format_stockfish_rerank_output(result, board)
        assert "d2d4" in output
        assert "+0.50" in output
        assert "overrode" in output.lower()
        assert "e2e4" in output

    def test_shows_candidates_when_requested(self):
        board = chess.Board()
        cands = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "g1f3")]
        scores = [30, 50, 20]
        result = _FakeRerankResult(
            move=cands[1],
            eval_cp=50, llm_top_pick=cands[0],
            all_candidates=cands, all_scores_cp=scores,
        )
        output = format_stockfish_rerank_output(result, board, show_candidates=True)
        for uci in ("e2e4", "d2d4", "g1f3"):
            assert uci in output
        assert "SELECTED" in output
        assert "+0.50" in output
        assert "+0.30" in output
        assert "+0.20" in output

    def test_does_not_show_candidates_by_default(self):
        board = chess.Board()
        cands = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4")]
        result = _FakeRerankResult(
            move=cands[0], eval_cp=30, llm_top_pick=cands[0],
            all_candidates=cands, all_scores_cp=[30, 10],
        )
        output = format_stockfish_rerank_output(result, board, show_candidates=False)
        assert "SELECTED" not in output

    def test_includes_legal_moves_count(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = _FakeRerankResult(
            move=move, eval_cp=30, llm_top_pick=move,
            all_candidates=[move], all_scores_cp=[30],
        )
        output = format_stockfish_rerank_output(result, board)
        assert "Legal moves available: 20" in output

    def test_negative_eval_shows_as_disadvantage(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = _FakeRerankResult(
            move=move, eval_cp=-100, llm_top_pick=move,
            all_candidates=[move], all_scores_cp=[-100],
        )
        output = format_stockfish_rerank_output(result, board)
        assert "-1.00" in output
        assert "disadvantage" in output


class TestFormatBasicOutput:
    def test_basic_format(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        output = format_basic_output(move, board, "prefix-constrained (token-level legality)")
        assert "e2e4" in output
        assert "prefix-constrained" in output
        assert "Side to move: White" in output
        assert "Legal moves available: 20" in output

    def test_extra_lines(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        extra = ["  Note: Controls the center."]
        output = format_basic_output(move, board, "greedy + post-hoc filter", extra)
        assert "Controls the center" in output


class TestCliParser:
    @pytest.fixture
    def parser(self):
        return _build_parser()

    def test_requires_fen_and_adapter_defaults_to_deployed_v2(self, parser):
        from model.config.training_config import BEST_HUB_REPO
        with pytest.raises(SystemExit):
            parser.parse_args([])
        with pytest.raises(SystemExit):
            parser.parse_args(["--adapter", "x"])
        args = parser.parse_args(["--fen", chess.STARTING_FEN])
        assert args.adapter == BEST_HUB_REPO
        assert "v2" in args.adapter

    def test_minimal_invocation(self, parser):
        args = parser.parse_args([
            "--adapter", "Leftstam/chess-llm-mistral-7b-v2-lora",
            "--fen", chess.STARTING_FEN,
        ])
        assert args.adapter == "Leftstam/chess-llm-mistral-7b-v2-lora"
        assert args.fen == chess.STARTING_FEN
        assert args.mode == DEFAULT_MODE
        assert args.show_candidates is False
        assert args.history == "(none — opening)"
        assert args.side is None

    def test_mode_choices(self, parser):
        for mode in ALL_MODES:
            args = parser.parse_args([
                "--adapter", "x", "--fen", chess.STARTING_FEN, "--mode", mode,
            ])
            assert args.mode == mode

    def test_invalid_mode_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--adapter", "x", "--fen", chess.STARTING_FEN, "--mode", "invalid",
            ])

    def test_show_candidates_flag(self, parser):
        args = parser.parse_args([
            "--adapter", "x", "--fen", chess.STARTING_FEN, "--show-candidates",
        ])
        assert args.show_candidates is True

    def test_side_choice(self, parser):
        args = parser.parse_args([
            "--adapter", "x", "--fen", chess.STARTING_FEN, "--side", "Black",
        ])
        assert args.side == "Black"

    def test_stockfish_rerank_knobs(self, parser):
        args = parser.parse_args([
            "--adapter", "x", "--fen", chess.STARTING_FEN,
            "--stockfish", "/custom/path/stockfish",
            "--stockfish-depth", "8",
            "--num-candidates", "15",
        ])
        assert args.stockfish == "/custom/path/stockfish"
        assert args.stockfish_depth == 8
        assert args.num_candidates == 15
