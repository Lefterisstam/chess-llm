from __future__ import annotations

import chess
import pytest

from model.inference.legal_move_filter import (
    extract_uci_candidate,
    filter_to_legal,
    find_closest_legal,
    try_parse_uci,
)


class TestExtractUciCandidate:
    @pytest.mark.parametrize("text, expected", [
        ("e2e4", "e2e4"),
        ("e2e4\nSomething else", "e2e4"),
        ("  e2e4  ", "e2e4"),
        ("e7e8q", "e7e8q"),
        ("e2e4 will be played", "e2e4"),
        ("E2E4", "e2e4"),
        ("e2e4.", "e2e4"),
        ("e2e4,", "e2e4"),
        ("", ""),
        ("\n\n", ""),
        ("not a move", "not"),
    ])
    def test_extraction(self, text, expected):
        assert extract_uci_candidate(text) == expected


class TestTryParseUci:
    def test_valid_simple_move(self):
        m = try_parse_uci("e2e4")
        assert m is not None
        assert m.uci() == "e2e4"

    def test_valid_promotion(self):
        m = try_parse_uci("e7e8q")
        assert m is not None
        assert m.uci() == "e7e8q"

    @pytest.mark.parametrize("bad", [
        "",
        "e2",
        "e2e",
        "i2i4",
        "e2e9",
        "xyz",
        "12345",
    ])
    def test_invalid(self, bad):
        assert try_parse_uci(bad) is None

    def test_5char_with_invalid_5th_falls_back_to_4(self):
        m = try_parse_uci("e2e4x")
        assert m is not None
        assert m.uci() == "e2e4"

        m = try_parse_uci("e2e4q5")
        assert m is not None
        assert m.uci() == "e2e4"


class TestFindClosestLegal:
    def test_same_from_priority(self):
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        target = chess.Move.from_uci("c8e6")
        closest = find_closest_legal(target, board)
        assert closest in board.legal_moves

    def test_same_to_priority(self):
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
        target = chess.Move.from_uci("d2d5")
        closest = find_closest_legal(target, board)
        assert closest in board.legal_moves
        assert closest.from_square == chess.D2

    def test_fallback_to_first_legal(self):
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6q1/5P2/PPPPP1PP/RNBQKBNR w KQkq - 1 3")
        target = chess.Move.from_uci("a1a8")
        closest = find_closest_legal(target, board)
        assert closest in board.legal_moves


class TestFilterToLegal:
    def test_legal_move_passes_through(self):
        board = chess.Board()
        result = filter_to_legal("e2e4", board)
        assert result.was_legal is True
        assert result.was_parseable is True
        assert result.fallback_strategy == "direct"
        assert result.filtered.uci() == "e2e4"

    def test_illegal_but_parseable_uses_fallback(self):
        board = chess.Board()
        result = filter_to_legal("e3e5", board)
        assert result.was_legal is False
        assert result.was_parseable is True
        assert result.filtered in board.legal_moves
        assert result.fallback_strategy != "direct"

    def test_malformed_falls_back(self):
        board = chess.Board()
        result = filter_to_legal("nonsense", board)
        assert result.was_legal is False
        assert result.was_parseable is False
        assert result.filtered in board.legal_moves
        assert result.fallback_strategy == "first_legal"

    def test_empty_falls_back(self):
        board = chess.Board()
        result = filter_to_legal("", board)
        assert result.was_legal is False
        assert result.was_parseable is False
        assert result.filtered in board.legal_moves

    def test_ruy_lopez_illegal_c8e6(self):
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        assert chess.Move.from_uci("c8e6") not in board.legal_moves

        result = filter_to_legal("c8e6", board)
        assert result.was_parseable is True
        assert result.was_legal is False
        assert result.filtered in board.legal_moves

    def test_uppercase_input(self):
        board = chess.Board()
        result = filter_to_legal("E2E4", board)
        assert result.was_legal is True
        assert result.filtered.uci() == "e2e4"

    def test_extra_whitespace_and_text(self):
        board = chess.Board()
        result = filter_to_legal("  e2e4  is the best move", board)
        assert result.was_legal is True
        assert result.filtered.uci() == "e2e4"

    def test_promotion_legal(self):
        board = chess.Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")
        result = filter_to_legal("e7e8q", board)
        assert result.was_legal is True
        assert result.filtered.uci() == "e7e8q"

    def test_filter_result_dataclass(self):
        board = chess.Board()
        result = filter_to_legal("e2e4", board)
        assert hasattr(result, "raw")
        assert hasattr(result, "filtered")
        assert hasattr(result, "was_legal")
        assert hasattr(result, "was_parseable")
        assert hasattr(result, "fallback_strategy")


class TestNoLegalMoves:
    def test_raises_on_checkmate(self):
        board = chess.Board()
        board.push_san("f3")
        board.push_san("e5")
        board.push_san("g4")
        board.push_san("Qh4#")
        assert board.is_checkmate()
        with pytest.raises(ValueError, match="no legal moves"):
            filter_to_legal("e2e4", board)
