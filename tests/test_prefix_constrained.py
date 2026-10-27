from __future__ import annotations

import chess
import pytest

from model.inference.prefix_constrained import (
    RESPONSE_ANCHOR,
    build_legal_move_token_sets,
    make_prefix_allowed_tokens_fn,
    tokenize_move_in_context,
)


class _FakeEncoding:
    def __init__(self, ids):
        self.input_ids = ids


class FakeCharTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) for c in text]
        return _FakeEncoding(ids)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i != 0)


@pytest.fixture
def tk():
    return FakeCharTokenizer()


class TestTokenizeMoveInContext:
    def test_returns_only_move_tokens(self, tk):
        result = tokenize_move_in_context(tk, "X", "e2e4")
        assert result == [ord("e"), ord("2"), ord("e"), ord("4")]

    def test_with_realistic_anchor(self, tk):
        result = tokenize_move_in_context(tk, RESPONSE_ANCHOR, "g1f3")
        assert result == [ord("g"), ord("1"), ord("f"), ord("3")]

    def test_promotion_move(self, tk):
        result = tokenize_move_in_context(tk, "X", "e7e8q")
        assert result == [ord("e"), ord("7"), ord("e"), ord("8"), ord("q")]

    def test_castling_move(self, tk):
        result = tokenize_move_in_context(tk, "X", "e1g1")
        assert result == [ord("e"), ord("1"), ord("g"), ord("1")]


class TestBuildLegalMoveTokenSets:
    def test_starting_position(self, tk):
        board = chess.Board()
        sets = build_legal_move_token_sets(tk, board)
        assert len(sets) == 20
        assert all(len(s) == 4 for s in sets)
        e2e4_tokens = [ord(c) for c in "e2e4"]
        assert e2e4_tokens in sets

    def test_promotion_position(self, tk):
        board = chess.Board("7K/4P3/8/8/8/8/8/k7 w - - 0 1")
        sets = build_legal_move_token_sets(tk, board)
        promo_uci = ["e7e8q", "e7e8r", "e7e8b", "e7e8n"]
        for p in promo_uci:
            tokens = [ord(c) for c in p]
            assert tokens in sets, f"Missing promotion: {p}"
        promo_token_sets = [s for s in sets if len(s) == 5]
        assert len(promo_token_sets) == 4, f"Expected 4 promotions, got {len(promo_token_sets)}"


class TestPrefixAllowedTokensFn:
    @pytest.fixture
    def legal_seqs(self):
        return [
            [ord(c) for c in "e2e4"],
            [ord(c) for c in "e2e3"],
            [ord(c) for c in "g1f3"],
        ]

    def test_first_step_allows_first_chars_of_all_moves(self, legal_seqs):
        fn = make_prefix_allowed_tokens_fn(legal_seqs, prompt_length=5, eos_token_id=0)
        allowed = fn(0, [1, 2, 3, 4, 5])
        assert set(allowed) == {ord("e"), ord("g")}

    def test_after_e_allows_only_2(self, legal_seqs):
        fn = make_prefix_allowed_tokens_fn(legal_seqs, prompt_length=5, eos_token_id=0)
        allowed = fn(0, [1, 2, 3, 4, 5, ord("e")])
        assert set(allowed) == {ord("2")}

    def test_after_e2e_allows_3_or_4(self, legal_seqs):
        fn = make_prefix_allowed_tokens_fn(legal_seqs, prompt_length=5, eos_token_id=0)
        prompt = [1, 2, 3, 4, 5]
        allowed = fn(0, prompt + [ord("e"), ord("2"), ord("e")])
        assert set(allowed) == {ord("3"), ord("4")}

    def test_after_g_allows_only_1(self, legal_seqs):
        fn = make_prefix_allowed_tokens_fn(legal_seqs, prompt_length=5, eos_token_id=0)
        prompt = [1, 2, 3, 4, 5]
        allowed = fn(0, prompt + [ord("g")])
        assert set(allowed) == {ord("1")}

    def test_complete_move_returns_eos(self, legal_seqs):
        fn = make_prefix_allowed_tokens_fn(legal_seqs, prompt_length=5, eos_token_id=99)
        prompt = [1, 2, 3, 4, 5]
        allowed = fn(0, prompt + [ord(c) for c in "e2e4"])
        assert allowed == [99]

    def test_complete_alternative_move_returns_eos(self, legal_seqs):
        fn = make_prefix_allowed_tokens_fn(legal_seqs, prompt_length=5, eos_token_id=99)
        prompt = [1, 2, 3, 4, 5]
        allowed = fn(0, prompt + [ord(c) for c in "g1f3"])
        assert allowed == [99]

    def test_no_match_returns_eos_defensively(self, legal_seqs):
        fn = make_prefix_allowed_tokens_fn(legal_seqs, prompt_length=5, eos_token_id=99)
        prompt = [1, 2, 3, 4, 5]
        allowed = fn(0, prompt + [ord("z")])
        assert allowed == [99]

    def test_works_with_tensor_input(self, legal_seqs):
        class FakeTensor:
            def __init__(self, data):
                self.data = data
            def tolist(self):
                return self.data

        fn = make_prefix_allowed_tokens_fn(legal_seqs, prompt_length=5, eos_token_id=0)
        allowed = fn(0, FakeTensor([1, 2, 3, 4, 5, ord("e"), ord("2")]))
        assert set(allowed) == {ord("e")}

    def test_prompt_length_zero_works(self, legal_seqs):
        fn = make_prefix_allowed_tokens_fn(legal_seqs, prompt_length=0, eos_token_id=0)
        allowed = fn(0, [ord("e")])
        assert set(allowed) == {ord("2")}


class TestIntegrationStartingPosition:
    def test_e2e4_path_navigates_correctly(self, tk):
        board = chess.Board()
        legal_seqs = build_legal_move_token_sets(tk, board)
        fn = make_prefix_allowed_tokens_fn(legal_seqs, prompt_length=0, eos_token_id=0)

        first_step = fn(0, [])
        expected_starts = {ord(c) for c in "abcdefgh"}
        assert set(first_step) == expected_starts

        step2 = fn(0, [ord("e")])
        assert set(step2) == {ord("2")}

        step3 = fn(0, [ord("e"), ord("2")])
        assert set(step3) == {ord("e")}

        step4 = fn(0, [ord("e"), ord("2"), ord("e")])
        assert set(step4) == {ord("3"), ord("4")}

        step5 = fn(0, [ord("e"), ord("2"), ord("e"), ord("4")])
        assert step5 == [0]
