from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import chess
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_HAS_FLASK = importlib.util.find_spec("flask") is not None


class _FakeRerankResult:
    def __init__(self, move, eval_cp, llm_top_pick, all_candidates, all_scores_cp):
        self.move = move
        self.stockfish_eval_cp = eval_cp
        self.num_candidates_legal = len(all_candidates)
        self.llm_top_pick = llm_top_pick
        self.selected_was_llm_top = (move == llm_top_pick)
        self.all_candidates = all_candidates
        self.all_scores_cp = all_scores_cp


class TestSerializeRerankResult:
    def test_basic_serialization(self):
        from web.app import serialize_rerank_result
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = _FakeRerankResult(
            move=move, eval_cp=30, llm_top_pick=move,
            all_candidates=[move], all_scores_cp=[30],
        )
        out = serialize_rerank_result(result, board)
        assert out["move_uci"] == "e2e4"
        assert out["move_san"] == "e4"
        assert out["eval_cp"] == 30
        assert out["eval_pawns"] == 0.30
        assert out["llm_top_uci"] == "e2e4"
        assert out["selected_was_llm_top"] is True
        assert out["num_candidates_legal"] == 1
        assert len(out["candidates"]) == 1

    def test_candidate_list_includes_selected_marker(self):
        from web.app import serialize_rerank_result
        board = chess.Board()
        cands = [chess.Move.from_uci(u) for u in ("e2e4", "d2d4", "g1f3")]
        scores = [10, 50, 20]
        result = _FakeRerankResult(
            move=cands[1], eval_cp=50, llm_top_pick=cands[0],
            all_candidates=cands, all_scores_cp=scores,
        )
        out = serialize_rerank_result(result, board)
        assert len(out["candidates"]) == 3
        selected = [c for c in out["candidates"] if c["selected"]]
        assert len(selected) == 1
        assert selected[0]["uci"] == "d2d4"

    def test_negative_eval_serialization(self):
        from web.app import serialize_rerank_result
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = _FakeRerankResult(
            move=move, eval_cp=-200, llm_top_pick=move,
            all_candidates=[move], all_scores_cp=[-200],
        )
        out = serialize_rerank_result(result, board)
        assert out["eval_cp"] == -200
        assert out["eval_pawns"] == -2.00

    def test_eval_cp_returned_as_int(self):
        from web.app import serialize_rerank_result
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = _FakeRerankResult(
            move=move, eval_cp=42, llm_top_pick=move,
            all_candidates=[move], all_scores_cp=[42],
        )
        out = serialize_rerank_result(result, board)
        assert isinstance(out["eval_cp"], int)
        assert isinstance(out["candidates"][0]["eval_cp"], int)


@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not installed")
class TestHealthEndpoint:
    def test_health_without_predictor(self):
        from web.app import create_app
        app = create_app(predictor=None)
        client = app.test_client()
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["predictor_loaded"] is False

    def test_health_with_predictor(self):
        from web.app import create_app
        app = create_app(predictor=object())
        client = app.test_client()
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["predictor_loaded"] is True

    def test_health_includes_server_meta(self):
        from web.app import create_app
        meta = {
            "device": "cpu",
            "adapter": "Leftstam/chess-llm-mistral-7b-v2-lora",
            "num_candidates": 10,
            "stockfish_depth": 6,
        }
        app = create_app(predictor=object(), meta=meta)
        client = app.test_client()
        data = client.get("/api/health").get_json()
        assert data["device"] == "cpu"
        assert data["num_candidates"] == 10
        assert data["stockfish_depth"] == 6

    def test_health_meta_optional(self):
        from web.app import create_app
        app = create_app(predictor=None)
        data = app.test_client().get("/api/health").get_json()
        assert "device" not in data
        assert data["status"] == "ok"


@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not installed")
class TestPredictEndpoint:
    def _make_stub_predictor(self, return_value):
        class _StubPredictor:
            def predict_reranked(self, fen, history, side):
                return return_value
        return _StubPredictor()

    def test_predict_no_predictor(self):
        from web.app import create_app
        app = create_app(predictor=None)
        client = app.test_client()
        response = client.post("/api/predict", json={"fen": chess.STARTING_FEN})
        assert response.status_code == 503

    def test_predict_missing_fen(self):
        from web.app import create_app
        app = create_app(predictor=object())
        client = app.test_client()
        response = client.post("/api/predict", json={})
        assert response.status_code == 400
        assert "fen required" in response.get_json()["error"]

    def test_predict_invalid_fen(self):
        from web.app import create_app
        app = create_app(predictor=object())
        client = app.test_client()
        response = client.post("/api/predict", json={"fen": "not a fen"})
        assert response.status_code == 400
        assert "invalid FEN" in response.get_json()["error"]

    def test_predict_checkmate_position(self):
        from web.app import create_app
        app = create_app(predictor=object())
        client = app.test_client()
        mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        response = client.post("/api/predict", json={"fen": mate_fen})
        assert response.status_code == 400
        assert "no legal moves" in response.get_json()["error"]

    def test_predict_happy_path(self):
        from web.app import create_app
        e2e4 = chess.Move.from_uci("e2e4")
        stub_result = _FakeRerankResult(
            move=e2e4, eval_cp=30, llm_top_pick=e2e4,
            all_candidates=[e2e4], all_scores_cp=[30],
        )
        predictor = self._make_stub_predictor(stub_result)
        app = create_app(predictor=predictor)
        client = app.test_client()
        response = client.post("/api/predict", json={
            "fen": chess.STARTING_FEN,
            "history": "(none — opening)",
        })
        assert response.status_code == 200
        data = response.get_json()
        assert data["move_uci"] == "e2e4"
        assert data["move_san"] == "e4"
        assert data["eval_cp"] == 30
        assert "inference_ms" in data
        assert isinstance(data["inference_ms"], int)
        assert data["inference_ms"] >= 0

    def test_predict_propagates_inference_error(self):
        from web.app import create_app

        class _BrokenPredictor:
            def predict_reranked(self, fen, history, side):
                raise RuntimeError("simulated GPU error")

        app = create_app(predictor=_BrokenPredictor())
        client = app.test_client()
        response = client.post("/api/predict", json={"fen": chess.STARTING_FEN})
        assert response.status_code == 500
        assert "simulated GPU error" in response.get_json()["error"]


@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not installed")
class TestIndexAndStatic:
    def test_root_returns_html(self):
        from web.app import create_app
        app = create_app(predictor=None)
        client = app.test_client()
        response = client.get("/")
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "<title>Chess LLM — Mistral 7B</title>" in body
        assert "chessboard" in body.lower()

    def test_index_has_both_tabs(self):
        from web.app import create_app
        body = create_app(predictor=None).test_client().get("/").get_data(as_text=True)
        assert 'data-tab="play"' in body and 'id="tab-play"' in body
        assert 'data-tab="ask"' in body and 'id="tab-ask"' in body
        assert 'id="analysis-moves"' in body
        assert 'id="ask-board"' in body
        assert 'id="btn-continue-play"' in body
        assert 'id="btn-load-fen"' not in body
        assert 'id="analysis-start-fen"' not in body

    def test_js_references_only_existing_ids(self):
        import re
        static = PROJECT_ROOT / "web" / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        js = (static / "script.js").read_text(encoding="utf-8")
        html_ids = set(re.findall(r'id="([^"]+)"', html))
        js_ids = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', js))
        js_ids |= set(re.findall(r'Chessboard\("([A-Za-z0-9_-]+)"', js))
        dynamic = {"retry-model"}
        missing = sorted(js_ids - html_ids - dynamic)
        assert missing == [], f"JS references ids missing from HTML: {missing}"


class TestExampleMoves:
    def _examples(self) -> list[str]:
        import json
        import re
        js = (PROJECT_ROOT / "web" / "static" / "script.js").read_text(encoding="utf-8")
        m = re.search(r"var EXAMPLES = \[(.*?)\];", js, flags=re.DOTALL)
        assert m, "EXAMPLES array not found in script.js"
        return json.loads("[" + m.group(1) + "]")

    def test_at_least_three_examples(self):
        assert len(self._examples()) >= 3

    def test_every_example_is_a_legal_sequence(self):
        from web.app import parse_moves_text
        for ex in self._examples():
            board, uci, san = parse_moves_text(ex)
            assert len(uci) >= 3, ex
            assert not board.is_game_over(), ex

    def test_examples_cover_san_uci_and_pgn(self):
        exs = self._examples()
        assert any(e.lstrip().startswith("1.") for e in exs)
        assert any(re.fullmatch(r"([a-h][1-8][a-h][1-8][qrbn]?\s*)+", e) for e in exs)
        assert any(e.lstrip().startswith("[") for e in exs)

    def test_examples_include_both_sides_to_move(self):
        from web.app import parse_moves_text
        sides = {parse_moves_text(e)[0].turn for e in self._examples()}
        assert sides == {chess.WHITE, chess.BLACK}


RUY_LOPEZ_FEN = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


class TestParseMovesText:
    def test_san_with_move_numbers(self):
        from web.app import parse_moves_text
        board, uci, san = parse_moves_text("1. e4 e5 2. Nf3 Nc6 3. Bb5")
        assert uci == ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
        assert san == ["e4", "e5", "Nf3", "Nc6", "Bb5"]
        assert board.fen() == RUY_LOPEZ_FEN
        assert board.turn == chess.BLACK

    def test_uci_tokens(self):
        from web.app import parse_moves_text
        board, uci, san = parse_moves_text("e2e4 e7e5 g1f3")
        assert uci == ["e2e4", "e7e5", "g1f3"]
        assert san == ["e4", "e5", "Nf3"]

    def test_glued_numbers_black_ellipsis_and_annotations(self):
        from web.app import parse_moves_text
        _, uci, _ = parse_moves_text("1.e4! e5?! 2.Nf3 Nc6 3.Bb5 3... a6")
        assert uci == ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]

    def test_pgn_headers_comments_variations_result_ignored(self):
        from web.app import parse_moves_text
        pgn = ('[Event "Casual"]\n[Site "lichess.org"]\n[Result "1-0"]\n\n'
               '1. e4 {best by test} e5 (1... c5 2. Nf3 (2. c3)) 2. Nf3 $1 Nc6 1-0')
        _, uci, _ = parse_moves_text(pgn)
        assert uci == ["e2e4", "e7e5", "g1f3", "b8c6"]

    def test_castling_both_notations(self):
        from web.app import parse_moves_text
        base = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. "
        for castle in ("O-O", "0-0"):
            board, uci, san = parse_moves_text(base + castle)
            assert uci[-1] == "e1g1" and san[-1] == "O-O"
            assert len(uci) == 9

    def test_promotion_with_start_fen(self):
        from web.app import parse_moves_text
        board, uci, san = parse_moves_text("e7e8q", start_fen="7K/4P3/8/8/8/8/8/k7 w - - 0 1")
        assert uci == ["e7e8q"]
        assert san[0].startswith("e8=Q")
        assert board.piece_at(chess.E8).piece_type == chess.QUEEN

    def test_illegal_move_reports_index_and_token(self):
        from web.app import parse_moves_text
        with pytest.raises(ValueError, match=r"#2 «e4»"):
            parse_moves_text("1. e4 e4")

    def test_unrecognized_token_reports_index(self):
        from web.app import parse_moves_text
        with pytest.raises(ValueError, match=r"#3 «xyz»"):
            parse_moves_text("1. e4 e5 xyz")

    def test_invalid_start_fen(self):
        from web.app import parse_moves_text
        with pytest.raises(ValueError, match="Άκυρο αρχικό FEN"):
            parse_moves_text("e2e4", start_fen="not a fen")

    def test_empty_text_is_start_position(self):
        from web.app import parse_moves_text
        board, uci, san = parse_moves_text("")
        assert uci == [] and san == []
        assert board.fen() == chess.STARTING_FEN

    def test_uci_preferred_over_san_for_uci_shaped_tokens(self):
        from web.app import parse_moves_text
        _, uci, san = parse_moves_text("b1c3")
        assert uci == ["b1c3"] and san == ["Nc3"]


class TestHistoryString:
    def test_empty(self):
        from web.app import NO_HISTORY, history_string
        assert history_string([]) == NO_HISTORY == "(none — opening)"

    def test_short(self):
        from web.app import history_string
        assert history_string(["e2e4", "e7e5", "g1f3"]) == "e2e4 e7e5 g1f3"

    def test_last_ten_plies_only(self):
        from web.app import HISTORY_PLIES, history_string
        moves = [f"m{i}" for i in range(14)]
        out = history_string(moves).split()
        assert HISTORY_PLIES == 10
        assert out == moves[-10:]


class TestLooksLikeFen:
    def test_fen(self):
        from web.app import looks_like_fen
        assert looks_like_fen(RUY_LOPEZ_FEN)
        assert looks_like_fen(chess.STARTING_FEN)

    def test_moves_are_not_fen(self):
        from web.app import looks_like_fen
        assert not looks_like_fen("1. e4 e5 2. Nf3")
        assert not looks_like_fen("e2e4 e7e5")
        assert not looks_like_fen("")


@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not installed")
class TestAdviseEndpoint:
    class _RecordingPredictor:
        def __init__(self):
            self.calls = []

        def predict_reranked(self, fen, history, side):
            self.calls.append((fen, history, side))
            board = chess.Board(fen)
            move = chess.Move.from_uci("a7a6")
            if move not in board.legal_moves:
                move = sorted(board.legal_moves, key=lambda m: m.uci())[0]
            return _FakeRerankResult(move=move, eval_cp=15, llm_top_pick=move,
                                     all_candidates=[move], all_scores_cp=[15])

    def _client(self, predictor):
        from web.app import create_app
        return create_app(predictor=predictor).test_client()

    def test_no_predictor(self):
        resp = self._client(None).post("/api/advise", json={"moves": "1. e4"})
        assert resp.status_code == 503

    def test_missing_moves(self):
        resp = self._client(object()).post("/api/advise", json={})
        assert resp.status_code == 400
        assert "moves required" in resp.get_json()["error"]

    def test_bad_move_returns_400_with_index(self):
        resp = self._client(object()).post("/api/advise", json={"moves": "1. e4 e4"})
        assert resp.status_code == 400
        assert "#2" in resp.get_json()["error"]

    def test_game_over_returns_400(self):
        resp = self._client(object()).post("/api/advise", json={"moves": "1. f3 e5 2. g4 Qh4#"})
        assert resp.status_code == 400
        assert "τελειώσει" in resp.get_json()["error"]

    def test_happy_path_sends_training_format_history(self):
        p = self._RecordingPredictor()
        resp = self._client(p).post("/api/advise", json={"moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5"})
        assert resp.status_code == 200
        d = resp.get_json()
        assert d["fen"] == RUY_LOPEZ_FEN
        assert d["side_to_move"] == "Black"
        assert d["ply"] == 5
        assert d["san_moves"] == ["e4", "e5", "Nf3", "Nc6", "Bb5"]
        assert d["uci_moves"] == ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
        assert d["history_used"] == "e2e4 e7e5 g1f3 b8c6 f1b5"
        assert d["history_provided"] is True
        assert d["warning"] is None
        assert d["move_uci"] == "a7a6" and d["move_san"] == "a6"
        assert isinstance(d["inference_ms"], int)
        assert p.calls == [(RUY_LOPEZ_FEN, "e2e4 e7e5 g1f3 b8c6 f1b5", "Black")]

    def test_long_game_uses_last_ten_plies(self):
        p = self._RecordingPredictor()
        moves = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6"
        resp = self._client(p).post("/api/advise", json={"moves": moves})
        assert resp.status_code == 200
        d = resp.get_json()
        assert d["ply"] == 14
        assert len(d["history_used"].split()) == 10
        assert d["history_used"].split()[-1] == "d7d6"

    def test_fen_only_fallback_warns_and_uses_no_history(self):
        from web.app import NO_HISTORY
        p = self._RecordingPredictor()
        resp = self._client(p).post("/api/advise", json={"moves": RUY_LOPEZ_FEN})
        assert resp.status_code == 200
        d = resp.get_json()
        assert d["history_provided"] is False
        assert d["history_used"] == NO_HISTORY
        assert d["warning"] and "ιστορικό" in d["warning"]
        assert p.calls[0][1] == NO_HISTORY

    def test_start_fen_plus_moves(self):
        p = self._RecordingPredictor()
        resp = self._client(p).post("/api/advise", json={
            "moves": "a6", "start_fen": RUY_LOPEZ_FEN,
        })
        assert resp.status_code == 200
        d = resp.get_json()
        assert d["ply"] == 1 and d["side_to_move"] == "White"
        assert d["history_used"] == "a7a6"
        assert d["start_fen"] == RUY_LOPEZ_FEN

    def test_predict_endpoint_unchanged_for_play_tab(self):
        p = self._RecordingPredictor()
        resp = self._client(p).post("/api/predict", json={"fen": RUY_LOPEZ_FEN, "history": "e2e4 e7e5 g1f3 b8c6 f1b5"})
        assert resp.status_code == 200
        assert p.calls == [(RUY_LOPEZ_FEN, "e2e4 e7e5 g1f3 b8c6 f1b5", "Black")]
