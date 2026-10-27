from __future__ import annotations

import pytest

from evaluation.metrics import (
    EvalStats,
    extract_fen_from_text,
    extract_ground_truth_from_text,
    extract_history_from_text,
    extract_side_from_text,
)
from model.inference.predict import (
    NO_ADAPTER,
    format_prompt,
    uses_adapter,
)


class TestFormatPrompt:
    def test_full_prompt_structure(self):
        prompt = format_prompt(
            fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
            history="e2e4 e7e5",
            side="White",
        )
        assert "### Instruction:" in prompt
        assert "### Response:" in prompt
        assert "Position (FEN): rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2" in prompt
        assert "Move history (last 5 moves): e2e4 e7e5" in prompt
        assert "Side to move: White" in prompt

    def test_history_n_configurable(self):
        prompt = format_prompt(fen="test", history="m1 m2", side="Black", history_n=3)
        assert "(last 3 moves)" in prompt

    def test_black_side(self):
        prompt = format_prompt(fen="test", history="e2e4", side="Black")
        assert "Side to move: Black" in prompt

    def test_template_matches_project_spec(self):
        prompt = format_prompt(fen="X", history="Y", side="White")
        assert prompt.startswith("### Instruction:\n")
        assert "You are a chess grandmaster" in prompt
        assert "UCI format" in prompt
        assert prompt.endswith("### Response:\n")


SAMPLE_TEXT = """### Instruction:
You are a chess grandmaster. Analyze the current position and suggest the best next move. Respond ONLY with the move in UCI format (e.g., e2e4, g1f3, e1g1 for castling).

Position (FEN): rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2
Move history (last 5 moves): e2e4 e7e5
Side to move: White

### Response:
g1f3"""


class TestExtractFenFromText:
    def test_valid_extraction(self):
        fen = extract_fen_from_text(SAMPLE_TEXT)
        assert fen == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"

    def test_no_fen_returns_empty(self):
        assert extract_fen_from_text("no fen here") == ""

    def test_empty(self):
        assert extract_fen_from_text("") == ""


class TestExtractHistoryFromText:
    def test_valid(self):
        assert extract_history_from_text(SAMPLE_TEXT) == "e2e4 e7e5"

    def test_opening(self):
        text = "Move history (last 5 moves): (none — opening)"
        assert extract_history_from_text(text) == "(none — opening)"

    def test_no_history(self):
        assert extract_history_from_text("no history line") == ""


class TestExtractSideFromText:
    def test_white(self):
        assert extract_side_from_text(SAMPLE_TEXT) == "White"

    def test_black(self):
        text = "Side to move: Black\n"
        assert extract_side_from_text(text) == "Black"

    def test_default(self):
        assert extract_side_from_text("") == "White"


class TestExtractGroundTruth:
    def test_valid(self):
        assert extract_ground_truth_from_text(SAMPLE_TEXT) == "g1f3"

    def test_with_trailing_text(self):
        text = "### Response:\ne2e4 explanation here"
        assert extract_ground_truth_from_text(text) == "e2e4"

    def test_no_response_section(self):
        assert extract_ground_truth_from_text("no response here") == ""

    def test_uppercase_normalised(self):
        text = "### Response:\nE2E4"
        assert extract_ground_truth_from_text(text) == "e2e4"


class TestEvalStats:
    def test_empty_stats_safe_divisions(self):
        s = EvalStats()
        assert s.lmr == 0.0
        assert s.top1_raw == 0.0
        assert s.top1_filtered == 0.0
        assert s.parse_rate == 0.0

    def test_metrics_computation(self):
        s = EvalStats(
            n_total=100,
            n_legal_raw=60,
            n_parseable_raw=95,
            n_correct_raw=20,
            n_correct_filtered=35,
        )
        assert s.lmr == 0.60
        assert s.parse_rate == 0.95
        assert s.top1_raw == 0.20
        assert s.top1_filtered == 0.35

    def test_to_dict(self):
        s = EvalStats(n_total=10, n_legal_raw=7, n_correct_filtered=3)
        d = s.to_dict()
        assert d["n_total"] == 10
        assert d["lmr"] == 0.70
        assert d["top1_filtered"] == 0.30
        assert isinstance(d["lmr"], float)
        assert "candidate_recall" not in d
        assert "recall_at_10" not in d

    def test_config_records_run_identity(self):
        import json
        s = EvalStats(n_total=10, adapter_path="/x/mistral7b-qlora-chess-v2/final_adapter",
                      config={"mode": "stockfish-rerank", "seed": 42,
                              "rerank_depth": 6, "rerank_k": 10, "rerank_candidates": "exact"})
        d = s.to_dict()
        assert d["config"]["mode"] == "stockfish-rerank"
        assert d["config"]["rerank_k"] == 10
        assert "v2" in d["adapter_path"]
        json.dumps(d)

    def test_config_defaults_to_empty(self):
        assert EvalStats().to_dict()["config"] == {}


class TestRoundTrip:
    def test_round_trip_extraction(self):
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
        history = "e2e4 e7e5"
        side = "White"
        ground_truth = "g1f3"

        prompt = format_prompt(fen, history, side)
        full_text = prompt + ground_truth

        assert extract_fen_from_text(full_text) == fen
        assert extract_history_from_text(full_text) == history
        assert extract_side_from_text(full_text) == side
        assert extract_ground_truth_from_text(full_text) == ground_truth


import importlib
import importlib.util

import chess as _chess


def _import_constrained_module():
    return importlib.import_module("model.inference.constrained_predict")


def _make_stub_predictor(beam_outputs: list[str]):
    import torch

    mod = _import_constrained_module()
    predictor = mod.ConstrainedChessMovePredictor.__new__(mod.ConstrainedChessMovePredictor)

    class _StubInputs(dict):
        def to(self, device):
            return self

    class _StubTokenizer:
        pad_token = "</s>"
        eos_token = "</s>"
        eos_token_id = 2

        def __call__(self, prompt, return_tensors="pt"):
            ids = torch.tensor([[1, 2, 3, 4, 5]])
            return _StubInputs(input_ids=ids, attention_mask=torch.ones_like(ids))

        def decode(self, _ids, skip_special_tokens=True):
            idx = int(_ids[0]) if len(_ids) else 0
            return beam_outputs[idx] if 0 <= idx < len(beam_outputs) else ""

    class _StubModel:
        device = "cpu"

        def generate(self, **kw):
            num = kw.get("num_return_sequences", 1)
            prompt_len = kw["input_ids"].shape[1]
            rows = [list(range(prompt_len)) + [i] for i in range(num)]
            return torch.tensor(rows)

    predictor.tokenizer = _StubTokenizer()
    predictor.model = _StubModel()
    return predictor


_HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(
    not _HAS_TORCH,
    reason="needs torch",
)
class TestConstrainedPredictorBeamSelection:
    def test_top_beam_legal_returns_top(self):
        predictor = _make_stub_predictor(["e2e4", "d2d4", "g1f3", "b1c3", "c2c4"])
        result = predictor.predict_constrained(
            fen=_chess.STARTING_FEN, history="", side="White", num_beams=5,
        )
        assert result.filtered.uci() == "e2e4"
        assert result.was_legal is True
        assert result.fallback_strategy == "top_beam_legal"

    def test_top_beam_illegal_falls_to_lower_legal(self):
        predictor = _make_stub_predictor(["e2e5", "d2d4", "a2a3", "b1c3", "g1f3"])
        result = predictor.predict_constrained(
            fen=_chess.STARTING_FEN, history="", side="White", num_beams=5,
        )
        assert result.filtered.uci() == "d2d4"
        assert result.was_legal is False
        assert result.fallback_strategy == "lower_beam_legal"

    def test_all_beams_illegal_uses_post_hoc_filter(self):
        predictor = _make_stub_predictor(["a1a8", "h1h8", "e1e2", "d1d2", "c1c2"])
        result = predictor.predict_constrained(
            fen=_chess.STARTING_FEN, history="", side="White", num_beams=5,
        )
        board = _chess.Board()
        assert result.filtered in board.legal_moves
        assert result.was_legal is False
        assert result.fallback_strategy.startswith("post_hoc_filter:")

    def test_no_legal_moves_raises(self):
        predictor = _make_stub_predictor(["e2e4"])
        mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        with pytest.raises(ValueError, match="0 legal moves"):
            predictor.predict_constrained(
                fen=mate_fen, history="", side="White", num_beams=5,
            )


class TestUsesAdapter:
    def test_none_means_base_model(self):
        assert uses_adapter(NO_ADAPTER) is False
        assert uses_adapter("None") is False
        assert uses_adapter(" none ") is False

    def test_path_or_hub_repo_means_adapter(self):
        assert uses_adapter("/content/drive/MyDrive/x/final_adapter") is True
        assert uses_adapter("Leftstam/chess-llm-mistral-7b-v2-lora") is True

    def test_empty_raises(self):
        for empty in ("", "   ", None):
            with pytest.raises(ValueError):
                uses_adapter(empty)


_HAS_HF = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "peft"))


@pytest.mark.skipif(not _HAS_HF, reason="torch/transformers/peft not available")
class TestBaseModelLoading:
    def _load(self, monkeypatch, adapter_path):
        import peft
        import torch
        import transformers

        class _Base:
            def eval(self):
                return self

        class _Tokenizer:
            pad_token = None
            eos_token = "</s>"

        peft_calls = []

        def _fake_peft(base, path):
            peft_calls.append(path)
            return base

        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **kw: False)
        monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **kw: kw)
        monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                            staticmethod(lambda *a, **kw: _Base()))
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                            staticmethod(lambda *a, **kw: _Tokenizer()))
        monkeypatch.setattr(peft.PeftModel, "from_pretrained", staticmethod(_fake_peft))

        from model.inference.predict import ChessMovePredictor
        return ChessMovePredictor(adapter_path=adapter_path), peft_calls

    def test_none_skips_adapter(self, monkeypatch):
        predictor, peft_calls = self._load(monkeypatch, NO_ADAPTER)
        assert peft_calls == []
        assert predictor.tokenizer.pad_token == "</s>"

    def test_path_loads_adapter(self, monkeypatch):
        _, peft_calls = self._load(monkeypatch, "/x/final_adapter")
        assert peft_calls == ["/x/final_adapter"]
