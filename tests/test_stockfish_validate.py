from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.engine
import pgn_to_training as ptt
import pytest
import stockfish_validate as sv


class TestComputeCpLoss:
    @pytest.mark.parametrize("best, human, expected", [
        (50, 50, 0),
        (100, 50, 50),
        (-50, -100, 50),
        (50, 100, 0),
        (10000, 100, 9900),
    ])
    def test_cases(self, best, human, expected):
        assert sv.compute_cp_loss(best, human) == expected


class FakeEngine:
    def __init__(self, results: dict[str, dict]):
        self.results = results
        self.quit_called = False
        self.config_options: dict = {}
        self.analyse_calls: list[str] = []

    def configure(self, options: dict) -> None:
        self.config_options.update(options)

    def analyse(self, board, limit, **kwargs):
        self.analyse_calls.append(board.fen())
        fen = board.fen()
        if fen in self.results:
            entry = self.results[fen]
            best_uci = entry["best_move"]
            cp_value = entry["best_cp"]
        else:
            first = next(iter(board.legal_moves), None)
            if first is None:
                pov_score = chess.engine.PovScore(chess.engine.Mate(0), board.turn)
                return {"score": pov_score, "pv": []}
            best_uci = first.uci()
            cp_value = 0

        best_move = chess.Move.from_uci(best_uci)
        pov_score = chess.engine.PovScore(chess.engine.Cp(cp_value), board.turn)
        return {"score": pov_score, "pv": [best_move]}

    def quit(self) -> None:
        self.quit_called = True


def _example_text(fen: str, target_uci: str) -> str:
    return ptt.format_training_example(
        fen=fen, history_uci=[], side_white=(" w " in fen), target_uci=target_uci,
    )


def _write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class TestEvaluatePositionAndMove:
    def test_same_move_zero_loss(self):
        board = chess.Board()
        engine = FakeEngine({board.fen(): {"best_move": "e2e4", "best_cp": 30}})
        cp_loss, best_uci, best_cp, human_cp = sv.evaluate_position_and_move(
            engine, board, chess.Move.from_uci("e2e4"), depth=8,
        )
        assert cp_loss == 0
        assert best_uci == "e2e4"
        assert best_cp == 30
        assert human_cp == 30
        assert len(engine.analyse_calls) == 1

    def test_different_move_computes_loss(self):
        board = chess.Board()
        board_after_human = board.copy()
        board_after_human.push(chess.Move.from_uci("b1c3"))
        engine = FakeEngine({
            board.fen(): {"best_move": "e2e4", "best_cp": 30},
            board_after_human.fen(): {"best_move": "e7e5", "best_cp": 10},
        })
        cp_loss, best_uci, best_cp, human_cp = sv.evaluate_position_and_move(
            engine, board, chess.Move.from_uci("b1c3"), depth=8,
        )
        assert best_uci == "e2e4"
        assert best_cp == 30
        assert human_cp == -10
        assert cp_loss == 40
        assert len(engine.analyse_calls) == 2


class TestValidateJsonlEndToEnd:
    def _build_example(self, fen: str, target_uci: str, site: str = "https://test/1") -> dict:
        return {
            "text": _example_text(fen, target_uci),
            "metadata": {"site": site, "ply": 0},
        }

    def test_keeps_good_moves_rejects_bad(self, tmp_path: Path):
        fen1 = chess.STARTING_FEN
        fen2 = chess.STARTING_FEN
        custom_board = chess.Board()
        custom_board.push_san("e4")
        custom_board.push_san("e5")
        fen3 = custom_board.fen()

        ex1 = self._build_example(fen1, "e2e4", "https://test/keep_best")
        ex2 = self._build_example(fen2, "b1c3", "https://test/keep_small_loss")
        ex3 = self._build_example(fen3, "b1c3", "https://test/reject_big_loss")

        input_path = tmp_path / "in.jsonl"
        output_path = tmp_path / "out.jsonl"
        _write_jsonl(input_path, [ex1, ex2, ex3])

        board1 = chess.Board()
        board2_after = chess.Board()
        board2_after.push(chess.Move.from_uci("b1c3"))
        board3 = chess.Board(fen3)
        board3_after = board3.copy()
        board3_after.push(chess.Move.from_uci("b1c3"))

        engine = FakeEngine({
            board1.fen(): {"best_move": "e2e4", "best_cp": 30},
            board2_after.fen(): {"best_move": "e7e5", "best_cp": 10},
            board3.fen(): {"best_move": "g1f3", "best_cp": 50},
            board3_after.fen(): {"best_move": "g8f6", "best_cp": 150},
        })

        stats = sv.validate_jsonl(
            input_path, output_path, engine, depth=8, max_cp_loss=50,
            checkpoint_every=10, write_stats=False,
        )

        assert stats.examples_seen == 3
        assert stats.examples_kept == 2
        assert stats.rejected_cp_loss == 1
        assert stats.examples_rejected == 1

        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "sf_best_uci" in obj["metadata"]
            assert "sf_cp_loss" in obj["metadata"]
            assert obj["metadata"]["sf_cp_loss"] <= 50

    def test_keeps_human_move_never_replaces_it(self, tmp_path: Path):
        fen = chess.STARTING_FEN
        ex = self._build_example(fen, "b1c3")
        input_path = tmp_path / "in.jsonl"
        output_path = tmp_path / "out.jsonl"
        _write_jsonl(input_path, [ex])

        board = chess.Board()
        board_after = board.copy()
        board_after.push(chess.Move.from_uci("b1c3"))
        engine = FakeEngine({
            board.fen(): {"best_move": "e2e4", "best_cp": 30},
            board_after.fen(): {"best_move": "e7e5", "best_cp": 10},
        })

        stats = sv.validate_jsonl(
            input_path, output_path, engine, depth=8, max_cp_loss=50, write_stats=False,
        )

        assert stats.examples_kept == 1
        obj = json.loads(output_path.read_text(encoding="utf-8").strip())
        assert obj["text"] == ex["text"]
        assert obj["text"].endswith("b1c3")
        assert obj["metadata"]["sf_best_uci"] == "e2e4"
        assert obj["metadata"]["sf_cp_loss"] == 40

    def test_rejects_missing_fen_or_target(self, tmp_path: Path):
        items = [
            {"text": "no fen no target", "metadata": {}},
            {"text": "### Response:\ne2e4", "metadata": {}},
        ]
        input_path = tmp_path / "in.jsonl"
        output_path = tmp_path / "out.jsonl"
        _write_jsonl(input_path, items)

        engine = FakeEngine({})
        stats = sv.validate_jsonl(input_path, output_path, engine, write_stats=False)

        assert stats.examples_seen == 2
        assert stats.examples_kept == 0
        assert stats.rejected_missing_fen >= 1

    def test_rejects_illegal_target(self, tmp_path: Path):
        fen = chess.STARTING_FEN
        ex = self._build_example(fen, "e2e9")
        input_path = tmp_path / "in.jsonl"
        output_path = tmp_path / "out.jsonl"
        _write_jsonl(input_path, [ex])

        engine = FakeEngine({})
        stats = sv.validate_jsonl(input_path, output_path, engine, write_stats=False)

        assert stats.examples_kept == 0
        assert stats.rejected_illegal_target == 1

    def test_resume_appends_to_existing_output(self, tmp_path: Path):
        fen = chess.STARTING_FEN
        items = [self._build_example(fen, "e2e4", f"https://test/{i}") for i in range(4)]
        input_path = tmp_path / "in.jsonl"
        output_path = tmp_path / "out.jsonl"
        _write_jsonl(input_path, items)

        with output_path.open("w", encoding="utf-8") as f:
            f.write('{"text": "fake1"}\n')
            f.write('{"text": "fake2"}\n')

        board = chess.Board()
        engine = FakeEngine({board.fen(): {"best_move": "e2e4", "best_cp": 30}})

        stats = sv.validate_jsonl(
            input_path, output_path, engine, depth=8, max_cp_loss=50,
            write_stats=False,
        )

        assert stats.examples_seen == 4
        assert stats.skipped_resume == 2
        assert stats.examples_kept == 2

        lines = output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4
        assert json.loads(lines[0])["text"] == "fake1"
        assert "sf_best_uci" in json.loads(lines[2]).get("metadata", {})

    def test_limit_caps_new_examples(self, tmp_path: Path):
        fen = chess.STARTING_FEN
        items = [self._build_example(fen, "e2e4", f"https://test/{i}") for i in range(10)]
        input_path = tmp_path / "in.jsonl"
        output_path = tmp_path / "out.jsonl"
        _write_jsonl(input_path, items)

        board = chess.Board()
        engine = FakeEngine({board.fen(): {"best_move": "e2e4", "best_cp": 30}})

        stats = sv.validate_jsonl(
            input_path, output_path, engine, depth=8, max_cp_loss=50,
            limit=3, write_stats=False,
        )

        assert stats.examples_kept == 3


class TestValidationStats:
    def test_keep_rate_zero_when_none_seen(self):
        s = sv.ValidationStats()
        assert s.keep_rate == 0.0
        assert s.avg_cp_loss_kept == 0.0

    def test_keep_rate_computation(self):
        s = sv.ValidationStats(examples_seen=100, examples_kept=70)
        assert s.keep_rate == 0.70

    def test_keep_rate_excludes_resume_skipped(self):
        s = sv.ValidationStats(examples_seen=100, examples_kept=42, skipped_resume=30)
        assert s.keep_rate == pytest.approx(0.60, abs=0.001)

    def test_avg_cp_loss(self):
        s = sv.ValidationStats(examples_kept=10, sum_cp_loss_kept=150)
        assert s.avg_cp_loss_kept == 15.0

    def test_to_dict_rounds_floats(self):
        s = sv.ValidationStats(
            examples_seen=100, examples_kept=37, sum_cp_loss_kept=123,
        )
        d = s.to_dict()
        assert d["keep_rate"] == 0.37
        assert d["avg_cp_loss_kept"] == round(123 / 37, 2)


class TestWorkerClassification:
    def _worker(self, monkeypatch, engine, line, depth=8, max_cp_loss=50):
        monkeypatch.setattr(sv, "_worker_engine", engine)
        return sv._validate_one_line_worker((line, depth, max_cp_loss))

    def _line(self, fen: str, target_uci: str) -> str:
        return json.dumps({"text": _example_text(fen, target_uci),
                           "metadata": {"site": "https://test/1"}})

    def test_good_move_is_kept_with_metadata(self, monkeypatch):
        board = chess.Board()
        engine = FakeEngine({board.fen(): {"best_move": "e2e4", "best_cp": 30}})
        status, obj, cp_loss = self._worker(
            monkeypatch, engine, self._line(chess.STARTING_FEN, "e2e4"))
        assert status == "kept"
        assert cp_loss == 0
        assert obj["metadata"]["sf_best_uci"] == "e2e4"
        assert obj["metadata"]["sf_cp_loss"] == 0
        assert obj["metadata"]["sf_depth"] == 8

    def test_move_over_threshold_is_rejected(self, monkeypatch):
        board = chess.Board()
        after = board.copy()
        after.push(chess.Move.from_uci("b1c3"))
        engine = FakeEngine({
            board.fen(): {"best_move": "e2e4", "best_cp": 30},
            after.fen(): {"best_move": "e7e5", "best_cp": 150},
        })
        status, obj, cp_loss = self._worker(
            monkeypatch, engine, self._line(chess.STARTING_FEN, "b1c3"), max_cp_loss=50)
        assert status == "cp_loss"
        assert obj is None
        assert cp_loss == 180

    def test_malformed_json(self, monkeypatch):
        status, obj, _ = self._worker(monkeypatch, FakeEngine({}), "{not json")
        assert status == "invalid_json"
        assert obj is None

    def test_missing_fen(self, monkeypatch):
        line = json.dumps({"text": "### Response:\ne2e4"})
        status, _, _ = self._worker(monkeypatch, FakeEngine({}), line)
        assert status == "missing_fen"

    def test_illegal_target(self, monkeypatch):
        status, _, _ = self._worker(
            monkeypatch, FakeEngine({}), self._line(chess.STARTING_FEN, "e7e5"))
        assert status == "illegal_target"

    def test_agrees_with_sequential_path(self, monkeypatch, tmp_path: Path):
        board = chess.Board()
        after = board.copy()
        after.push(chess.Move.from_uci("b1c3"))
        results = {
            board.fen(): {"best_move": "e2e4", "best_cp": 30},
            after.fen(): {"best_move": "e7e5", "best_cp": 150},
        }
        lines = [
            self._line(chess.STARTING_FEN, "e2e4"),
            self._line(chess.STARTING_FEN, "b1c3"),
            "{not json",
            self._line(chess.STARTING_FEN, "e7e5"),
        ]

        in_path = tmp_path / "in.jsonl"
        in_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out_path = tmp_path / "out.jsonl"
        seq_stats = sv.validate_jsonl(
            in_path, out_path, FakeEngine(results),
            depth=8, max_cp_loss=50, write_stats=False,
        )

        worker_statuses = [
            self._worker(monkeypatch, FakeEngine(results), line)[0] for line in lines
        ]
        assert worker_statuses.count("kept") == seq_stats.examples_kept == 1
        assert worker_statuses.count("cp_loss") == seq_stats.rejected_cp_loss == 1
        assert worker_statuses.count("illegal_target") == seq_stats.rejected_illegal_target == 1
        assert worker_statuses.count("invalid_json") == seq_stats.rejected_invalid_fen == 1


class TestStockfishValidateCLI:
    def test_required_args(self):
        parser = sv._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self, tmp_path: Path):
        parser = sv._build_parser()
        args = parser.parse_args([
            "--input", str(tmp_path / "in.jsonl"),
            "--output", str(tmp_path / "out.jsonl"),
            "--stockfish", "/usr/games/stockfish",
        ])
        assert args.depth == 12
        assert args.max_cp_loss == 50
        assert args.threads == 1
        assert args.limit is None

    def test_overrides(self, tmp_path: Path):
        parser = sv._build_parser()
        args = parser.parse_args([
            "--input", str(tmp_path / "in.jsonl"),
            "--output", str(tmp_path / "out.jsonl"),
            "--stockfish", "/usr/games/stockfish",
            "--depth", "15",
            "--max-cp-loss", "30",
            "--threads", "4",
            "--limit", "1000",
            "--no-stats",
        ])
        assert args.depth == 15
        assert args.max_cp_loss == 30
        assert args.threads == 4
        assert args.limit == 1000
        assert args.no_stats is True
