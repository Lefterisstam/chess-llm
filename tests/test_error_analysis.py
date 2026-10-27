from __future__ import annotations

import chess

from evaluation.error_analysis import (
    ENDGAME_MATERIAL_MAX,
    OPENING_FULLMOVE_MAX,
    PHASES,
    PhaseAggregate,
    PhaseRecord,
    aggregate_by_phase,
    build_records,
    classify_phase,
    non_pawn_material,
)
from evaluation.stockfish_eval import CplPositionResult


class TestNonPawnMaterial:
    def test_starting_position(self):
        board = chess.Board()
        assert non_pawn_material(board, chess.WHITE) == 31
        assert non_pawn_material(board, chess.BLACK) == 31

    def test_bare_king(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        assert non_pawn_material(board, chess.WHITE) == 0
        assert non_pawn_material(board, chess.BLACK) == 0

    def test_pawns_excluded(self):
        board = chess.Board("4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1")
        assert non_pawn_material(board, chess.WHITE) == 0
        assert non_pawn_material(board, chess.BLACK) == 0

    def test_two_rooks_and_bishop(self):
        board = chess.Board("4k3/8/8/8/8/8/8/R3KB1R w KQ - 0 1")
        assert non_pawn_material(board, chess.WHITE) == 13


class TestClassifyPhase:
    def test_starting_position_is_opening(self):
        assert classify_phase(chess.Board()) == "opening"

    def test_full_material_late_is_middlegame(self):
        board = chess.Board()
        board.fullmove_number = OPENING_FULLMOVE_MAX + 5
        assert classify_phase(board) == "middlegame"

    def test_king_and_pawn_is_endgame(self):
        board = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 40")
        assert classify_phase(board) == "endgame"

    def test_endgame_checked_before_opening(self):
        board = chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 2")
        assert classify_phase(board) == "endgame"

    def test_material_boundary_inclusive(self):
        board = chess.Board("r3kb1r/8/8/8/8/8/8/R3KB1R w KQkq - 0 30")
        assert non_pawn_material(board, chess.WHITE) == ENDGAME_MATERIAL_MAX
        assert non_pawn_material(board, chess.BLACK) == ENDGAME_MATERIAL_MAX
        assert classify_phase(board) == "endgame"

    def test_one_side_heavy_is_not_endgame(self):
        board = chess.Board("4k3/8/8/8/8/8/8/R2QK2R w KQ - 0 30")
        assert non_pawn_material(board, chess.WHITE) == 19
        assert classify_phase(board) == "middlegame"


def _make_cpl_result(fen, gt, pred, cpl, matched_best=False):
    return CplPositionResult(
        fen=fen,
        ground_truth_uci=gt,
        predicted_uci=pred,
        best_move_uci=gt,
        eval_best_cp=0,
        eval_ours_cp=-cpl,
        cpl=cpl,
        matched_best=matched_best,
    )


class TestBuildRecords:
    def test_top1_flag(self):
        results = [
            _make_cpl_result(chess.STARTING_FEN, "e2e4", "e2e4", 0),
            _make_cpl_result(chess.STARTING_FEN, "e2e4", "d2d4", 30),
        ]
        records = build_records(results)
        assert records[0].top1 is True
        assert records[1].top1 is False

    def test_blunder_threshold(self):
        results = [
            _make_cpl_result(chess.STARTING_FEN, "e2e4", "a2a3", 150),
            _make_cpl_result(chess.STARTING_FEN, "e2e4", "a2a3", 250),
        ]
        records = build_records(results, blunder_cp=200)
        assert records[0].is_blunder is False
        assert records[1].is_blunder is True

    def test_phase_assigned_from_fen(self):
        results = [
            _make_cpl_result(chess.STARTING_FEN, "e2e4", "e2e4", 0),
            _make_cpl_result("8/8/4k3/8/8/4K3/4P3/8 w - - 0 40", "e2e4", "e2e4", 0),
        ]
        records = build_records(results)
        assert records[0].phase == "opening"
        assert records[1].phase == "endgame"

    def test_invalid_fen_skipped(self):
        results = [_make_cpl_result("not-a-fen", "e2e4", "e2e4", 0)]
        assert build_records(results) == []


class TestAggregate:
    def test_phase_aggregate_rates(self):
        agg = PhaseAggregate()
        agg.add(PhaseRecord("opening", top1=True, matched_best=True, cpl=0, is_blunder=False))
        agg.add(PhaseRecord("opening", top1=False, matched_best=False, cpl=300, is_blunder=True))
        assert agg.n == 2
        assert agg.top1_rate == 0.5
        assert agg.best_match_rate == 0.5
        assert agg.blunder_rate == 0.5
        assert agg.median_cpl == 150.0
        assert agg.bucket_counts["excellent"] == 1
        assert agg.bucket_counts["blunder"] == 1

    def test_aggregate_schema_stable(self):
        records = [
            PhaseRecord("opening", top1=True, matched_best=False, cpl=10, is_blunder=False),
        ]
        report = aggregate_by_phase(records)
        assert set(report["by_phase"].keys()) == set(PHASES)
        assert report["by_phase"]["middlegame"]["n"] == 0
        assert report["by_phase"]["endgame"]["n"] == 0
        assert report["overall"]["n"] == 1

    def test_overall_sums_phases(self):
        records = [
            PhaseRecord("opening", top1=True, matched_best=False, cpl=10, is_blunder=False),
            PhaseRecord("middlegame", top1=False, matched_best=False, cpl=20, is_blunder=False),
            PhaseRecord("endgame", top1=True, matched_best=True, cpl=300, is_blunder=True),
        ]
        report = aggregate_by_phase(records)
        assert report["overall"]["n"] == 3
        assert report["overall"]["top1_rate"] == round(2 / 3, 4)
        assert report["by_phase"]["endgame"]["blunder_rate"] == 1.0

    def test_empty_records(self):
        report = aggregate_by_phase([])
        assert report["overall"]["n"] == 0
        for phase in PHASES:
            assert report["by_phase"][phase]["n"] == 0
