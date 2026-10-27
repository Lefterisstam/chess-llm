from __future__ import annotations

import pytest

from evaluation.stockfish_eval import (
    BUCKET_ACCEPTABLE_MAX,
    BUCKET_EXCELLENT_MAX,
    BUCKET_GOOD_MAX,
    CplStats,
    classify_cpl,
    compute_cpl,
)


class TestComputeCpl:
    def test_zero_when_eval_equal(self):
        assert compute_cpl(eval_best_cp=30, eval_ours_cp=30) == 0

    def test_positive_diff_when_worse(self):
        assert compute_cpl(eval_best_cp=30, eval_ours_cp=10) == 20

    def test_clipped_at_zero_when_better(self):
        assert compute_cpl(eval_best_cp=20, eval_ours_cp=30) == 0

    def test_large_blunder(self):
        assert compute_cpl(eval_best_cp=100, eval_ours_cp=-500) == 600

    def test_negative_evals(self):
        assert compute_cpl(eval_best_cp=-100, eval_ours_cp=-300) == 200


class TestClassifyCpl:
    def test_excellent(self):
        assert classify_cpl(0) == "excellent"
        assert classify_cpl(5) == "excellent"
        assert classify_cpl(BUCKET_EXCELLENT_MAX) == "excellent"

    def test_good(self):
        assert classify_cpl(BUCKET_EXCELLENT_MAX + 1) == "good"
        assert classify_cpl(30) == "good"
        assert classify_cpl(BUCKET_GOOD_MAX) == "good"

    def test_acceptable(self):
        assert classify_cpl(BUCKET_GOOD_MAX + 1) == "acceptable"
        assert classify_cpl(100) == "acceptable"
        assert classify_cpl(BUCKET_ACCEPTABLE_MAX) == "acceptable"

    def test_blunder(self):
        assert classify_cpl(BUCKET_ACCEPTABLE_MAX + 1) == "blunder"
        assert classify_cpl(500) == "blunder"
        assert classify_cpl(10000) == "blunder"


class TestCplStatsEmpty:
    def test_zero_positions(self):
        s = CplStats()
        assert s.n_positions == 0
        assert s.mean_cpl == 0.0
        assert s.median_cpl == 0.0
        assert s.best_match_rate == 0.0
        assert s.percentile(50) == 0.0
        assert s.percentile(95) == 0.0
        assert s.bucket_counts == {
            "excellent": 0, "good": 0, "acceptable": 0, "blunder": 0,
        }


class TestCplStatsBasic:
    @pytest.fixture
    def stats_with_values(self):
        s = CplStats(
            n_positions=5,
            cpl_values=[0, 5, 20, 100, 500],
            n_matched_best=2,
        )
        return s

    def test_mean(self, stats_with_values):
        assert stats_with_values.mean_cpl == 125.0

    def test_median(self, stats_with_values):
        assert stats_with_values.median_cpl == 20.0

    def test_best_match_rate(self, stats_with_values):
        assert stats_with_values.best_match_rate == 0.40

    def test_bucket_counts(self, stats_with_values):
        assert stats_with_values.bucket_counts == {
            "excellent": 2, "good": 1, "acceptable": 1, "blunder": 1,
        }

    def test_median_even_length(self):
        s = CplStats(n_positions=4, cpl_values=[10, 20, 30, 40])
        assert s.median_cpl == 25.0

    def test_unsorted_input(self):
        s = CplStats(n_positions=4, cpl_values=[40, 10, 30, 20])
        assert s.mean_cpl == 25.0
        assert s.median_cpl == 25.0


class TestCplStatsPercentiles:
    @pytest.fixture
    def stats_100_values(self):
        s = CplStats(n_positions=100, cpl_values=list(range(100)))
        return s

    def test_p25(self, stats_100_values):
        assert stats_100_values.percentile(25) == pytest.approx(24.75)

    def test_p50_equals_median(self, stats_100_values):
        assert stats_100_values.percentile(50) == pytest.approx(49.5)

    def test_p75(self, stats_100_values):
        assert stats_100_values.percentile(75) == pytest.approx(74.25)

    def test_p95(self, stats_100_values):
        assert stats_100_values.percentile(95) == pytest.approx(94.05)

    def test_p100(self, stats_100_values):
        assert stats_100_values.percentile(100) == 99.0

    def test_p0(self, stats_100_values):
        assert stats_100_values.percentile(0) == 0.0


class TestCplStatsToDict:
    def test_to_dict_basic_fields(self):
        s = CplStats(
            n_positions=3,
            cpl_values=[0, 50, 100],
            n_matched_best=1,
            adapter_path="/test/adapter",
            test_file="/test/test.jsonl",
            stockfish_depth=12,
            mode="greedy",
            duration_seconds=42.5,
        )
        d = s.to_dict()
        assert d["n_positions"] == 3
        assert d["adapter_path"] == "/test/adapter"
        assert d["stockfish_depth"] == 12
        assert "mean_cpl" in d
        assert "median_cpl" in d
        assert "p25_cpl" in d
        assert "p75_cpl" in d
        assert "p95_cpl" in d
        assert "best_match_rate" in d
        assert "bucket_counts" in d

    def test_to_dict_computed_values(self):
        s = CplStats(
            n_positions=4,
            cpl_values=[0, 10, 50, 100],
            n_matched_best=2,
        )
        d = s.to_dict()
        assert d["mean_cpl"] == 40.0
        assert d["median_cpl"] == 30.0
        assert d["best_match_rate"] == 0.50

    def test_to_dict_buckets(self):
        s = CplStats(
            n_positions=4,
            cpl_values=[0, 30, 100, 500],
        )
        d = s.to_dict()
        assert d["bucket_counts"] == {
            "excellent": 1, "good": 1, "acceptable": 1, "blunder": 1,
        }


class TestRoundTripJson:
    def test_can_serialize_to_json(self):
        import json
        s = CplStats(
            n_positions=10,
            cpl_values=[0, 5, 10, 15, 20, 30, 50, 100, 200, 500],
            n_matched_best=3,
            adapter_path="/test",
            stockfish_depth=12,
            mode="exact",
        )
        d = s.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["n_positions"] == 10
        assert deserialized["best_match_rate"] == 0.30
        assert deserialized["bucket_counts"]["excellent"] == 3
