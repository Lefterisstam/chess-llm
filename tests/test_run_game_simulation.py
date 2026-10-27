from __future__ import annotations

from evaluation.run_game_simulation import (
    DEFAULT_OPPONENTS,
    build_matchup_argv,
    format_table,
    output_path_for,
    spec_tag,
    summarize,
)


class TestSpecTag:
    def test_zero_shot_model_id_made_safe(self):
        assert spec_tag("zero-shot:mistralai/Mistral-7B-v0.1") == "zeroshot-mistralai-Mistral-7B-v0.1"

    def test_stockfish_depth(self):
        assert spec_tag("stockfish:1") == "stockfish-1"

    def test_llm_mode(self):
        assert spec_tag("llm:stockfish-rerank") == "llm-stockfish-rerank"

    def test_no_slashes_or_colons(self):
        for opp in DEFAULT_OPPONENTS:
            tag = spec_tag(opp)
            assert "/" not in tag and ":" not in tag


class TestOutputPath:
    def test_filename_shape(self, tmp_path):
        p = output_path_for(tmp_path, "llm:stockfish-rerank", "random")
        assert p.name == "game_sim_llm-stockfish-rerank_vs_random.json"
        assert p.parent == tmp_path


class TestBuildMatchupArgv:
    def test_core_flags_present(self, tmp_path):
        out = tmp_path / "x.json"
        argv = build_matchup_argv(
            "llm:stockfish-rerank", "random",
            adapter="/a/final_adapter", stockfish="/usr/games/stockfish",
            games=30, max_plies=200, rerank_depth=6, rerank_k=10, seed=42, output=out,
        )
        assert argv[argv.index("--agent-a") + 1] == "llm:stockfish-rerank"
        assert argv[argv.index("--agent-b") + 1] == "random"
        assert argv[argv.index("--games") + 1] == "30"
        assert argv[argv.index("--output") + 1] == str(out)
        assert argv[argv.index("--adapter") + 1] == "/a/final_adapter"

    def test_adapter_omitted_when_none(self, tmp_path):
        argv = build_matchup_argv(
            "random", "random",
            adapter=None, stockfish="/usr/games/stockfish",
            games=10, max_plies=200, rerank_depth=6, rerank_k=10, seed=1,
            output=tmp_path / "x.json",
        )
        assert "--adapter" not in argv


class TestSummarize:
    def test_maps_stats_dict(self):
        results = [("random", {
            "n_games": 20, "wins_a": 18, "draws": 2, "wins_b": 0,
            "win_rate_a": 0.9, "score_a": 0.95,
        })]
        rows = summarize(results)
        assert rows[0] == {
            "opponent": "random", "n_games": 20,
            "wins": 18, "draws": 2, "losses": 0,
            "win_rate": 0.9, "score": 0.95,
        }

    def test_missing_keys_default_zero(self):
        rows = summarize([("sf", {"error": "OOM"})])
        assert rows[0]["wins"] == 0 and rows[0]["n_games"] == 0

    def test_format_table_has_rows(self):
        rows = summarize([("random", {
            "n_games": 20, "wins_a": 18, "draws": 2, "wins_b": 0,
            "win_rate_a": 0.9, "score_a": 0.95,
        })])
        table = format_table("llm:stockfish-rerank", rows)
        assert "random" in table
        assert "llm:stockfish-rerank" in table
