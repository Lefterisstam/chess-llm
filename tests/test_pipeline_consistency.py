from __future__ import annotations

import filter_games as fg
import pgn_to_training as ptt
import run_pipeline as rp
import split_dataset as sd
import stockfish_validate as sv

from model.config.training_config import (
    BASE_ADAPTER_PATH,
    BEST_ADAPTER_PATH,
    BEST_HUB_REPO,
    TrainingConfig,
    make_continued_config,
    make_smoke_config,
)


class TestTrainingChainProducesDeployedModel:
    def test_default_run_writes_the_base_adapter(self):
        assert f"{TrainingConfig().output_dir}/final_adapter" == BASE_ADAPTER_PATH

    def test_continued_starts_from_the_base_adapter(self):
        assert make_continued_config().start_adapter_path == BASE_ADAPTER_PATH

    def test_continued_writes_the_deployed_adapter(self):
        assert f"{make_continued_config().output_dir}/final_adapter" == BEST_ADAPTER_PATH

    def test_continued_pushes_to_the_deployed_repo(self):
        assert make_continued_config().hub_model_id == BEST_HUB_REPO

    def test_continued_runs_a_full_epoch(self):
        cfg = make_continued_config()
        assert cfg.max_steps <= 0 and cfg.num_train_epochs >= 1

    def test_continued_does_not_stop_early(self):
        assert make_continued_config().early_stopping_patience > 10**6

    def test_smoke_cannot_touch_production_checkpoints(self):
        smoke = make_smoke_config()
        assert smoke.output_dir not in (TrainingConfig().output_dir,
                                        make_continued_config().output_dir)
        assert smoke.push_to_hub is False

    def test_base_and_v2_share_the_same_data(self):
        base, cont = TrainingConfig(), make_continued_config()
        assert base.train_jsonl == cont.train_jsonl
        assert base.val_jsonl == cont.val_jsonl


class TestInferenceDefaultsPointAtV2:
    def test_play_cli(self):
        from model.inference.play import _build_parser
        args = _build_parser().parse_args(["--fen", "8/8/8/4k3/8/8/8/4K3 w - - 0 1"])
        assert args.adapter == BEST_HUB_REPO
        assert args.mode == "stockfish-rerank"
        assert (args.stockfish_depth, args.num_candidates) == (6, 10)
        assert args.rerank_candidates == "exact"

    def test_web_app_cli(self):
        import inspect

        import web.app as webapp
        assert "default=BEST_HUB_REPO" in inspect.getsource(webapp.main)


class TestDataFiltersAgreeAcrossScripts:
    def test_elo(self):
        assert rp.DEFAULT_MIN_ELO == fg.DEFAULT_MIN_ELO == 2200

    def test_time_control(self):
        assert rp.DEFAULT_MIN_BASE_SECONDS == fg.DEFAULT_MIN_BASE_SECONDS == 480

    def test_game_length(self):
        assert (rp.DEFAULT_MIN_FULL_MOVES, rp.DEFAULT_MAX_FULL_MOVES) == (20, 150)
        assert (fg.DEFAULT_MIN_FULL_MOVES, fg.DEFAULT_MAX_FULL_MOVES) == (20, 150)

    def test_stockfish_validation(self):
        assert rp.DEFAULT_STOCKFISH_DEPTH == sv.DEFAULT_DEPTH == 12
        assert rp.DEFAULT_MAX_CP_LOSS == sv.DEFAULT_MAX_CP_LOSS == 50

    def test_split_percentages(self):
        assert (rp.DEFAULT_VAL_PCT, rp.DEFAULT_TEST_PCT) == (10, 10)
        assert (sd.DEFAULT_VAL_PCT, sd.DEFAULT_TEST_PCT) == (10, 10)


class TestPromptIsIdenticalInTrainingAndInference:
    def test_training_example_equals_prompt_plus_move(self):
        from model.inference.predict import format_prompt

        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
        history = ["e2e4", "e7e5"]
        training = ptt.format_training_example(fen, history, True, "g1f3")
        inference = format_prompt(
            fen, ptt.trim_history(history, ptt.DEFAULT_HISTORY_MOVES), "White")
        assert training == inference + "g1f3"

    def test_history_length_is_ten_plies_everywhere(self):
        import evaluation.game_simulation as gs
        import web.app as webapp
        assert ptt.DEFAULT_HISTORY_MOVES * 2 == 10
        assert webapp.HISTORY_PLIES == 10
        assert gs.HISTORY_PLIES == 10

    def test_no_history_placeholder_matches(self):
        import web.app as webapp
        assert ptt.trim_history([], 5) == webapp.NO_HISTORY


class TestRerankDefaultsHaveOneSourceOfTruth:
    def test_module_defaults(self):
        from model.inference.stockfish_rerank import (
            DEFAULT_CANDIDATE_SOURCE,
            DEFAULT_NUM_CANDIDATES,
            DEFAULT_STOCKFISH_DEPTH,
        )
        assert DEFAULT_CANDIDATE_SOURCE == "exact"
        assert (DEFAULT_STOCKFISH_DEPTH, DEFAULT_NUM_CANDIDATES) == (6, 10)

    def test_every_cli_exposes_the_same_candidate_default(self):
        import evaluation.game_simulation as gs
        import evaluation.run_game_simulation as rgs
        from model.inference.play import _build_parser as play_parser
        from model.inference.stockfish_rerank import DEFAULT_CANDIDATE_SOURCE

        cases = [
            gs._build_parser().parse_args(
                ["--agent-a", "random", "--agent-b", "random", "--output", "o.json"]),
            rgs._build_parser().parse_args(["--output-dir", "d"]),
        ]
        for args in cases:
            assert args.rerank_candidates == DEFAULT_CANDIDATE_SOURCE
            assert (args.rerank_depth, args.rerank_k) == (6, 10)

        play = play_parser().parse_args(["--fen", "8/8/8/4k3/8/8/8/4K3 w - - 0 1"])
        assert play.rerank_candidates == DEFAULT_CANDIDATE_SOURCE
