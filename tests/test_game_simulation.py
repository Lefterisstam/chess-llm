from __future__ import annotations

import json

import chess
import pytest

from evaluation.game_simulation import (
    Agent,
    MatchStats,
    RandomAgent,
    StockfishAgent,
    ZeroShotAgent,
    _build_parser,
    export_pgn,
    parse_agent_spec,
    play_game,
    play_match,
)


class ScriptedAgent(Agent):
    def __init__(self, name: str, white_script: list[str], black_script: list[str]):
        self.name = name
        self.white_script = white_script
        self.black_script = black_script

    def select_move(self, board: chess.Board, history_uci: list[str]) -> chess.Move:
        script = self.white_script if board.turn == chess.WHITE else self.black_script
        idx = len(history_uci) // 2
        return chess.Move.from_uci(script[idx])


def make_strong_agent():
    return ScriptedAgent(
        "strong",
        white_script=["e2e4", "d1h5", "f1c4", "h5f7"],
        black_script=["e7e5", "d8h4"],
    )


def make_weak_agent():
    return ScriptedAgent(
        "weak",
        white_script=["f2f3", "g2g4"],
        black_script=["a7a6", "b7b6", "c7c6", "d7d6"],
    )


class KnightShuffleAgent(Agent):
    def __init__(self, name="shuffler"):
        self.name = name

    def select_move(self, board: chess.Board, history_uci: list[str]) -> chess.Move:
        idx = (len(history_uci) // 2) % 2
        if board.turn == chess.WHITE:
            return chess.Move.from_uci(["b1c3", "c3b1"][idx])
        return chess.Move.from_uci(["b8c6", "c6b8"][idx])


class IllegalAgent(Agent):
    name = "illegal"

    def select_move(self, board: chess.Board, history_uci: list[str]) -> chess.Move:
        return chess.Move.from_uci("e2e5")


class TestPlayGame:
    def test_fools_mate_black_wins(self):
        white = make_weak_agent()
        black = make_strong_agent()
        result = play_game(white, black)
        assert result.result == "0-1"
        assert result.termination == "checkmate"
        assert result.n_plies == 4
        assert result.moves_uci == ["f2f3", "e7e5", "g2g4", "d8h4"]
        assert result.white_name == "weak"
        assert result.black_name == "strong"

    def test_scholars_mate_white_wins(self):
        white = make_strong_agent()
        black = make_weak_agent()
        result = play_game(white, black)
        assert result.result == "1-0"
        assert result.termination == "checkmate"
        assert result.n_plies == 7

    def test_max_plies_adjudicated_draw(self):
        result = play_game(KnightShuffleAgent("a"), KnightShuffleAgent("b"), max_plies=6)
        assert result.result == "1/2-1/2"
        assert result.termination == "max_plies"
        assert result.n_plies == 6

    def test_repetition_draw_without_cap(self):
        result = play_game(
            KnightShuffleAgent("a"), KnightShuffleAgent("b"), max_plies=100,
        )
        assert result.result == "1/2-1/2"
        assert result.termination == "fivefold_repetition"
        assert result.n_plies < 100

    def test_illegal_move_raises(self):
        with pytest.raises(ValueError, match="illegal move"):
            play_game(IllegalAgent(), make_strong_agent())

    def test_final_fen_recorded(self):
        result = play_game(make_weak_agent(), make_strong_agent())
        board = chess.Board(result.final_fen)
        assert board.is_checkmate()


class TestRandomAgent:
    def test_returns_legal_moves(self):
        agent = RandomAgent(seed=7)
        board = chess.Board()
        for _ in range(10):
            move = agent.select_move(board, [])
            assert move in board.legal_moves
            board.push(move)
            board.pop()

    def test_seeded_determinism(self):
        a1 = RandomAgent(seed=7)
        a2 = RandomAgent(seed=7)
        board = chess.Board()
        assert a1.select_move(board, []) == a2.select_move(board, [])

    def test_random_vs_random_terminates(self):
        result = play_game(RandomAgent(seed=1), RandomAgent(seed=2), max_plies=150)
        assert result.result in ("1-0", "0-1", "1/2-1/2")
        assert result.n_plies <= 150


class TestPlayMatch:
    def test_strong_sweeps_weak_with_alternation(self):
        stats = play_match(make_strong_agent(), make_weak_agent(), n_games=4)
        assert stats.n_games == 4
        assert stats.wins_a == 4
        assert stats.wins_b == 0
        assert stats.draws == 0
        assert stats.wins_a_as_white == 2
        assert stats.wins_a_as_black == 2
        assert stats.win_rate_a == 1.0
        assert stats.score_a == 1.0

    def test_colors_alternate(self):
        stats = play_match(make_strong_agent(), make_weak_agent(), n_games=4)
        whites = [g["white_name"] for g in stats.games]
        assert whites == ["strong", "weak", "strong", "weak"]

    def test_no_alternation_keeps_a_white(self):
        stats = play_match(
            make_strong_agent(), make_weak_agent(), n_games=3, alternate_colors=False,
        )
        assert all(g["white_name"] == "strong" for g in stats.games)

    def test_draws_counted(self):
        stats = play_match(
            KnightShuffleAgent("a"), KnightShuffleAgent("b"),
            n_games=2, max_plies=6,
        )
        assert stats.draws == 2
        assert stats.wins_a == stats.wins_b == 0
        assert stats.score_a == 0.5

    def test_accounting_sums(self):
        stats = play_match(make_strong_agent(), make_weak_agent(), n_games=5)
        assert stats.wins_a + stats.wins_b + stats.draws == stats.n_games == 5

    def test_to_dict_has_notebook_keys(self):
        stats = play_match(make_strong_agent(), make_weak_agent(), n_games=2)
        d = stats.to_dict()
        for key in ("agent_a", "agent_b", "n_games", "wins_a", "wins_b", "draws",
                    "win_rate_a", "score_a", "games"):
            assert key in d
        json.dumps(d)


class TestMatchStats:
    def test_empty_stats_safe(self):
        s = MatchStats()
        assert s.win_rate_a == 0.0
        assert s.score_a == 0.0

    def test_score_includes_half_for_draws(self):
        s = MatchStats(n_games=4, wins_a=1, wins_b=1, draws=2)
        assert s.score_a == (1 + 0.5 * 2) / 4

    def test_config_is_serialised_so_runs_stay_identifiable(self):
        s = MatchStats(n_games=2, config={
            "adapter": "/drive/.../mistral7b-qlora-chess-v2/final_adapter",
            "rerank_depth": 6, "rerank_k": 10, "rerank_candidates": "exact", "seed": 42,
        })
        d = s.to_dict()
        assert d["config"]["adapter"].endswith("mistral7b-qlora-chess-v2/final_adapter")
        assert d["config"]["rerank_depth"] == 6 and d["config"]["rerank_k"] == 10
        json.dumps(d)

    def test_config_defaults_to_empty_dict(self):
        assert MatchStats().to_dict()["config"] == {}


def _args(extra=None):
    argv = ["--agent-a", "random", "--agent-b", "random",
            "--output", "out.json"] + (extra or [])
    return _build_parser().parse_args(argv)


class TestParseAgentSpec:
    def test_random(self):
        agent = parse_agent_spec("random", _args(), {})
        assert isinstance(agent, RandomAgent)
        assert agent.name == "random"

    def test_stockfish_lazy_no_binary_needed(self):
        agent = parse_agent_spec("stockfish:3", _args(), {})
        assert isinstance(agent, StockfishAgent)
        assert agent.name == "stockfish-d3"
        assert agent.depth == 3
        assert agent._engine is None

    def test_stockfish_depth_over_10_rejected(self):
        with pytest.raises(ValueError, match="depth > 10"):
            parse_agent_spec("stockfish:15", _args(), {})

    def test_llm_without_adapter_rejected(self):
        with pytest.raises(ValueError, match="--adapter"):
            parse_agent_spec("llm:greedy", _args(), {})

    def test_zero_shot_lazy_construction(self):
        agent = parse_agent_spec("zero-shot:gpt2", _args(), {})
        assert isinstance(agent, ZeroShotAgent)
        assert agent.name == "zero-shot-gpt2"
        assert agent._model is None

    def test_unknown_spec_rejected(self):
        with pytest.raises(ValueError, match="Unknown agent spec"):
            parse_agent_spec("alphazero", _args(), {})


class TestExportPgn:
    def test_pgn_round_trip(self, tmp_path):
        import chess.pgn
        stats = play_match(make_strong_agent(), make_weak_agent(), n_games=2)
        pgn_path = tmp_path / "games.pgn"
        export_pgn(stats, pgn_path)

        assert pgn_path.exists()
        with pgn_path.open(encoding="utf-8") as f:
            game1 = chess.pgn.read_game(f)
            game2 = chess.pgn.read_game(f)
            game3 = chess.pgn.read_game(f)
        assert game1 is not None and game2 is not None
        assert game3 is None
        assert game1.headers["White"] == "strong"
        assert game2.headers["White"] == "weak"
        assert len(list(game1.mainline_moves())) == 7
        assert game1.headers["Result"] == "1-0"


class TestCliParser:
    def test_required_args(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

    def test_defaults(self):
        args = _args()
        assert args.games == 20
        assert args.max_plies == 300
        assert args.seed == 42
        assert args.rerank_depth == 6
        assert args.rerank_k == 10
        assert args.save_pgn is None

    def test_full_invocation(self):
        args = _build_parser().parse_args([
            "--agent-a", "llm:stockfish-rerank",
            "--agent-b", "stockfish:1",
            "--adapter", "/path/adapter",
            "--stockfish", "/usr/games/stockfish",
            "--games", "10",
            "--max-plies", "200",
            "--output", "game_sim_test.json",
            "--save-pgn", "games.pgn",
        ])
        assert args.agent_a == "llm:stockfish-rerank"
        assert args.games == 10
        assert str(args.save_pgn) == "games.pgn"
