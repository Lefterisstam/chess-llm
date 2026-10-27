from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import chess
import chess.pgn
import download_lichess as dl
import filter_games as fg
import pgn_to_training as ptt
import pytest
import split_dataset as sd


class TestMonthToFilename:
    def test_valid_month(self):
        assert dl.filename_for_month("2018-01") == "lichess_db_standard_rated_2018-01.pgn.zst"

    def test_valid_month_recent(self):
        assert dl.filename_for_month("2025-12") == "lichess_db_standard_rated_2025-12.pgn.zst"

    @pytest.mark.parametrize("bad", [
        "2018-1",
        "2018/01",
        "18-01",
        "201801",
        "",
        "abcd-ef",
        "2018-01 ",
        " 2018-01",
    ])
    def test_invalid_month_raises(self, bad):
        with pytest.raises(ValueError):
            dl.filename_for_month(bad)

    def test_url_construction(self):
        url = dl.url_for_month("2018-01")
        assert url == "https://database.lichess.org/standard/lichess_db_standard_rated_2018-01.pgn.zst"


class TestParseChecksumIndex:
    def test_canonical_line(self):
        text = (
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  "
            "lichess_db_standard_rated_2018-01.pgn.zst"
        )
        result = dl.parse_checksum_index(text)
        assert result == {
            "lichess_db_standard_rated_2018-01.pgn.zst":
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        }

    def test_multiple_files(self):
        text = "\n".join([
            "aa" * 32 + "  file1.pgn.zst",
            "bb" * 32 + "  file2.pgn.zst",
        ])
        result = dl.parse_checksum_index(text)
        assert len(result) == 2
        assert result["file1.pgn.zst"] == "aa" * 32
        assert result["file2.pgn.zst"] == "bb" * 32

    def test_ignores_blank_and_comments(self):
        text = "\n".join([
            "",
            "# comment line",
            "  ",
            "aa" * 32 + "  real_file.pgn.zst",
            "# trailing comment",
        ])
        result = dl.parse_checksum_index(text)
        assert result == {"real_file.pgn.zst": "aa" * 32}

    def test_ignores_malformed_hash(self):
        text = "\n".join([
            "not-a-hash  file.pgn.zst",
            "ZZ" * 32 + "  bad_hex.pgn.zst",
            "aa" * 31 + "  short.pgn.zst",
            "aa" * 32 + "  ok.pgn.zst",
        ])
        result = dl.parse_checksum_index(text)
        assert result == {"ok.pgn.zst": "aa" * 32}

    def test_handles_uppercase_hash(self):
        text = "AA" * 32 + "  upper.pgn.zst"
        result = dl.parse_checksum_index(text)
        assert result == {"upper.pgn.zst": "aa" * 32}

    def test_empty_input(self):
        assert dl.parse_checksum_index("") == {}


class TestComputeSha256:
    def test_known_content(self, tmp_path: Path):
        payload = b"hello chess llm thesis"
        f = tmp_path / "sample.bin"
        f.write_bytes(payload)
        expected = hashlib.sha256(payload).hexdigest()
        assert dl.compute_sha256(f) == expected

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert dl.compute_sha256(f) == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )

    def test_large_file_chunked(self, tmp_path: Path):
        payload = b"A" * (5 * 1024 * 1024)
        f = tmp_path / "big.bin"
        f.write_bytes(payload)
        expected = hashlib.sha256(payload).hexdigest()
        assert dl.compute_sha256(f, chunk_size=1024 * 1024) == expected


class TestCLIParser:
    def test_requires_months_and_output(self):
        parser = dl._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_basic_invocation(self, tmp_path: Path):
        parser = dl._build_parser()
        args = parser.parse_args([
            "--months", "2018-01",
            "--output-dir", str(tmp_path),
        ])
        assert args.months == ["2018-01"]
        assert args.output_dir == tmp_path
        assert args.verify_only is False
        assert args.no_verify is False

    def test_parses_multiple_months(self, tmp_path: Path):
        parser = dl._build_parser()
        args = parser.parse_args([
            "--months", "2018-01", "2018-02", "2018-03",
            "--output-dir", str(tmp_path),
        ])
        assert args.months == ["2018-01", "2018-02", "2018-03"]

    def test_flags(self, tmp_path: Path):
        parser = dl._build_parser()
        args = parser.parse_args([
            "--months", "2018-01",
            "--output-dir", str(tmp_path),
            "--verify-only",
            "--no-verify",
            "--log-level", "DEBUG",
        ])
        assert args.verify_only is True
        assert args.no_verify is True
        assert args.log_level == "DEBUG"


class TestParseTimeControl:
    @pytest.mark.parametrize("tc, expected", [
        ("60+0",   (60, 0)),
        ("180+2",  (180, 2)),
        ("300+0",  (300, 0)),
        ("600+10", (600, 10)),
        ("5400+30",(5400, 30)),
        ("90",     (90, 0)),
        ("-",      (-1, 0)),
    ])
    def test_valid(self, tc, expected):
        assert fg.parse_time_control(tc) == expected

    @pytest.mark.parametrize("bad", [None, "", "abc", "60+", "+0", "60+x", "x+0"])
    def test_invalid(self, bad):
        assert fg.parse_time_control(bad) is None


class TestIsBelowMinTime:
    @pytest.mark.parametrize("tc, threshold, expected", [
        ("60+0", 180, True),
        ("179+0", 180, True),
        ("180+0", 180, False),
        ("300+0", 180, False),
        ("180+0", 480, True),
        ("300+0", 480, True),
        ("479+0", 480, True),
        ("480+0", 480, False),
        ("600+0", 480, False),
        ("1800+30", 480, False),
        ("600+0", 1500, True),
        ("1500+0", 1500, False),
    ])
    def test_threshold(self, tc, threshold, expected):
        assert fg.is_below_min_time(tc, threshold) is expected

    def test_correspondence_never_too_fast(self):
        assert fg.is_below_min_time("-", 180) is False
        assert fg.is_below_min_time("-", 480) is False
        assert fg.is_below_min_time("-", 5000) is False

    def test_unknown_never_too_fast(self):
        assert fg.is_below_min_time("garbage", 180) is False
        assert fg.is_below_min_time(None, 480) is False

    def test_pipeline_default_is_rapid_threshold(self):
        assert fg.DEFAULT_MIN_BASE_SECONDS == fg.LICHESS_RAPID_MIN_SECONDS == 480


def _headers(**kw) -> chess.pgn.Headers:
    h = chess.pgn.Headers()
    for k, v in kw.items():
        h[k] = str(v)
    return h


class TestEloExtraction:
    def test_valid(self):
        h = _headers(WhiteElo=2100, BlackElo=1850)
        assert fg.extract_elo(h, "White") == 2100
        assert fg.extract_elo(h, "Black") == 1850

    def test_missing(self):
        h = _headers()
        assert fg.extract_elo(h, "White") is None

    def test_question_mark(self):
        h = _headers(WhiteElo="?", BlackElo="?")
        assert fg.extract_elo(h, "White") is None
        assert fg.extract_elo(h, "Black") is None

    def test_malformed(self):
        h = _headers(WhiteElo="abc", BlackElo="??")
        assert fg.extract_elo(h, "White") is None
        assert fg.extract_elo(h, "Black") is None


class TestEloFilter:
    def test_both_pass(self):
        assert fg.passes_elo_filter(_headers(WhiteElo=2000, BlackElo=2000), 1800)

    def test_both_at_threshold(self):
        assert fg.passes_elo_filter(_headers(WhiteElo=1800, BlackElo=1800), 1800)

    def test_one_below(self):
        assert not fg.passes_elo_filter(_headers(WhiteElo=2200, BlackElo=1799), 1800)

    def test_both_below(self):
        assert not fg.passes_elo_filter(_headers(WhiteElo=1500, BlackElo=1600), 1800)

    def test_missing_elo(self):
        assert not fg.passes_elo_filter(_headers(WhiteElo=2000), 1800)
        assert not fg.passes_elo_filter(_headers(), 1800)


class TestTerminationFilter:
    @pytest.mark.parametrize("term, expected", [
        ("Normal", True),
        ("Time forfeit", False),
        ("Abandoned", False),
        ("Rules infraction", False),
        ("Unterminated", False),
        ("", False),
    ])
    def test_termination(self, term, expected):
        assert fg.passes_termination_filter(_headers(Termination=term)) is expected


class TestResultFilter:
    @pytest.mark.parametrize("res, expected", [
        ("1-0", True),
        ("0-1", True),
        ("1/2-1/2", True),
        ("*", False),
        ("", False),
        ("0.5-0.5", False),
    ])
    def test_result(self, res, expected):
        assert fg.passes_result_filter(_headers(Result=res)) is expected


class TestTimeControlFilter:
    def test_blitz_rejected_by_default(self):
        assert not fg.passes_timecontrol_filter(_headers(TimeControl="300+0"))

    def test_rapid_passes(self):
        assert fg.passes_timecontrol_filter(_headers(TimeControl="600+10"))

    def test_classical_passes(self):
        assert fg.passes_timecontrol_filter(_headers(TimeControl="5400+30"))

    def test_correspondence_passes(self):
        assert fg.passes_timecontrol_filter(_headers(TimeControl="-"))

    def test_bullet_rejected(self):
        assert not fg.passes_timecontrol_filter(_headers(TimeControl="60+0"))

    def test_missing_rejected(self):
        assert not fg.passes_timecontrol_filter(_headers())

    def test_malformed_rejected(self):
        assert not fg.passes_timecontrol_filter(_headers(TimeControl="garbage"))

    def test_blitz_passes_with_relaxed_threshold(self):
        assert fg.passes_timecontrol_filter(
            _headers(TimeControl="300+0"), min_base_seconds=180,
        )

    def test_blitz_rejected_when_rapid_threshold(self):
        assert not fg.passes_timecontrol_filter(
            _headers(TimeControl="300+0"), min_base_seconds=480,
        )

    def test_rapid_passes_at_rapid_threshold(self):
        assert fg.passes_timecontrol_filter(
            _headers(TimeControl="600+10"), min_base_seconds=480,
        )

    def test_classical_passes_at_rapid_threshold(self):
        assert fg.passes_timecontrol_filter(
            _headers(TimeControl="5400+30"), min_base_seconds=480,
        )

    def test_correspondence_passes_at_any_threshold(self):
        assert fg.passes_timecontrol_filter(
            _headers(TimeControl="-"), min_base_seconds=480,
        )
        assert fg.passes_timecontrol_filter(
            _headers(TimeControl="-"), min_base_seconds=5000,
        )


_MOVES_20 = (
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 "
    "8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 "
    "14. Ng3 g6 15. a4 c5 16. d5 c4 17. Bg5 h6 18. Be3 Nc5 19. Qd2 Bg7 "
    "20. Rad1 Qc7"
)

PGN_GAME_PASSES_ALL = f"""\
[Event "Test"]
[WhiteElo "2300"]
[BlackElo "2300"]
[TimeControl "600+0"]
[Termination "Normal"]
[Result "1-0"]

{_MOVES_20} 1-0
"""

PGN_TOO_SHORT = """\
[Event "Test"]
[WhiteElo "2300"]
[BlackElo "2300"]
[TimeControl "600+0"]
[Termination "Normal"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
"""

PGN_LOW_ELO = f"""\
[Event "Test"]
[WhiteElo "1500"]
[BlackElo "1500"]
[TimeControl "600+0"]
[Termination "Normal"]
[Result "1-0"]

{_MOVES_20} 1-0
"""

PGN_BULLET = f"""\
[Event "Test"]
[WhiteElo "2300"]
[BlackElo "2300"]
[TimeControl "60+0"]
[Termination "Normal"]
[Result "1-0"]

{_MOVES_20} 1-0
"""

PGN_BLITZ_5MIN = """\
[Event "Test"]
[WhiteElo "2300"]
[BlackElo "2300"]
[TimeControl "300+0"]
[Termination "Normal"]
[Result "1-0"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 9. Bxc4 Nd5 10. Bxe7 Qxe7 1-0
"""

PGN_RAPID_10MIN = """\
[Event "Test"]
[WhiteElo "2300"]
[BlackElo "2300"]
[TimeControl "600+0"]
[Termination "Normal"]
[Result "1-0"]

1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 9. Nd2 Nd7 10. b4 f5 1-0
"""

PGN_TIMEOUT = f"""\
[Event "Test"]
[WhiteElo "2300"]
[BlackElo "2300"]
[TimeControl "600+0"]
[Termination "Time forfeit"]
[Result "1-0"]

{_MOVES_20} 1-0
"""

PGN_UNFINISHED = f"""\
[Event "Test"]
[WhiteElo "2300"]
[BlackElo "2300"]
[TimeControl "600+0"]
[Termination "Normal"]
[Result "*"]

{_MOVES_20} *
"""

PGN_DUPLICATE_OF_FIRST = f"""\
[Event "Different event"]
[White "DifferentName"]
[WhiteElo "2400"]
[BlackElo "2350"]
[TimeControl "600+0"]
[Termination "Normal"]
[Result "1-0"]

{_MOVES_20} 1-0
"""


def _read_one_game(pgn_str: str) -> chess.pgn.Game:
    g = chess.pgn.read_game(io.StringIO(pgn_str))
    assert g is not None
    return g


class TestCountFullMoves:
    def test_standard_game(self):
        g = _read_one_game(PGN_GAME_PASSES_ALL)
        assert fg.count_full_moves(g) == 20

    def test_short_game(self):
        g = _read_one_game(PGN_TOO_SHORT)
        assert fg.count_full_moves(g) == 2

    def test_empty_game(self):
        g = chess.pgn.Game()
        assert fg.count_full_moves(g) == 0


class TestLengthFilter:
    def test_passes_at_min(self):
        g = _read_one_game(PGN_GAME_PASSES_ALL)
        assert fg.passes_length_filter(g, min_full=20, max_full=150)

    def test_too_short(self):
        g = _read_one_game(PGN_TOO_SHORT)
        assert not fg.passes_length_filter(g, min_full=20, max_full=150)

    def test_too_long(self):
        g = _read_one_game(PGN_GAME_PASSES_ALL)
        assert not fg.passes_length_filter(g, min_full=20, max_full=5)


class TestGameSignature:
    def test_same_moves_same_hash(self):
        g1 = _read_one_game(PGN_GAME_PASSES_ALL)
        g2 = _read_one_game(PGN_DUPLICATE_OF_FIRST)
        assert fg.game_signature(g1) == fg.game_signature(g2)

    def test_different_moves_different_hash(self):
        g1 = _read_one_game(PGN_GAME_PASSES_ALL)
        g2 = _read_one_game(PGN_TOO_SHORT)
        assert fg.game_signature(g1) != fg.game_signature(g2)

    def test_empty_game_hash_is_stable(self):
        sig = fg.game_signature(chess.pgn.Game())
        assert isinstance(sig, bytes) and len(sig) == 20


class TestFilterIntegration:
    def _run_filter(self, *pgns: str, **kwargs) -> tuple[fg.FilterStats, str]:
        in_stream = io.StringIO("\n".join(pgns))
        out_stream = io.StringIO()
        stats = fg.filter_pgn_stream(in_stream, out_stream, progress=False, **kwargs)
        return stats, out_stream.getvalue()

    def test_all_pass(self):
        stats, out = self._run_filter(PGN_GAME_PASSES_ALL)
        assert stats.total_seen == 1
        assert stats.kept == 1
        assert "1. e4 e5" in out

    def test_rejected_low_elo(self):
        stats, _ = self._run_filter(PGN_LOW_ELO)
        assert stats.kept == 0
        assert stats.rejected_elo == 1

    def test_rejected_bullet(self):
        stats, _ = self._run_filter(PGN_BULLET)
        assert stats.kept == 0
        assert stats.rejected_timecontrol == 1

    def test_rejected_timeout(self):
        stats, _ = self._run_filter(PGN_TIMEOUT)
        assert stats.kept == 0
        assert stats.rejected_termination == 1

    def test_rejected_unfinished(self):
        stats, _ = self._run_filter(PGN_UNFINISHED)
        assert stats.kept == 0
        assert stats.rejected_result == 1

    def test_rejected_too_short(self):
        stats, _ = self._run_filter(PGN_TOO_SHORT)
        assert stats.kept == 0
        assert stats.rejected_length == 1

    def test_duplicate_rejected(self):
        stats, out = self._run_filter(PGN_GAME_PASSES_ALL, PGN_DUPLICATE_OF_FIRST)
        assert stats.total_seen == 2
        assert stats.kept == 1
        assert stats.rejected_duplicate == 1
        assert out.count('1. e4 e5') == 1

    def test_mixed_batch(self):
        stats, _ = self._run_filter(
            PGN_GAME_PASSES_ALL,
            PGN_LOW_ELO,
            PGN_BULLET,
            PGN_TIMEOUT,
            PGN_UNFINISHED,
            PGN_TOO_SHORT,
            PGN_DUPLICATE_OF_FIRST,
        )
        assert stats.total_seen == 7
        assert stats.kept == 1
        assert stats.rejected_elo == 1
        assert stats.rejected_timecontrol == 1
        assert stats.rejected_termination == 1
        assert stats.rejected_result == 1
        assert stats.rejected_length == 1
        assert stats.rejected_duplicate == 1
        assert stats.yield_percent == pytest.approx(100.0 / 7, abs=0.01)

    def test_limit_respected(self):
        stats, _ = self._run_filter(
            PGN_GAME_PASSES_ALL,
            PGN_GAME_PASSES_ALL,
            PGN_GAME_PASSES_ALL,
            limit=2,
        )
        assert stats.total_seen == 2

    def test_default_rejects_blitz_keeps_rapid(self):
        stats, _ = self._run_filter(
            PGN_BLITZ_5MIN, PGN_RAPID_10MIN, min_full_moves=10,
        )
        assert stats.rejected_timecontrol == 1
        assert stats.kept == 1

    def test_relaxed_threshold_keeps_blitz_and_rapid(self):
        stats, _ = self._run_filter(
            PGN_BLITZ_5MIN, PGN_RAPID_10MIN,
            min_base_seconds=180, min_full_moves=10,
        )
        assert stats.kept == 2

    def test_min_base_seconds_propagates_to_stats(self):
        stats, _ = self._run_filter(PGN_RAPID_10MIN, min_base_seconds=480, min_full_moves=10)
        assert stats.min_base_seconds == 480

    def test_filter_pgn_file_writes_stats_sidecar(self, tmp_path: Path):
        input_path = tmp_path / "tiny.pgn"
        output_path = tmp_path / "filtered.pgn"
        input_path.write_text(PGN_GAME_PASSES_ALL + "\n" + PGN_LOW_ELO, encoding="utf-8")

        stats = fg.filter_pgn_file(input_path, output_path)

        assert stats.kept == 1
        assert output_path.exists()
        sidecar = output_path.with_suffix(output_path.suffix + ".stats.json")
        assert sidecar.exists()
        import json
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert loaded["kept"] == 1
        assert loaded["rejected_elo"] == 1


class TestPgnGameTextIterator:
    def test_single_game(self):
        texts = list(fg._iter_pgn_game_texts(io.StringIO(PGN_GAME_PASSES_ALL)))
        assert len(texts) == 1
        assert "[Event " in texts[0]
        assert texts[0].rstrip().endswith("1-0")

    def test_multiple_games_separated(self):
        combined = "\n".join([
            PGN_GAME_PASSES_ALL,
            PGN_LOW_ELO,
            PGN_GAME_PASSES_ALL,
        ])
        texts = list(fg._iter_pgn_game_texts(io.StringIO(combined)))
        assert len(texts) == 3
        for t in texts:
            assert t.startswith("[Event ")

    def test_each_game_self_contained(self):
        combined = PGN_GAME_PASSES_ALL + "\n" + PGN_LOW_ELO
        texts = list(fg._iter_pgn_game_texts(io.StringIO(combined)))
        for t in texts:
            g = chess.pgn.read_game(io.StringIO(t))
            assert g is not None
            assert "Event" in g.headers

    def test_unfinished_result_token(self):
        unfinished = PGN_GAME_PASSES_ALL.replace("1-0", "*")
        texts = list(fg._iter_pgn_game_texts(io.StringIO(unfinished + "\n" + PGN_GAME_PASSES_ALL)))
        assert len(texts) == 2


class TestFilterParallelEquivalence:
    def _run_both(self, pgns, **kwargs):
        combined = "\n".join(pgns)
        seq_out = io.StringIO()
        par_out = io.StringIO()
        seq_stats = fg.filter_pgn_stream(
            io.StringIO(combined), seq_out, progress=False, **kwargs
        )
        par_stats = fg.filter_pgn_stream_parallel(
            io.StringIO(combined), par_out, progress=False, workers=2, **kwargs
        )
        return seq_stats, seq_out.getvalue(), par_stats, par_out.getvalue()

    def test_stats_match(self):
        seq, _, par, _ = self._run_both([
            PGN_GAME_PASSES_ALL,
            PGN_LOW_ELO,
            PGN_BULLET,
            PGN_TIMEOUT,
            PGN_UNFINISHED,
            PGN_TOO_SHORT,
        ])
        assert seq.total_seen == par.total_seen
        assert seq.kept == par.kept
        assert seq.rejected_elo == par.rejected_elo
        assert seq.rejected_timecontrol == par.rejected_timecontrol
        assert seq.rejected_termination == par.rejected_termination
        assert seq.rejected_result == par.rejected_result
        assert seq.rejected_length == par.rejected_length

    def test_dedup_works_in_parallel(self):
        seq, _, par, _ = self._run_both([
            PGN_GAME_PASSES_ALL,
            PGN_DUPLICATE_OF_FIRST,
        ])
        assert seq.kept == par.kept == 1
        assert seq.rejected_duplicate == par.rejected_duplicate == 1

    def test_kept_games_match_as_set(self):
        pgns = [PGN_GAME_PASSES_ALL] * 1 + [PGN_LOW_ELO, PGN_BULLET]
        _, seq_text, _, par_text = self._run_both(pgns)
        assert seq_text.count("1. e4 e5") == par_text.count("1. e4 e5") == 1


class TestFilterCLIParser:
    def test_defaults(self, tmp_path: Path):
        parser = fg._build_parser()
        args = parser.parse_args([
            "--input", str(tmp_path / "in.pgn.zst"),
            "--output", str(tmp_path / "out.pgn"),
        ])
        assert args.min_elo == 2200
        assert args.min_base_seconds == 480
        assert args.min_full_moves == 20
        assert args.max_full_moves == 150
        assert args.limit is None
        assert args.no_stats is False

    def test_overrides(self, tmp_path: Path):
        parser = fg._build_parser()
        args = parser.parse_args([
            "--input", str(tmp_path / "in.pgn.zst"),
            "--output", str(tmp_path / "out.pgn"),
            "--min-elo", "2000",
            "--min-base-seconds", "480",
            "--min-moves", "20",
            "--max-moves", "150",
            "--limit", "10000",
            "--no-stats",
        ])
        assert args.min_elo == 2000
        assert args.min_base_seconds == 480
        assert args.min_full_moves == 20
        assert args.max_full_moves == 150
        assert args.limit == 10000
        assert args.no_stats is True


import random as _random_module


class TestSamplePlyIndices:
    def test_normal_sample(self):
        rng = _random_module.Random(42)
        idx = ptt.sample_ply_indices(total_ply=40, n_samples=5, skip_opening_ply=6, rng=rng)
        assert len(idx) == 5
        assert all(6 <= i <= 39 for i in idx)
        assert idx == sorted(idx)
        assert len(set(idx)) == 5

    def test_fewer_candidates_than_requested(self):
        rng = _random_module.Random(42)
        idx = ptt.sample_ply_indices(total_ply=10, n_samples=20, skip_opening_ply=6, rng=rng)
        assert idx == [6, 7, 8, 9]

    def test_all_skipped(self):
        rng = _random_module.Random(42)
        idx = ptt.sample_ply_indices(total_ply=5, n_samples=10, skip_opening_ply=6, rng=rng)
        assert idx == []

    def test_zero_skip(self):
        rng = _random_module.Random(42)
        idx = ptt.sample_ply_indices(total_ply=10, n_samples=3, skip_opening_ply=0, rng=rng)
        assert len(idx) == 3
        assert all(0 <= i <= 9 for i in idx)

    def test_deterministic_with_same_seed(self):
        rng1 = _random_module.Random(42)
        rng2 = _random_module.Random(42)
        idx1 = ptt.sample_ply_indices(40, 5, 6, rng1)
        idx2 = ptt.sample_ply_indices(40, 5, 6, rng2)
        assert idx1 == idx2


class TestPositionAtPly:
    def _ruy_lopez_moves(self) -> list[chess.Move]:
        board = chess.Board()
        moves = []
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]:
            mv = chess.Move.from_uci(uci)
            assert mv in board.legal_moves
            moves.append(mv)
            board.push(mv)
        return moves

    def test_starting_position(self):
        moves = self._ruy_lopez_moves()
        fen, target, side_w, hist = ptt.position_at_ply(moves, 0)
        assert fen == chess.STARTING_FEN
        assert target == "e2e4"
        assert side_w is True
        assert hist == []

    def test_after_first_move(self):
        moves = self._ruy_lopez_moves()
        fen, target, side_w, hist = ptt.position_at_ply(moves, 1)
        assert "4P3" in fen
        assert " b " in fen
        assert target == "e7e5"
        assert side_w is False
        assert hist == ["e2e4"]

    def test_mid_sequence(self):
        moves = self._ruy_lopez_moves()
        fen, target, side_w, hist = ptt.position_at_ply(moves, 4)
        assert target == "f1b5"
        assert side_w is True
        assert hist == ["e2e4", "e7e5", "g1f3", "b8c6"]


class TestTrimHistory:
    def test_empty_history(self):
        assert ptt.trim_history([], 5) == "(none — opening)"

    def test_less_than_window(self):
        hist = ["e2e4", "e7e5"]
        assert ptt.trim_history(hist, 5) == "e2e4 e7e5"

    def test_exactly_window(self):
        hist = ["e2e4", "e7e5", "g1f3", "b8c6"]
        assert ptt.trim_history(hist, 2) == "e2e4 e7e5 g1f3 b8c6"

    def test_more_than_window_takes_tail(self):
        hist = ["a1", "b1", "c1", "d1", "e1", "f1"]
        assert ptt.trim_history(hist, 2) == "c1 d1 e1 f1"


class TestFormatTrainingExample:
    def test_format_exact(self):
        text = ptt.format_training_example(
            fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            history_uci=["e2e4", "e7e5"],
            side_white=True,
            target_uci="g1f3",
            history_full_moves=5,
        )
        assert "### Instruction:" in text
        assert "### Response:" in text
        assert "Position (FEN): rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2" in text
        assert "Move history (last 5 moves): e2e4 e7e5" in text
        assert "Side to move: White" in text
        assert text.endswith("g1f3")

    def test_black_to_move(self):
        text = ptt.format_training_example(
            fen=chess.STARTING_FEN, history_uci=["e2e4"], side_white=False,
            target_uci="e7e5",
        )
        assert "Side to move: Black" in text

    def test_opening_position_history(self):
        text = ptt.format_training_example(
            fen=chess.STARTING_FEN, history_uci=[], side_white=True, target_uci="e2e4",
        )
        assert "(none — opening)" in text


class TestSafeInt:
    @pytest.mark.parametrize("s, expected", [
        ("2000", 2000),
        ("0", 0),
        ("?", None),
        ("", None),
        (None, None),
        ("abc", None),
    ])
    def test_parsing(self, s, expected):
        assert ptt._safe_int(s) == expected


PGN_GOOD = """\
[Event "Test"]
[WhiteElo "2100"]
[BlackElo "2050"]
[Result "1-0"]
[TimeControl "600+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0
"""

PGN_TOO_SHORT_FOR_EXTRACT = """\
[Event "Test"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "2000"]

1. e4 e5 2. Nf3 Nc6 1-0
"""


def _read_one(pgn_str: str) -> chess.pgn.Game:
    g = chess.pgn.read_game(io.StringIO(pgn_str))
    assert g is not None
    return g


class TestExtractExamplesFromGame:
    def test_normal_game_produces_examples(self):
        rng = _random_module.Random(42)
        examples = list(ptt.extract_examples_from_game(
            _read_one(PGN_GOOD),
            n_positions=5, history_full_moves=2, skip_opening_ply=6, rng=rng,
        ))
        assert len(examples) == 5
        for ex in examples:
            assert "text" in ex and "metadata" in ex
            assert ex["metadata"]["white_elo"] == 2100
            assert ex["metadata"]["black_elo"] == 2050
            assert ex["metadata"]["result"] == "1-0"
            assert ex["metadata"]["side_to_move"] in ("white", "black")
            assert "### Instruction:" in ex["text"]
            assert "### Response:" in ex["text"]

    def test_short_game_produces_none(self):
        rng = _random_module.Random(42)
        examples = list(ptt.extract_examples_from_game(
            _read_one(PGN_TOO_SHORT_FOR_EXTRACT),
            n_positions=5, history_full_moves=2, skip_opening_ply=6, rng=rng,
        ))
        assert examples == []

    def test_examples_capped_by_available_ply(self):
        rng = _random_module.Random(42)
        examples = list(ptt.extract_examples_from_game(
            _read_one(PGN_GOOD),
            n_positions=100, history_full_moves=2, skip_opening_ply=6, rng=rng,
        ))
        assert len(examples) == 14

    def test_deterministic_with_seed(self):
        rng1 = _random_module.Random(42)
        rng2 = _random_module.Random(42)
        ex1 = list(ptt.extract_examples_from_game(_read_one(PGN_GOOD), 3, 2, 6, rng1))
        ex2 = list(ptt.extract_examples_from_game(_read_one(PGN_GOOD), 3, 2, 6, rng2))
        assert [e["text"] for e in ex1] == [e["text"] for e in ex2]


class TestConvertPgnFilesEndToEnd:
    def test_writes_jsonl_and_stats(self, tmp_path: Path):
        input_pgn = tmp_path / "input.pgn"
        input_pgn.write_text(PGN_GOOD + "\n" + PGN_GOOD + "\n" + PGN_TOO_SHORT_FOR_EXTRACT,
                             encoding="utf-8")
        output_jsonl = tmp_path / "out.jsonl"

        stats = ptt.convert_pgn_files(
            [input_pgn], output_jsonl,
            positions_per_game=4, history_full_moves=2, skip_opening_ply=6, seed=42,
        )
        assert stats.games_seen == 3
        assert stats.games_used == 2
        assert stats.games_too_short == 1
        assert stats.examples_written == 8
        assert output_jsonl.exists()

        lines = output_jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 8
        for line in lines:
            obj = json.loads(line)
            assert set(obj.keys()) == {"text", "metadata"}
            assert "### Instruction:" in obj["text"]
            assert "### Response:" in obj["text"]
            assert "ply" in obj["metadata"]
            assert "side_to_move" in obj["metadata"]

        assert stats.side_to_move_white + stats.side_to_move_black == stats.examples_written

        sidecar = output_jsonl.with_suffix(output_jsonl.suffix + ".stats.json")
        assert sidecar.exists()
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert loaded["examples_written"] == 8

    def test_max_examples_respected(self, tmp_path: Path):
        input_pgn = tmp_path / "input.pgn"
        input_pgn.write_text((PGN_GOOD + "\n") * 5, encoding="utf-8")
        output_jsonl = tmp_path / "out.jsonl"

        stats = ptt.convert_pgn_files(
            [input_pgn], output_jsonl,
            positions_per_game=10, history_full_moves=2, skip_opening_ply=6, seed=42,
            max_examples=12,
        )
        assert stats.examples_written == 12

    def test_multiple_input_files_combined(self, tmp_path: Path):
        f1 = tmp_path / "a.pgn"
        f2 = tmp_path / "b.pgn"
        f1.write_text(PGN_GOOD, encoding="utf-8")
        f2.write_text(PGN_GOOD, encoding="utf-8")
        output_jsonl = tmp_path / "merged.jsonl"

        stats = ptt.convert_pgn_files(
            [f1, f2], output_jsonl,
            positions_per_game=3, history_full_moves=2, skip_opening_ply=6, seed=42,
        )
        assert stats.games_seen == 2
        assert stats.examples_written == 6


class TestPgnToTrainingCLI:
    def test_defaults(self, tmp_path: Path):
        parser = ptt._build_parser()
        args = parser.parse_args([
            "--inputs", str(tmp_path / "a.pgn"),
            "--output", str(tmp_path / "out.jsonl"),
        ])
        assert args.positions_per_game == 10
        assert args.history_moves == 5
        assert args.skip_opening_ply == 6
        assert args.seed == 42
        assert args.max_examples is None

    def test_multiple_inputs(self, tmp_path: Path):
        parser = ptt._build_parser()
        args = parser.parse_args([
            "--inputs", str(tmp_path / "a.pgn"), str(tmp_path / "b.pgn"),
            "--output", str(tmp_path / "out.jsonl"),
            "--positions-per-game", "5",
            "--history-moves", "3",
            "--skip-opening-ply", "10",
            "--seed", "7",
            "--max-examples", "1000",
        ])
        assert len(args.inputs) == 2
        assert args.positions_per_game == 5
        assert args.history_moves == 3
        assert args.skip_opening_ply == 10
        assert args.seed == 7
        assert args.max_examples == 1000


class TestAssignSplit:
    def test_deterministic(self):
        s1 = sd.assign_split("https://lichess.org/abc123", seed=42)
        s2 = sd.assign_split("https://lichess.org/abc123", seed=42)
        assert s1 == s2

    def test_different_ids_can_get_different_splits(self):
        splits = {sd.assign_split(f"https://lichess.org/{i:05d}", seed=42) for i in range(500)}
        assert splits == {"train", "val", "test"}

    def test_seed_changes_assignment(self):
        sites = [f"site_{i}" for i in range(100)]
        s1 = [sd.assign_split(s, seed=1) for s in sites]
        s2 = [sd.assign_split(s, seed=999) for s in sites]
        diffs = sum(1 for a, b in zip(s1, s2) if a != b)
        assert diffs > 20, f"only {diffs}/100 changed, the seed has no effect"

    def test_approximate_distribution(self):
        counts = {"train": 0, "val": 0, "test": 0}
        for i in range(10_000):
            counts[sd.assign_split(f"id_{i}", seed=42)] += 1
        assert 0.78 <= counts["train"] / 10_000 <= 0.82
        assert 0.08 <= counts["val"]   / 10_000 <= 0.12
        assert 0.08 <= counts["test"]  / 10_000 <= 0.12

    def test_custom_percentages(self):
        counts = {"train": 0, "val": 0, "test": 0}
        for i in range(5_000):
            counts[sd.assign_split(f"id_{i}", seed=42, val_pct=15, test_pct=15)] += 1
        assert 0.67 <= counts["train"] / 5_000 <= 0.73
        assert 0.13 <= counts["val"]   / 5_000 <= 0.17
        assert 0.13 <= counts["test"]  / 5_000 <= 0.17

    @pytest.mark.parametrize("val, test", [
        (-1, 10), (10, -1), (50, 50), (60, 50),
    ])
    def test_invalid_percentages(self, val, test):
        with pytest.raises(ValueError):
            sd.assign_split("any_id", val_pct=val, test_pct=test)


class TestExtractGameId:
    def test_valid_site(self):
        assert sd.extract_game_id({"site": "https://lichess.org/abc"}, 0) == "https://lichess.org/abc"

    def test_missing_site_fallback(self):
        gid1 = sd.extract_game_id({}, 5)
        gid2 = sd.extract_game_id({"site": ""}, 7)
        gid3 = sd.extract_game_id({"site": "?"}, 11)
        assert gid1.startswith("__missing_site_")
        assert gid2.startswith("__missing_site_")
        assert gid3.startswith("__missing_site_")
        assert gid1 != gid2 != gid3

    def test_none_metadata(self):
        assert sd.extract_game_id(None, 0).startswith("__missing_site_")


def _make_example(site: str, ply: int, side: str = "white") -> dict:
    return {
        "text": f"DUMMY TEXT for {site} ply {ply}",
        "metadata": {
            "site": site,
            "ply": ply,
            "side_to_move": side,
            "white_elo": 2000,
            "black_elo": 2000,
            "result": "1-0",
        },
    }


class TestSplitJsonlFile:
    def _write_jsonl(self, path: Path, games: dict[str, int]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for site, n in games.items():
                for ply in range(n):
                    f.write(json.dumps(_make_example(site, ply)) + "\n")

    def test_no_data_leakage(self, tmp_path: Path):
        games = {f"https://lichess.org/game{i:03d}": 10 for i in range(100)}
        in_path = tmp_path / "in.jsonl"
        self._write_jsonl(in_path, games)

        stats = sd.split_jsonl_file(in_path, tmp_path, val_pct=10, test_pct=10, seed=42)

        sites_per_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        for name in ("train", "val", "test"):
            for line in (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
                obj = json.loads(line)
                sites_per_split[name].add(obj["metadata"]["site"])

        assert sites_per_split["train"].isdisjoint(sites_per_split["val"])
        assert sites_per_split["train"].isdisjoint(sites_per_split["test"])
        assert sites_per_split["val"].isdisjoint(sites_per_split["test"])

        assert len(sites_per_split["train"]) + len(sites_per_split["val"]) + len(sites_per_split["test"]) == 100
        assert stats.games_total == 100

    def test_counts_sum_to_input(self, tmp_path: Path):
        games = {f"site_{i}": 10 for i in range(200)}
        in_path = tmp_path / "in.jsonl"
        self._write_jsonl(in_path, games)
        stats = sd.split_jsonl_file(in_path, tmp_path, val_pct=10, test_pct=10, seed=42)
        assert stats.examples_train + stats.examples_val + stats.examples_test == stats.examples_total == 2000

    def test_approximate_split_pct(self, tmp_path: Path):
        games = {f"site_{i}": 5 for i in range(1000)}
        in_path = tmp_path / "in.jsonl"
        self._write_jsonl(in_path, games)
        stats = sd.split_jsonl_file(in_path, tmp_path, val_pct=10, test_pct=10, seed=42)
        assert 0.76 <= stats.games_train / 1000 <= 0.84
        assert 0.07 <= stats.games_val   / 1000 <= 0.13
        assert 0.07 <= stats.games_test  / 1000 <= 0.13

    def test_examples_from_same_game_stay_together(self, tmp_path: Path):
        games = {f"https://lichess.org/abc{i}": 7 for i in range(50)}
        in_path = tmp_path / "in.jsonl"
        self._write_jsonl(in_path, games)
        sd.split_jsonl_file(in_path, tmp_path, val_pct=10, test_pct=10, seed=42)

        site_to_split: dict[str, str] = {}
        for name in ("train", "val", "test"):
            for line in (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
                obj = json.loads(line)
                site = obj["metadata"]["site"]
                if site in site_to_split:
                    assert site_to_split[site] == name, f"{site} is in both {site_to_split[site]} and {name}"
                site_to_split[site] = name

    def test_stats_sidecar_written(self, tmp_path: Path):
        games = {f"site_{i}": 3 for i in range(30)}
        in_path = tmp_path / "in.jsonl"
        self._write_jsonl(in_path, games)
        sd.split_jsonl_file(in_path, tmp_path, val_pct=10, test_pct=10, seed=42)

        sidecar = tmp_path / "split.stats.json"
        assert sidecar.exists()
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        assert loaded["examples_total"] == 90
        assert loaded["games_total"] == 30
        assert loaded["seed"] == 42

    def test_missing_site_handled_separately(self, tmp_path: Path):
        in_path = tmp_path / "in.jsonl"
        with in_path.open("w", encoding="utf-8") as f:
            for i in range(50):
                f.write(json.dumps({"text": f"x{i}", "metadata": {"site": "?", "ply": 0}}) + "\n")
        stats = sd.split_jsonl_file(in_path, tmp_path, val_pct=10, test_pct=10, seed=42)
        assert stats.missing_site_count == 50
        assert stats.examples_train + stats.examples_val + stats.examples_test == 50

    def test_seed_changes_assignment(self, tmp_path: Path):
        games = {f"site_{i}": 1 for i in range(100)}
        in_path = tmp_path / "in.jsonl"
        self._write_jsonl(in_path, games)

        out1 = tmp_path / "s1"
        out1.mkdir()
        sd.split_jsonl_file(in_path, out1, val_pct=10, test_pct=10, seed=1)
        train_s1 = {json.loads(line)["metadata"]["site"]
                    for line in (out1 / "train.jsonl").read_text().splitlines()}

        out2 = tmp_path / "s2"
        out2.mkdir()
        sd.split_jsonl_file(in_path, out2, val_pct=10, test_pct=10, seed=999)
        train_s2 = {json.loads(line)["metadata"]["site"]
                    for line in (out2 / "train.jsonl").read_text().splitlines()}

        assert len(train_s1 ^ train_s2) > 5


class TestSplitCLIParser:
    def test_defaults(self, tmp_path: Path):
        parser = sd._build_parser()
        args = parser.parse_args([
            "--input", str(tmp_path / "in.jsonl"),
            "--output-dir", str(tmp_path),
        ])
        assert args.val_pct == 10
        assert args.test_pct == 10
        assert args.seed == 42

    def test_overrides(self, tmp_path: Path):
        parser = sd._build_parser()
        args = parser.parse_args([
            "--input", str(tmp_path / "in.jsonl"),
            "--output-dir", str(tmp_path),
            "--val-pct", "15",
            "--test-pct", "20",
            "--seed", "7",
            "--no-stats",
        ])
        assert args.val_pct == 15
        assert args.test_pct == 20
        assert args.seed == 7
        assert args.no_stats is True
