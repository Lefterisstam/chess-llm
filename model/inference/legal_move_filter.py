from __future__ import annotations

import re
from dataclasses import dataclass

import chess

UCI_PATTERN = re.compile(r"^([a-h][1-8])([a-h][1-8])([qrbn])?$")


@dataclass
class FilterResult:
    raw: str
    filtered: chess.Move
    was_legal: bool
    was_parseable: bool
    fallback_strategy: str


def extract_uci_candidate(text: str) -> str:
    tokens = (text or "").split()
    if not tokens:
        return ""
    return tokens[0][:5].lower().rstrip(".,;:!?")


def try_parse_uci(uci_str: str) -> chess.Move | None:
    if not uci_str:
        return None
    if not UCI_PATTERN.match(uci_str):
        if len(uci_str) >= 4 and UCI_PATTERN.match(uci_str[:4]):
            uci_str = uci_str[:4]
        else:
            return None
    try:
        return chess.Move.from_uci(uci_str)
    except ValueError:
        return None


def same_line(a: chess.Square, b: chess.Square) -> bool:
    return chess.square_rank(a) == chess.square_rank(b) or chess.square_file(a) == chess.square_file(b)


def closest_legal_move(target: chess.Move, board: chess.Board) -> tuple[chess.Move, str]:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("Board has no legal moves (checkmate or stalemate).")

    rules = [
        ("same_from", lambda m: m.from_square == target.from_square),
        ("same_to", lambda m: m.to_square == target.to_square),
        ("same_rank_file", lambda m: same_line(m.from_square, target.from_square)),
        ("same_rank_file", lambda m: same_line(m.to_square, target.to_square)),
    ]
    for name, rule in rules:
        for move in legal_moves:
            if rule(move):
                return move, name
    return min(legal_moves, key=lambda m: m.uci()), "first_legal"


def find_closest_legal(target: chess.Move, board: chess.Board) -> chess.Move:
    return closest_legal_move(target, board)[0]


def filter_to_legal(raw_text: str, board: chess.Board) -> FilterResult:
    raw = extract_uci_candidate(raw_text)
    parsed = try_parse_uci(raw)

    if parsed is not None and parsed in board.legal_moves:
        return FilterResult(raw, parsed, was_legal=True, was_parseable=True, fallback_strategy="direct")

    if parsed is not None:
        move, strategy = closest_legal_move(parsed, board)
        return FilterResult(raw, move, was_legal=False, was_parseable=True, fallback_strategy=strategy)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("Board has no legal moves.")
    first = min(legal_moves, key=lambda m: m.uci())
    return FilterResult(raw, first, was_legal=False, was_parseable=False, fallback_strategy="first_legal")
