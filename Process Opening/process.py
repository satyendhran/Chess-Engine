import re

import chess
import pandas as pd

from Board import Board, parse_fen
from Board_Move_gen import Move, Move_generator, move_to_uci, unmove


def clean_pgn(pgn):
    pgn = re.sub(r"\d+\.", "", pgn)
    pgn = re.sub(r"(1-0|0-1|1/2-1/2)", "", pgn)
    return pgn.split()


dfs = []
for f in "abcde":
    dfs.append(pd.read_table(f"{f}.tsv", usecols=["name", "pgn"]))

df = pd.concat(dfs, ignore_index=True)
df = df.drop_duplicates(subset=["name", "pgn"]).reset_index(drop=True)
tree = {}

for pgn in df["pgn"]:
    moves = clean_pgn(pgn)
    node = tree

    for move in moves:
        node = node.setdefault(move, {})



def convert_tree_to_uci(tree, board=None):
    if board is None:
        board = chess.Board()

    uci_tree = {}

    for san, subtree in tree.items():
        move = board.parse_san(san)
        uci = move.uci()

        board.push(move)
        uci_tree[uci] = convert_tree_to_uci(subtree, board)
        board.pop()

    return uci_tree


tree = convert_tree_to_uci(tree)


def convert_tree_to_mine(tree, board=None):
    if board is None:
        board = Board(
            *parse_fen(b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        )

    uci_tree = {}
    c, e, p = board.castle, board.enpassant, board.halfmove
    movegen = Move_generator(board)

    for uci, subtree in tree.items():
        for i in range(movegen.counter):
            move = movegen.moves[i]
            if move_to_uci(move) == uci:
                mine = move
                break
        Move(board, mine)
        uci_tree[int(mine)] = convert_tree_to_mine(subtree, board)
        unmove(board, mine, c, e, p)

    return uci_tree


# tree = convert_tree_to_mine(tree)

import msgpack

with open("opening_tree.msgpack", "wb") as f:
    msgpack.pack(tree, f, use_bin_type=True, strict_types=True)
