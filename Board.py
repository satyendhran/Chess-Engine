from typing import Literal

import numpy as np
from numba import njit
from numba.types import int64,int16 # type:ignore
from numba.typed import Dict
from numba.experimental import jitclass
from numba.types import uint8 as u8  # type:ignore
from numba.types import uint64 as u64  # type:ignore

from Constants import Color, Pieces, piece_sym
from Move_gen_pieces import (
    get_bishop_attacks,
    get_king_attacks,
    get_knight_attacks,
    get_pawn_attacks,
    get_queen_attacks,
    get_rook_attacks,
)

specs = [
    ("bitboard", u64[:]),
    ("occupancy", u64[:]),
    ("side", u8),
    ("castle", u8),
    ("enpassant", u8),
    ("halfmove", u8),
]


@njit
def parse_fen(fen: bytes):
    start_fen = b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    if len(start_fen) == len(fen) and np.all(
        np.frombuffer(fen, dtype=np.uint8) == np.frombuffer(start_fen, dtype=np.uint8)
    ):
        pieces = np.array(
            [
                0xFF000000000000,
                0x4200000000000000,
                0x2400000000000000,
                0x8100000000000000,
                0x800000000000000,
                0x1000000000000000,
                0xFF00,
                0x42,
                0x24,
                0x81,
                0x8,
                0x10,
            ],
            dtype=np.uint64,
        )
        occupancy = np.zeros(3, dtype=np.uint64)
        occupancy[0] = (
            pieces[0] | pieces[1] | pieces[2] | pieces[3] | pieces[4] | pieces[5]
        )
        occupancy[1] = (
            pieces[6] | pieces[7] | pieces[8] | pieces[9] | pieces[10] | pieces[11]
        )
        occupancy[2] = occupancy[0] | occupancy[1]
        return (pieces, occupancy, np.uint8(0), np.uint8(0b1111), np.uint8(64), 0)

    parts = fen.split(b" ")
    piece_map = b"\x02\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x01\x00\x00\x04\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x07\x00\x06\n\t"
    pieces = np.zeros(12, dtype=np.uint64)
    sq = np.uint8(0)
    for b in np.frombuffer(parts[0], dtype=np.uint8):
        if b == 47:
            continue
        if b < 58:
            sq = np.uint8(sq + np.uint8(b - 48))
        else:
            idx = piece_map[b - 66]
            pieces[idx] |= np.uint64(1) << sq
            sq = np.uint8(sq + np.uint8(1))
    castle = np.uint8(0)
    if 75 == parts[2][0]:
        castle = np.uint8(castle | 0b0001)
    if 81 == parts[2][1]:
        castle = np.uint8(castle | 0b0010)
    if 107 == parts[2][2]:
        castle = np.uint8(castle | 0b0100)
    if 113 == parts[2][3]:
        castle = np.uint8(castle | 0b1000)
    occupancy = np.zeros(3, dtype=np.uint64)
    occupancy[0] = pieces[0] | pieces[1] | pieces[2] | pieces[3] | pieces[4] | pieces[5]
    occupancy[1] = (
        pieces[6] | pieces[7] | pieces[8] | pieces[9] | pieces[10] | pieces[11]
    )
    occupancy[2] = occupancy[0] | occupancy[1]
    if parts[1][0] == 119:
        side = np.uint8(0)
    else:
        side = np.uint8(1)
    if len(parts[3]) == 1 and parts[3][0] == ord("-"):
        enpassant = np.uint8(64)
    else:
        file = np.uint8(parts[3][0] - 97)
        rank = np.uint8(parts[3][1] - 49)
        enpassant = np.uint8((8 - rank) * 8 + file)
    l = len(parts[3])
    if l == 1:
        ply = parts[3][0] - 48
    else:
        ply = 10 * (parts[3][0] - 48) + (parts[3][1] - 48)

    return (pieces, occupancy, side, castle, enpassant, ply)


@jitclass(specs)  # type:ignore
class Board:
    def __init__(self, bitboard, occupancy, side, castle, enpassant, halfmove):
        self.bitboard = bitboard
        self.occupancy = occupancy
        self.side = side
        self.castle = castle
        self.enpassant = enpassant
        self.halfmove = halfmove

    def copy(self):
        return Board(
            self.bitboard.copy(),
            self.occupancy.copy(),
            self.side,
            self.castle,
            self.enpassant,
            self.halfmove,
        )
spec = []
@njit(u64(u64, u8))
def get_bit(bitboard: int, square: int) -> int:
    return (bitboard >> square) & (1)


def print_board(board: Board, perspective: Color):
    ranks = range(8) if perspective == Color.WHITE else range(7, -1, -1)
    files = range(8) if perspective == Color.WHITE else range(7, -1, -1)

    print("  +---+---+---+---+---+---+---+---+")
    for r in ranks:
        print(
            (8 - r) if perspective == Color.WHITE else (r + 1),
            end=" |",
        )
        for f in files:
            sq = 8 * r + f
            for piece in Pieces:
                if piece == Pieces.NONE:
                    continue
                if get_bit(
                    board.bitboard[piece],
                    sq,
                ):
                    print(
                        f" {piece_sym[piece]} |",
                        end="",
                    )
                    break
            else:
                print("   |", end="")
        print()
        print("  +---+---+---+---+---+---+---+---+")

    if perspective == Color.WHITE:
        print("    a   b   c   d   e   f   g   h")
    else:
        print("    h   g   f   e   d   c   b   a")


@njit
def is_square_attacked(
    board: Board,
    square: int,
    color: int,
) -> Literal[0, 1]:
    # Pawns
    if color == Color.WHITE:
        if get_pawn_attacks(square, Color.BLACK) & board.bitboard[Pieces.P.value]:
            return 1
    else:
        if get_pawn_attacks(square, Color.WHITE) & board.bitboard[Pieces.p.value]:
            return 1

    # Knights
    knight_attacks = get_knight_attacks(square)
    if (color == Color.WHITE and knight_attacks & board.bitboard[Pieces.N.value]) or (
        color == Color.BLACK and knight_attacks & board.bitboard[Pieces.n.value]
    ):
        return 1

    # Kings
    king_attacks = get_king_attacks(square)
    if (color == Color.WHITE and king_attacks & board.bitboard[Pieces.K.value]) or (
        color == Color.BLACK and king_attacks & board.bitboard[Pieces.k.value]
    ):
        return 1

    # Bishops
    bishop_attacks = get_bishop_attacks(square, board.occupancy[2])
    if (color == Color.WHITE and bishop_attacks & board.bitboard[Pieces.B.value]) or (
        color == Color.BLACK and bishop_attacks & board.bitboard[Pieces.b.value]
    ):
        return 1

    # Rooks
    rook_attacks = get_rook_attacks(square, board.occupancy[2])
    if (color == Color.WHITE and rook_attacks & board.bitboard[Pieces.R.value]) or (
        color == Color.BLACK and rook_attacks & board.bitboard[Pieces.r.value]
    ):
        return 1

    # Queens
    queen_attacks = get_queen_attacks(square, board.occupancy[2])
    if (color == Color.WHITE and queen_attacks & board.bitboard[Pieces.Q.value]) or (
        color == Color.BLACK and queen_attacks & board.bitboard[Pieces.q.value]
    ):
        return 1

    return 0


def print_attacked_squares(board: Board, color) -> None:
    for r in range(8):
        for f in range(8):
            if f == 0:
                print(8 - r, end="  ")
            square = 8 * r + f

            print(
                is_square_attacked(
                    board,
                    square=square,
                    color=color,
                ),
                end=" ",
            )
        print()
    print("\n   a b c d e f g h")
