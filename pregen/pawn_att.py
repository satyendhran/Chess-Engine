import numpy as np
from numba import njit
from numba.types import uint8 as u8  # type:ignore
from numba.types import uint64 as u64  # type:ignore
from Utilities import Color, print_bitboard, save_pregen


@njit(u64(u8, u8))
def pawn_att(square, color):
    A_FILE = 0x101010101010101
    H_FILE = 0x8080808080808080

    pawn_sq = 1 << square
    if color == Color.BLACK:
        att1, att2 = 1 << (square + 9), 1 << (square + 7)
    else:
        att1, att2 = 1 << (square - 7), 1 << (square - 9)

    att = 0
    if not (pawn_sq & H_FILE):
        att |= att1

    if not (pawn_sq & A_FILE):
        att |= att2

    return att


def init_pawn_att():
    PAWN_ATTACKS = [[0] * 64, [0] * 64]
    for sq in range(64):
        PAWN_ATTACKS[Color.WHITE][sq] = pawn_att(sq, Color.WHITE)
        PAWN_ATTACKS[Color.BLACK][sq] = pawn_att(sq, Color.BLACK)

    return PAWN_ATTACKS


if __name__ == "__main__":
    PAWN_ATTACKS = init_pawn_att()
    save_pregen("PAWN_ATTACKS", PAWN_ATTACKS)

    # # FOR DEBUG
    # # FOR WHITE
    # for i,x in enumerate(PAWN_ATTACKS[0]):
    #     print_bitboard(x,debug_square=i)
    #     input()

    # # FOR BLACK
    # for i,x in enumerate(PAWN_ATTACKS[1]):
    #     print_bitboard(x,debug_square=i)
    #     input()
