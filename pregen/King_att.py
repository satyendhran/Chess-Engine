import numpy as np
from numba import njit
from numba.types import uint8 as u8  # type:ignore
from numba.types import uint64 as u64  # type:ignore
from Utilities import Color, print_bitboard, save_pregen


@njit(u64(u8))
def king_att(square):
    A_FILE = 0x101010101010101
    H_FILE = 0x8080808080808080
    king_square = 1 << square
    att = 0

    if not (king_square & A_FILE):
        if square > 7:
            att |= 1 << (square - 9)
        if square < 56:
            att |= 1 << (square + 7)
        att |= 1 << (square - 1)

    if not (king_square & H_FILE):
        if square > 7:
            att |= 1 << (square - 7)
        if square < 56:
            att |= 1 << (square + 9)
        att |= 1 << (square + 1)

    if square > 7:
        att |= 1 << (square - 8)
    if square < 56:
        att |= 1 << (square + 8)

    return att


def init_king_att():
    king_ATTACKS = [0] * 64
    for sq in range(64):
        king_ATTACKS[sq] = king_att(sq)

    return king_ATTACKS


if __name__ == "__main__":
    KING_ATTACKS = init_king_att()
    save_pregen("king_ATTACKS", KING_ATTACKS)
    # # FOR DEBUG
    # for x in range(64):
    #     print_bitboard(king_att(x), debug_square=x)
    #     input()
