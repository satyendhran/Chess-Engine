from numba import njit
from numba.types import uint8 as u8  # type:ignore
from numba.types import uint64 as u64  # type:ignore
from Utilities import save_pregen


@njit(u64(u8))
def knight_att(square):
    A_FILE = 0x101010101010101
    H_FILE = 0x8080808080808080
    AB_FILE = 0x202020202020202 | A_FILE
    GH_FILE = 0x4040404040404040 | H_FILE

    knight_square = 1 << square
    att = 0

    if not (knight_square & A_FILE):
        if square > 15:
            att |= 1 << (square - 17)
        if square < 48:
            att |= 1 << (square + 15)

    if not (knight_square & H_FILE):
        if square > 15:
            att |= 1 << (square - 15)
        if square < 48:
            att |= 1 << (square + 17)

    if not (knight_square & AB_FILE):
        if square >= 8:
            att |= 1 << (square - 10)
        if square <= 55:
            att |= 1 << (square + 6)

    if not (knight_square & GH_FILE):
        if square >= 8:
            att |= 1 << (square - 6)
        if square <= 55:
            att |= 1 << (square + 10)

    return att


def init_knight_att():
    KNIGHT_ATTACKS = [0] * 64
    for sq in range(64):
        KNIGHT_ATTACKS[sq] = knight_att(sq)

    return KNIGHT_ATTACKS


if __name__ == "__main__":
    KNIGHT_ATTACKS = init_knight_att()
    save_pregen("KNIGHT_ATTACKS", KNIGHT_ATTACKS)
    # # FOR DEBUG
    # for x in range(64):
    #     print_bitboard(knight_att(x),debug_square=x,highlight = True)
    #     input()
    # print_bitboard(knight_att(8),debug_square=8)
