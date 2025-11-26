import numpy as np
from numba import njit
from numba.types import uint8 as u8  # type:ignore
from numba.types import uint64 as u64  # type:ignore
from Utilities import Color, count_bits, print_bitboard, save_pregen


@njit(u64(u8))
def mask_rook_att(square: int):
    r, f = divmod(square, 8)
    att = 0
    for dr in range(r + 1, 7):
        sq = 8 * (dr) + (f)
        att |= 1 << sq

    for dr in range(r - 1, 0, -1):
        sq = 8 * (dr) + (f)
        att |= 1 << sq

    for df in range(f - 1, 0, -1):
        sq = 8 * (r) + (df)
        att |= 1 << sq

    for df in range(f + 1, 7):
        sq = 8 * (r) + (df)
        att |= 1 << sq
    return att


# # FOR DEBUG
# for x in range(64):
#     print_bitboard(mask_rook_att(x),debug_square=x)
#     input()


@njit(u64(u8, u64))
def rook_att(square: int, occupancy: int) -> int:
    att = 0
    r, f = divmod(square, 8)

    for dr in range(r + 1, 8):
        sq = 8 * (dr) + (f)
        att |= 1 << sq
        if occupancy & (1 << sq):
            break

    for dr in range(r - 1, -1, -1):
        sq = 8 * (dr) + (f)
        att |= 1 << sq
        if occupancy & (1 << sq):
            break

    for df in range(f - 1, -1, -1):
        sq = 8 * (r) + (df)
        att |= 1 << sq
        if occupancy & (1 << sq):
            break

    for df in range(f + 1, 8):
        sq = 8 * (r) + (df)
        att |= 1 << sq
        if occupancy & (1 << sq):
            break

    return att


# # FOR DEBUG
# for x in range(64):
#     print_bitboard(rook_att(x,0),debug_square=x)
#     input()

if __name__ == "__main__":
    rook_mask_pregen = np.array([mask_rook_att(x) for x in range(64)], dtype=np.uint64)
    save_pregen("ROOK_MASK.npy", rook_mask_pregen)

    rook_relevent_bits = np.array(
        [count_bits(mask_rook_att(x)) for x in range(64)], dtype=np.uint8
    )
    save_pregen("ROOK_RELEVENT_BITS.npy", rook_relevent_bits)
