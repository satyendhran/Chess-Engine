import numpy as np
from numba import njit
from numba.types import uint8 as u8  # type:ignore
from numba.types import uint64 as u64  # type:ignore
from Utilities import count_bits, save_pregen


@njit(u64(u8))
def mask_bishop_att(square: int):
    r, f = divmod(square, 8)
    att = 0
    for dr, df in zip(range(r + 1, 7), range(f + 1, 7)):
        sq = 8 * (dr) + df
        att |= 1 << sq

    for dr, df in zip(
        range(r - 1, 0, -1),
        range(f - 1, 0, -1),
    ):
        sq = 8 * (dr) + (df)
        att |= 1 << sq

    for dr, df in zip(
        range(r + 1, 7),
        range(f - 1, 0, -1),
    ):
        sq = 8 * (dr) + (df)
        att |= 1 << sq

    for dr, df in zip(
        range(r - 1, 0, -1),
        range(f + 1, 7),
    ):
        sq = 8 * (dr) + (df)
        att |= 1 << sq
    return att


# # FOR DEBUG
# for x in range(64):
#     print_bitboard(mask_bishop_att(x),debug_square=x)
#     input()


@njit(u64(u8, u64))
def bishop_att(square: int, occupancy: int) -> int:
    att = 0
    r, f = divmod(square, 8)

    for dr, df in zip(range(r + 1, 8), range(f + 1, 8)):
        sq = 8 * (dr) + (df)
        att |= 1 << sq
        if occupancy & (1 << sq):
            break

    for dr, df in zip(
        range(r - 1, -1, -1),
        range(f - 1, -1, -1),
    ):
        sq = 8 * (dr) + (df)
        att |= 1 << sq
        if occupancy & (1 << sq):
            break

    for dr, df in zip(
        range(r + 1, 8),
        range(f - 1, -1, -1),
    ):
        sq = 8 * (dr) + (df)
        att |= 1 << sq
        if occupancy & (1 << sq):
            break

    for dr, df in zip(
        range(r - 1, -1, -1),
        range(f + 1, 8),
    ):
        sq = 8 * (dr) + (df)
        att |= 1 << sq
        if occupancy & (1 << sq):
            break

    return att


# # FOR DEBUG
# for x in range(64):
#     print_bitboard(bishop_att(x,0),debug_square=x)
#     input()

if __name__ == "__main__":
    bishop_mask_pregen = np.array(
        [mask_bishop_att(x) for x in range(64)],
        dtype=np.uint64,
    )
    save_pregen(
        "BISHOP_MASK.npy",
        bishop_mask_pregen,
    )

    bishop_relevent_bits = np.array(
        [count_bits(mask_bishop_att(x)) for x in range(64)],
        dtype=np.uint8,
    )
    save_pregen(
        "BISHOP_RELEVENT_BITS.npy",
        bishop_relevent_bits,
    )
