import os
from enum import IntEnum

import numpy as np
from numba import njit
from numba.types import uint8 as u8  # type:ignore
from numba.types import uint64 as u64  # type:ignore


@njit(u64(u64, u8))
def set_bit(bitboard: int, square: int) -> int:
    return bitboard | (1 << square)


@njit(u64(u64, u8))
def get_bit(bitboard: int, square: int) -> int:
    return (bitboard >> square) & 1


@njit(u64(u64, u8))
def pop_bit(bitboard: int, square: int) -> int:
    return bitboard ^ (1 << square)


def print_bitboard(
    bitboard: int,
    debug_square=64,
    highlight=False,
) -> None:
    cyan = "\033[36m"
    red = "\033[31m"
    reset = "\033[0m"
    for r in range(8):
        for f in range(8):
            if f == 0:
                print(8 - r, end="  ")
            square = 8 * r + f
            if square == debug_square:
                print(
                    f"{cyan}X{reset}",
                    end=" ",
                )
                continue
            if get_bit(bitboard, square=square):
                print(
                    f"{red}1{reset}",
                    end=" ",
                )
            else:
                print("0", end=" ")
        print()
    print("\n   a b c d e f g h")
    print(f"\n   Bitboard : {bitboard}")
    if debug_square < 64:
        print(
            f"   Debug square Value : {get_bit(bitboard, square=debug_square)}"
        )


class Color(IntEnum):
    WHITE = 0
    BLACK = 1


def save_pregen(name, arr):
    if isinstance(arr, list):
        np.save(
            os.path.join(name),
            np.array(arr, dtype=np.uint64),
            allow_pickle=True,
        )
        return
    np.save(
        os.path.join(name),
        arr=arr,
        allow_pickle=True,
    )


@njit(u8(u64))
def count_bits(bitboard: int) -> int:
    count = u8(0)
    while bitboard:
        count += u8(1)
        bitboard &= bitboard - u64(1)
    return count


@njit(u8(u64))
def get_lsb1_index(bitboard: int):
    if bitboard:
        return count_bits((bitboard & -bitboard) - 1)
    return 64


@njit(u64(u64, u8, u64))
def set_occupancy(index, bits_in_mask, attack_mask):
    occupancy = u64(0)
    temp_mask = attack_mask
    for count in range(bits_in_mask):
        square = get_lsb1_index(temp_mask)
        temp_mask = pop_bit(temp_mask, square)
        if index & (u64(1) << count):
            occupancy = set_bit(occupancy, square)
    return occupancy
