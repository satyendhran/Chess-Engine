from os.path import exists
from random import getrandbits

import numpy as np
from bishop_att import bishop_att, mask_bishop_att
from numba import njit, prange
from numba.types import uint8 as u8  # type:ignore
from numba.types import uint32 as u32  # type:ignore
from numba.types import uint64 as u64  # type:ignore
from rook_att import mask_rook_att, rook_att
from Utilities import count_bits, save_pregen, set_occupancy


@njit(u32(u32))
def get_random_number(state):
    state ^= state << np.uint32(13)
    state ^= state >> np.uint32(7)
    state ^= state << np.uint32(17)
    return state


@njit(u64(u64))
def get_64_bit_random_number(state):
    n1 = get_random_number(np.uint32(state & np.uint64(0xFFFFFFFF)))
    n2 = get_random_number(n1)
    n3 = get_random_number(n2)
    n4 = get_random_number(n3)
    return (
        (np.uint64(n1) << np.uint64(48))
        | (np.uint64(n2) << np.uint64(32))
        | (np.uint64(n3) << np.uint64(16))
        | np.uint64(n4)
    )


@njit(u64[:](u32, u8, u8))
def find_magic_number_bishop(state, square, relevant_bits):
    occupancies = np.zeros(4096, dtype=np.uint64)
    attacks = np.zeros(4096, dtype=np.uint64)
    used_attacks = np.zeros(4096, dtype=np.uint64)

    attack_mask = np.uint64(mask_bishop_att(square))
    occupancy_indices = np.uint64(1) << relevant_bits

    for index in range(occupancy_indices):
        occupancies[index] = np.uint64(set_occupancy(index, relevant_bits, attack_mask))
        attacks[index] = np.uint64(bishop_att(square, occupancies[index]))

    for _ in range(10**6):
        state = get_64_bit_random_number(state)
        magic_number = np.uint64(state)

        # Keep product in 64-bit space
        product = (attack_mask * magic_number) & np.uint64(0xFFFFFFFFFFFFFFFF)
        if count_bits(product) < 6:  # type:ignore
            continue

        for shift in range(30, 60):
            used_attacks[:] = np.uint64(0)
            fail = False

            for index in range(occupancy_indices):
                magic_index = (
                    np.uint64(occupancies[index]) * magic_number
                ) >> np.uint64(shift)
                if magic_index > 4095:
                    fail = True
                    break
                if used_attacks[magic_index] == 0:
                    used_attacks[magic_index] = attacks[index]
                elif used_attacks[magic_index] != attacks[index]:
                    fail = True
                    break

            if not fail:
                return np.array(
                    [magic_number, np.uint64(shift)], dtype=np.uint64
                )  # store shift in high bits

    return np.array([np.uint64(0), np.uint64(0)], dtype=np.uint64)  # Failed


# === Driver with retries ===
bishop_relevant_bits = np.load("BISHOP_RELEVENT_BITS.npy", allow_pickle=True)
bishop_magics = np.load("BISHOP_MAGICS.npy", allow_pickle=True)
bishop_shifts = np.load("BISHOP_SHIFTS.npy", allow_pickle=True)
failed_squares = []

state = np.uint64(getrandbits(32))
state = get_64_bit_random_number(state)

for sq in range(64):
    print(f"Finding magic number for bishop on square {sq}")
    found = False
    state = get_64_bit_random_number(state)

    for attempt in range(5):
        magic, shift = find_magic_number_bishop(
            state, np.uint64(sq), np.uint64(bishop_relevant_bits[sq])
        )
        if magic != 0:
            bishop_magics[sq] = magic
            bishop_shifts[sq] = shift
            print(
                f"Magic number for bishop on square {sq} is {magic} with shift {shift}"
            )
            found = True
            break
        state = get_64_bit_random_number(state)

    if not found:
        print(f"Failed to find magic number for square {sq} in first pass")
        failed_squares.append(sq)


save_pregen("BISHOP_MAGICS.npy", bishop_magics)
save_pregen("BISHOP_SHIFTS.npy", bishop_shifts)

# Second pass for failed squares (10 attempts each)
if failed_squares:
    print(f"\nRetrying {len(failed_squares)} failed squares with fresh states...")
    for sq in failed_squares[:]:
        found = False
        for _ in range(20):  # fixed at 10 attempts
            print(f"Finding magic number for bishop on square {sq} attempt {_ + 1}")
            state = np.uint64(getrandbits(32))
            magic, shift = find_magic_number_bishop(
                state, np.uint8(sq), np.uint8(bishop_relevant_bits[sq])
            )
            if magic != 0:
                bishop_magics[sq] = magic
                bishop_shifts[sq] = shift
                print(
                    f"Recovered magic number for square {sq} is {magic} with shift {shift}"
                )
                failed_squares.remove(sq)
                found = True
                break
        if not found:
            print(f"Final fail: square {sq}")

save_pregen("BISHOP_MAGICS.npy", bishop_magics)
save_pregen("BISHOP_SHIFTS.npy", bishop_shifts)

if failed_squares:
    print(f"\nWARNING: Failed to generate magic numbers for {failed_squares}")


@njit(u64[:](u64, u64, u64))
def find_magic_number_rook(state, square, relevant_bits):
    occupancies = np.zeros(8192, dtype=np.uint64)
    attacks = np.zeros(8192, dtype=np.uint64)
    used_attacks = np.zeros(8192, dtype=np.uint64)

    attack_mask = np.uint64(mask_rook_att(square))
    occupancy_indices = np.uint64(1) << relevant_bits

    for index in range(occupancy_indices):
        occupancies[index] = np.uint64(set_occupancy(index, relevant_bits, attack_mask))
        attacks[index] = np.uint64(rook_att(square, occupancies[index]))

    for _ in range(10**6):
        state = get_64_bit_random_number(state)
        magic_number = np.uint64(state)

        product = (attack_mask * magic_number) & np.uint64(0xFFFFFFFFFFFFFFFF)
        if count_bits(product) < 6:  # type:ignore
            continue
        shift = u64(60)
        while shift > 30:
            used_attacks[:] = np.uint64(0)
            fail = False

            for index in range(occupancy_indices):
                magic_index = (
                    np.uint64(occupancies[index]) * magic_number
                ) >> np.uint64(shift)
                if magic_index > 8192 - 1:
                    fail = True
                    break

                if used_attacks[magic_index] == 0:
                    used_attacks[magic_index] = attacks[index]
                elif used_attacks[magic_index] != attacks[index]:
                    fail = True
                    break

            if not fail:
                return np.array([magic_number, (np.uint64(shift))], dtype=np.uint64)

            shift -= 1

    return np.array([0, 0], dtype=np.uint64)


rook_relevant_bits = np.load("ROOK_RELEVENT_BITS.npy", allow_pickle=True)

# Create magics/shifts arrays if not existing
if exists("ROOK_MAGICS.npy"):
    rook_magics = np.load("ROOK_MAGICS.npy", allow_pickle=True)
else:
    rook_magics = np.zeros(64, dtype=np.uint64)

if exists("ROOK_SHIFTS.npy"):
    rook_shifts = np.load("ROOK_SHIFTS.npy", allow_pickle=True)
else:
    rook_shifts = np.zeros(64, dtype=np.uint64)

failed_squares = []

state = np.uint64(getrandbits(32))
state = get_64_bit_random_number(state)

for sq in range(64):
    print(f"Finding magic number for rook on square {sq}")
    found = False
    state = get_64_bit_random_number(state)

    for attempt in range(5):
        magic, shift = find_magic_number_rook(
            state, np.uint8(sq), np.uint8(rook_relevant_bits[sq])
        )
        if magic != 0:
            rook_magics[sq] = magic
            rook_shifts[sq] = shift
            print(f"Magic number for rook on square {sq} is {magic} with shift {shift}")
            found = True
            break
        state = get_64_bit_random_number(state)

    if not found:
        print(f"Failed to find magic number for square {sq} in first pass")
        failed_squares.append(sq)

save_pregen("ROOK_MAGICS.npy", rook_magics)
save_pregen("ROOK_SHIFTS.npy", rook_shifts)

# Second pass for failed squares
if failed_squares:
    print(f"\nRetrying {len(failed_squares)} failed squares with fresh states...")
    for sq in failed_squares[:]:
        found = False
        for _ in range(10):
            print(f"Finding magic number for rook on square {sq} attempt {_ + 1}")
            state = get_64_bit_random_number(state)
            magic, shift = find_magic_number_rook(
                state, np.uint8(sq), np.uint8(rook_relevant_bits[sq])
            )
            if magic != 0:
                rook_magics[sq] = magic
                rook_shifts[sq] = shift
                print(
                    f"Recovered magic number for square {sq} is {magic} with shift {shift}"
                )
                failed_squares.remove(sq)
                found = True
                break
        if not found:
            print(f"Final fail: square {sq}")

# Save after retry phase
save_pregen("ROOK_MAGICS.npy", rook_magics)
save_pregen("ROOK_SHIFTS.npy", rook_shifts)

# Print final failed list
if failed_squares:
    print(f"\nFinal failed squares list: {failed_squares}")
else:
    print("\nAll rook magic numbers found successfully!")
