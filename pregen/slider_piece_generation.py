import numpy as np
from bishop_att import bishop_att
from rook_att import rook_att
from Utilities import save_pregen, set_occupancy

bishop_attacks = np.zeros((64, 4096), dtype=np.uint64)
rook_attacks = np.zeros((64, 4096 * 4), dtype=np.uint64)
bishop_mask = np.load("BISHOP_MASK.npy", allow_pickle=True)
rook_mask = np.load("ROOK_MASK.npy", allow_pickle=True)
bishop_relevent_bits = np.load(
    "BISHOP_RELEVENT_BITS.npy",
    allow_pickle=True,
)
bishop_shift = np.load(
    "BISHOP_SHIFTS.npy",
    allow_pickle=True,
)
rook_relevent_bits = np.load(
    "ROOK_RELEVENT_BITS.npy",
    allow_pickle=True,
)
rook_shift = np.load("ROOK_SHIFTS.npy", allow_pickle=True)
bishop_magic = np.load(
    "BISHOP_MAGICS.npy",
    allow_pickle=True,
)
rook_magic = np.load("ROOK_MAGICS.npy", allow_pickle=True)


def init_slider_attacks(
    is_bishop: bool,
):
    for sq in range(64):
        attack_mask = bishop_mask[sq] if is_bishop else rook_mask[sq]
        relevent_bits = (
            bishop_relevent_bits[sq] if is_bishop else rook_relevent_bits[sq]
        )
        occupancy_indices = np.uint64(1) << relevent_bits
        # # FOR DEBUG
        # print(occupancy_indices,relevent_bits)
        # print_bitboard(attack_mask)
        for index in range(occupancy_indices):
            if is_bishop:
                occupancy = set_occupancy(
                    index,
                    relevent_bits,
                    attack_mask,
                )
                magic_index = (occupancy * bishop_magic[sq]) >> bishop_shift[sq]
                bishop_attacks[sq][magic_index] = bishop_att(sq, occupancy)

            else:
                occupancy = set_occupancy(
                    index,
                    relevent_bits,
                    attack_mask,
                )
                magic_index = (occupancy * rook_magic[sq]) >> rook_shift[sq]

                k = rook_att(sq, occupancy)
                rook_attacks[sq][magic_index] = k
                # # FOR DEBUG
                # print(rook_attacks[sq][magic_index])


if __name__ == "__main__":
    init_slider_attacks(True)
    init_slider_attacks(False)
    save_pregen("ROOK_ATTACKS", rook_attacks)
    save_pregen("BISHOP_ATTACKS", bishop_attacks)
