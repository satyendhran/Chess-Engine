import os

import numpy as np
from numba import njit
from numba.types import uint8 as u8  # type:ignore
from numba.types import uint64 as u64  # type:ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pawn_att = np.load(
    os.path.join(BASE_DIR, "PAWN_ATTACKS.npy"),
    allow_pickle=True,
)

knight_att = np.load(
    os.path.join(BASE_DIR, "KNIGHT_ATTACKS.npy"),
    allow_pickle=True,
)

king_att = np.load(
    os.path.join(BASE_DIR, "KING_ATTACKS.npy"),
    allow_pickle=True,
)

bishop_att = np.load(
    os.path.join(BASE_DIR, "BISHOP_ATTACKS.npy"),
    allow_pickle=True,
)

rook_att = np.load(
    os.path.join(BASE_DIR, "ROOK_ATTACKS.npy"),
    allow_pickle=True,
)

bishop_mask = np.load(
    os.path.join(BASE_DIR, "BISHOP_MASK.npy"),
    allow_pickle=True,
)

rook_mask = np.load(
    os.path.join(BASE_DIR, "ROOK_MASK.npy"),
    allow_pickle=True,
)

bishop_shift = np.load(
    os.path.join(BASE_DIR, "BISHOP_SHIFTS.npy"),
    allow_pickle=True,
)

rook_shift = np.load(
    os.path.join(BASE_DIR, "ROOK_SHIFTS.npy"),
    allow_pickle=True,
)

bishop_magic = np.load(
    os.path.join(BASE_DIR, "BISHOP_MAGICS.npy"),
    allow_pickle=True,
)

rook_magic = np.load(
    os.path.join(BASE_DIR, "ROOK_MAGICS.npy"),
    allow_pickle=True,
)


@njit(u64(u8, u8), nogil=True)
def get_pawn_attacks(sq, color):
    return pawn_att[color, sq]


@njit(u64(u8), nogil=True)
def get_knight_attacks(sq):
    return knight_att[sq]


@njit(u64(u8), nogil=True)
def get_king_attacks(sq):
    return king_att[sq]


@njit(u64(u8, u64), nogil=True)
def get_bishop_attacks(sq, occupancy):
    occupancy &= bishop_mask[sq]
    occupancy *= bishop_magic[sq]
    occupancy >>= bishop_shift[sq]
    return bishop_att[sq, occupancy]


@njit(u64(u8, u64), nogil=True)
def get_rook_attacks(sq, occupancy):
    occupancy &= rook_mask[sq]
    occupancy *= rook_magic[sq]
    occupancy >>= rook_shift[sq]
    return rook_att[sq, occupancy]


@njit(u64(u8, u64), nogil=True)
def get_queen_attacks(sq, occupancy):
    return get_bishop_attacks(sq, occupancy) | get_rook_attacks(sq, occupancy)


# # FOR DEBUG
# if __name__ == "__main__":
#     for sq in range(64):
#         print_bitboard(get_pawn_attacks(sq,0),debug_square=sq,highlight=True)

#     for sq in range(64):
#         print_bitboard(get_pawn_attacks(sq,1),debug_square=sq,highlight=True)

#     for sq in range(64):
#         print_bitboard(get_knight_attacks(sq),debug_square=sq,highlight=True)

#     for sq in range(64):
#         print_bitboard(get_bishop_attacks(sq,0),debug_square=sq,highlight=True)

#     for sq in range(64):
#         print_bitboard(get_rook_attacks(sq,0),debug_square=sq,highlight=True)

#     for sq in range(64):
#         print_bitboard(get_king_attacks(sq),debug_square=sq,highlight=True)
