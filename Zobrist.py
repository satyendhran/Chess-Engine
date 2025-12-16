import numpy as np
from pregen.Utilities import get_lsb1_index,pop_bit
from numba import njit,u4 as u32,u8 as u64


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

piece_keys = np.empty((12, 64), dtype=np.uint64)
enpassant_keys = np.empty(64, dtype=np.uint64)
castle_keys = np.empty(16, dtype=np.uint64)
state = 3810971855884203905
state = get_64_bit_random_number(state)
side_key = state
def init_keys(piece_keys, enpassant_keys, castle_keys,state):
    for p in range(12):
        for q in range(64):
            state = get_64_bit_random_number(state)
            piece_keys[p, q] = state
    for p in range(64):
        state = get_64_bit_random_number(state)
        enpassant_keys[p] = state

    for p in range(16):
        state = get_64_bit_random_number(state)
        castle_keys[p] = state
    return state

state = init_keys(piece_keys, enpassant_keys, castle_keys,state)



if __name__ == '__main__':
    from Board import Board,parse_fen
    from GUI import FENS
    board = Board(*parse_fen(FENS.STARTING))
    hash = genhash(board)
    print(hex(hash))