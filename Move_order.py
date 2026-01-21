import numpy as np
from numba import i4, njit, u4, u8

from Board_Move_gen import (
    get_capture_piece,
    get_flag,
    get_starting_piece,
    get_target_square,
)
from Constants import Flag

killers = np.zeros(shape=(2, 64), dtype=np.uint64)
history = np.zeros((12, 64), dtype=np.uint32)
pv_length = np.zeros(shape=64, dtype=np.uint32)
pv_table = np.zeros(shape=(64, 64), dtype=np.uint32)

mvv_lva = np.array(
    [
        [105, 205, 305, 405, 505, 605, 105, 205, 305, 405, 505, 605],
        [104, 204, 304, 404, 504, 604, 104, 204, 304, 404, 504, 604],
        [103, 203, 303, 403, 503, 603, 103, 203, 303, 403, 503, 603],
        [102, 202, 302, 402, 502, 602, 102, 202, 302, 402, 502, 602],
        [101, 201, 301, 401, 501, 601, 101, 201, 301, 401, 501, 601],
        [100, 200, 300, 400, 500, 600, 100, 200, 300, 400, 500, 600],
        [105, 205, 305, 405, 505, 605, 105, 205, 305, 405, 505, 605],
        [104, 204, 304, 404, 504, 604, 104, 204, 304, 404, 504, 604],
        [103, 203, 303, 403, 503, 603, 103, 203, 303, 403, 503, 603],
        [102, 202, 302, 402, 502, 602, 102, 202, 302, 402, 502, 602],
        [101, 201, 301, 401, 501, 601, 101, 201, 301, 401, 501, 601],
        [100, 200, 300, 400, 500, 600, 100, 200, 300, 400, 500, 600],
    ],
    dtype=np.uint32,
)


@njit(i4(u8, u4))
def score(move, ply):
    s = np.uint32(0)
    flag = get_flag(move)
    if (
        flag == Flag.CAPTURE
        or flag == Flag.CAPTURE_PROMOTION_ROOK
        or flag == Flag.CAPTURE_PROMOTION_BISHOP
        or flag == Flag.CAPTURE_PROMOTION_KNIGHT
        or flag == Flag.CAPTURE_PROMOTION_QUEEN
    ):
        s += mvv_lva[get_starting_piece(move)][get_capture_piece(move)]
    else:
        if killers[0][ply] == move:
            s += 9000
        elif killers[1][ply] == move:
            s += 8000
        else:
            s += history[get_starting_piece(move)][get_target_square(move)]

    return s


@njit
def sort_moves(moves, ply):
    n = moves.counter
    scores = np.zeros(n, dtype=np.int32)

    for i in range(n):
        scores[i] = score(moves.moves[i], ply)

    for i in range(n):
        best = i
        best_score = scores[i]

        for j in range(i + 1, n):
            if scores[j] > best_score:
                best = np.uint8(j)
                best_score = scores[j]

        if best != i:
            # swap scores
            tmp_score = scores[i]
            scores[i] = scores[best]
            scores[best] = tmp_score

            # swap moves
            tmp_move = moves.moves[i]
            moves.moves[i] = moves.moves[best]
            moves.moves[best] = tmp_move
