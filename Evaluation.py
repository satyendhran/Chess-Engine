import numpy as np
from numba import njit, u4, u8
from numba.experimental import jitclass

from Board import Board
from Move_gen_pieces import (
    get_bishop_attacks,
    get_knight_attacks,
    get_pawn_attacks,
    get_queen_attacks,
    get_rook_attacks,
)
from pregen.Utilities import count_bits, get_lsb1_index, pop_bit
from PST import pst_eval

PIECE_VALUES = np.array(
    [100, 320, 330, 500, 900, 100000, -100, -320, -330, -500, -900, -100000],
    dtype=np.int32,
)

ENDGAME_CONTRIB = np.array([0, 1, 1, 2, 4, 0, 0, 1, 1, 2, 4, 0], dtype=np.int32)

MAX_ENDGAME_CONTRIB = np.int32(
    8 * ENDGAME_CONTRIB[0]
    + 2 * ENDGAME_CONTRIB[1]
    + 2 * ENDGAME_CONTRIB[2]
    + 2 * ENDGAME_CONTRIB[3]
    + 1 * ENDGAME_CONTRIB[4]
    + 8 * ENDGAME_CONTRIB[6]
    + 2 * ENDGAME_CONTRIB[7]
    + 2 * ENDGAME_CONTRIB[8]
    + 2 * ENDGAME_CONTRIB[9]
    + 1 * ENDGAME_CONTRIB[10]
)

MOBILITY_CONTRIB_MID = np.array(
    [1, 8, 15, 9, -6, -2, -1, -8, -15, -9, +6, +2], dtype=np.int32
)

MOBILITY_CONTRIB_END = np.array(
    [1, 6, 11, 15, 12, 15, -1, -6, -11, -15, -12, -15], dtype=np.int32
)

spec = [("board", Board.class_type.instance_type)]


KING_ACT = []


@njit(u8(u4, u4), inline="always")
def manhattan_center(x, y):
    ix = np.uint16(3 - x if x < 4 else x - 4)
    iy = np.uint16(3 - y if y < 4 else y - 4)
    return ix + iy

@njit
def same_quadrant(x1, y1, x2, y2):
    return (x1 >= 4) == (x2 >= 4) and (y1 >= 4) == (y2 >= 4)


@jitclass(spec)
class Evaluation:
    def __init__(self, board):
        self.board = board

    def endgame_score_calc(self):
        bb_arr = self.board.bitboard
        end_contrib = ENDGAME_CONTRIB
        total = np.int32(0)
        for p in range(12):
            total += np.int32(count_bits(bb_arr[p])) * end_contrib[p]
        eg = MAX_ENDGAME_CONTRIB - total
        return np.uint8((eg * np.int32(256)) // MAX_ENDGAME_CONTRIB)

    def get_mobility_piece(self, square, occupancy, piece):
        if piece == 0 or piece == 6:
            return get_pawn_attacks(square, piece == 6)
        elif piece == 1 or piece == 7:
            return get_knight_attacks(square)
        elif piece == 2 or piece == 8:
            return get_bishop_attacks(square, occupancy)
        elif piece == 3 or piece == 9:
            return get_rook_attacks(square, occupancy)
        else:
            return get_queen_attacks(square, occupancy)

    def piece_based_evals(self, endgame_weight):
        bb_arr = self.board.bitboard
        occ_all = self.board.occupancy[2]
        pv = PIECE_VALUES
        mid_tab = MOBILITY_CONTRIB_MID
        end_tab = MOBILITY_CONTRIB_END
        local_get_lsb = get_lsb1_index
        local_pop = pop_bit
        local_cnt = count_bits
        local_pst = pst_eval
        local_mob = self.get_mobility_piece

        inv_weight = np.int64(256 - endgame_weight)
        e_weight = np.int64(endgame_weight)

        score = np.int64(0)

        for piece in range(12):
            bb = bb_arr[piece]
            if bb == 0:
                continue

            pvalue = np.int64(pv[piece])
            mid = np.int64(mid_tab[piece])
            endv = np.int64(end_tab[piece])
            mob_scale = (inv_weight * mid + e_weight * endv) >> 8

            while bb:
                sq = local_get_lsb(bb)
                mg = np.int64(local_pst(piece, sq, 0))
                eg = np.int64(local_pst(piece, sq, 1))
                blended = ((inv_weight * mg) + (e_weight * eg)) >> 8

                mob = np.int64(local_cnt(local_mob(sq, occ_all, piece)))
                score += pvalue + blended + mob * mob_scale

                bb = local_pop(bb, sq)

        return score

    def king_activity(self, endgame_weight):
        if endgame_weight < 128:
            return 0
        score = np.int32(0)
        wk_sq = get_lsb1_index(self.board.bitboard[5])
        bk_sq = get_lsb1_index(self.board.bitboard[11])

        wx, wy = wk_sq >> 3, wk_sq & 7
        bx, by = bk_sq >> 3, bk_sq & 7

        dw, db = manhattan_center(wx, wy), manhattan_center(bx, by)
        dx, dy = np.abs(wx - bx), np.abs(wy - by)
        sign = 1 - (1 << self.board.side)

        if dx == 0 and dy & 1:
            score += 100 * (7 - dy) * sign
        elif dx == 0:
            score -= 100 * (7 - dy) * sign
        if dy == 0 and dx & 1:
            score += 100 * (7 - dx) * sign
        elif dy == 0:
            score -= 100 * (7 - dx) * sign
        if (dx == 1 and dy == 2) or (dx == 2 and dy == 1):
            if dw < db:
                score += 75
            elif dw > db:
                score -= 75

        if dx == dy:
            if dx & 1:
                score += 100 * sign * (7 - dx)
            else:
                score -= 100 * sign * (7 - dx)
        if same_quadrant(wx,wy,bx,by):
            if dw > db:
                score -= 100 * dw
            elif dw < db:
                score += 100 * db
        if dw == 6 :
            score -= 200
        if db == 6:
            score += 200
        return np.int64(score * endgame_weight) >> 8

    def evaluate(self):
        eg = self.endgame_score_calc()
        mat_pst = self.piece_based_evals(eg)
        king_term = self.king_activity(eg)
        king_scaled = king_term >> 3
        return np.int64(mat_pst + king_scaled)
