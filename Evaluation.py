import numpy as np
from numba import int32, int64, uint8, uint64
from numba.experimental import jitclass

from Board import Board
from pregen.Utilities import count_bits, get_lsb1_index, pop_bit
from PST import pst_eval

# -------------------------------------------------
# CONSTANT TABLES (STRICT NUMPY INTEGERS)
# -------------------------------------------------

PIECE_VALUES = np.array([
     100,  320,  330,  500,  900, 100000,
    -100, -320, -330, -500, -900,-100000
], dtype=np.int32)

ENDGAME_CONTRIB = np.array([
    0, 1, 1, 2, 4, 0,
    0, 1, 1, 2, 4, 0
], dtype=np.int32)


# Precomputed maximum endgame contribution (NO runtime math)
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


# -------------------------------------------------
# JITCLASS SPEC
# -------------------------------------------------

spec = [
    ("board", Board.class_type.instance_type),
]


# -------------------------------------------------
# EVALUATION ENGINE (FULLY NUMBA SAFE)
# -------------------------------------------------

@jitclass(spec)
class Evaluation:

    def __init__(self, board):
        self.board = board


    # -------------------------------------------------
    # ENDGAME WEIGHT (RETURNS uint8 0–255)
    # -------------------------------------------------
    def endgame_score_calc(self):
        score = np.int32(0)

        for piece in range(12):
            score += count_bits(self.board.bitboard[piece]) * ENDGAME_CONTRIB[piece]

        eg = MAX_ENDGAME_CONTRIB - score

        return np.uint8((eg << 8) // MAX_ENDGAME_CONTRIB)  # faster than *256


    # -------------------------------------------------
    # MATERIAL + PST EVALUATION (INT64 SAFE)
    # -------------------------------------------------
    def piece_based_evals(self, endgame_weight):
        score = np.int64(0)
        inv_weight = np.int32(256 - endgame_weight)

        for piece in range(12):
            bb = self.board.bitboard[piece]
            value = np.int64(PIECE_VALUES[piece])

            while bb:
                sq = get_lsb1_index(bb)

                mg = np.int64(pst_eval(piece, sq, 0))
                eg = np.int64(pst_eval(piece, sq, 1))

                blended = (inv_weight * mg + endgame_weight * eg) >> 8
                score += value + blended

                bb = pop_bit(bb, sq)

        return score


    # -------------------------------------------------
    # FINAL EVALUATION
    # -------------------------------------------------
    def evaluate(self):
        endgame_weight = self.endgame_score_calc()
        return self.piece_based_evals(endgame_weight)
