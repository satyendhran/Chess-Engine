from numba import i8, njit

from Board import Board, is_square_attacked
from Board_Move_gen import Move, Move_generator, unmove
from Constants import Color, Pieces
from Evaluation import Evaluation
from pregen.Utilities import get_lsb1_index

# ---------- Precompute integer constants (Numba-friendly) ----------
P_K = int(Pieces.K)
P_k = int(Pieces.k)
COLOR_WHITE = int(Color.WHITE)
# ------------------------------------------------------------------

sig = (Board.class_type.instance_type, i8, i8, i8, i8)

@njit
def minimax_max(board, depth, alpha, beta, root):
    root_bool = root != 0
    mvs = Move_generator(board)
    a, b = board.castle, board.enpassant
    side = board.side  # assume board.side is an int (0/1)

    legal_moves = 0
    best_score = -10000000
    best_move = 0

    for i in range(mvs.counter):
        mv = mvs.moves[i]
        if mv == 0:
            continue
        if Move(board, mv):
            legal_moves += 1
            if depth > 1:
                score = minimax_min(board, depth - 1, alpha, beta, 0)
            else:
                score = Evaluation(board).evaluate()
            unmove(board, mv, a, b)

            if score > best_score:
                best_score = score
                best_move = mv

            if score > alpha:
                alpha = score
            if beta <= alpha:
                break

    if legal_moves == 0:
        # use integer constants for indexing
        if side == COLOR_WHITE:
            king_bb = board.bitboard[P_K]
        else:
            king_bb = board.bitboard[P_k]
        king_sq = get_lsb1_index(king_bb)
        if is_square_attacked(board, king_sq, 1 - side):
            return -10000000 + depth
        else:
            return 0

    return best_move if root_bool else best_score


@njit
def minimax_min(board, depth, alpha, beta, root):
    root_bool = root != 0
    mvs = Move_generator(board)
    a, b = board.castle, board.enpassant
    side = board.side

    legal_moves = 0
    best_score = 10000000
    best_move = 0

    for i in range(mvs.counter):
        mv = mvs.moves[i]
        if mv == 0:
            continue
        if Move(board, mv):
            legal_moves += 1
            if depth > 1:
                score = minimax_max(board, depth - 1, alpha, beta, 0)
            else:
                score = Evaluation(board).evaluate()
            unmove(board, mv, a, b)

            if score < best_score:
                best_score = score
                best_move = mv

            if score < beta:
                beta = score
            if beta <= alpha:
                break

    if legal_moves == 0:
        if side == COLOR_WHITE:
            king_bb = board.bitboard[P_K]
        else:
            king_bb = board.bitboard[P_k]
        king_sq = get_lsb1_index(king_bb)
        if is_square_attacked(board, king_sq, 1 - side):
            return 10000000 - depth
        else:
            return 0

    return best_move if root_bool else best_score


# -------------------------------------------
# compile AFTER both functions exist (typed sig)
# -------------------------------------------
minimax_max.compile(sig)
minimax_min.compile(sig)
