from numba import i8, njit, u8,i4

from Board import Board, is_square_attacked
from Board_Move_gen import (
    Capture_generator,
    Move,
    Move_generator,
    get_capture_piece,
    get_starting_piece,
    get_target_square,
    unmove,
)
from Constants import Color, Pieces
from Evaluation import Evaluation
from Move_order import sort_moves
from pregen.Utilities import get_lsb1_index

P_K = int(Pieces.K)
P_k = int(Pieces.k)
COLOR_WHITE = int(Color.WHITE)

sig = i8(Board.class_type.instance_type, i8, i8, i8, i8, u8[:, :], u8[:, :],)


@njit
def minimax_max(board, depth, alpha, beta, root, killers, history):
    # pv_length[board.halfmove] = board.halfmove
    if board.halfmove == 50:
        return 0
    root_bool = root != 0
    mvs = Move_generator(board)
    sort_moves(mvs, board.halfmove)
    a, b, c = board.castle, board.enpassant, board.halfmove
    side = board.side
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
                score = minimax_min(board, depth - 1, alpha, beta, 0, killers, history)
            else:
                score = quiescence_max(board, alpha, beta, killers, history)
            unmove(board, mv, a, b, c)
            if root_bool:
                print(mv, score)
            if score > best_score:
                best_score = score
                best_move = mv
            if score > alpha:
                history[get_starting_piece(mv)][get_starting_piece(mv)] = score
                alpha = score

            if beta <= alpha:
                if get_capture_piece(mv) != 12:
                    killers[1][c] = killers[0][c]
                    killers[0][c] = mv
                break
    if legal_moves == 0:
        if side == COLOR_WHITE:
            king_bb = board.bitboard[P_K]
        else:
            king_bb = board.bitboard[P_k]
        king_sq = get_lsb1_index(king_bb)
        if is_square_attacked(board, king_sq, 1 - side):
            return -10000000000000 + depth
        else:
            return 0
    return best_move if root_bool else best_score


@njit
def minimax_min(board, depth, alpha, beta, root, killers, history):
    if board.halfmove == 50:
        return 0
    root_bool = root != 0
    mvs = Move_generator(board)
    sort_moves(mvs, board.halfmove)
    a, b, c = board.castle, board.enpassant, board.halfmove
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
                score = minimax_max(board, depth - 1, alpha, beta, 0, killers, history)
            else:
                score = quiescence_min(board, alpha, beta, killers, history)
            if root_bool:
                print(mv, score)
            unmove(board, mv, a, b, c)
            if score < best_score:
                best_score = score
                best_move = mv
            if score < beta:
                history[get_starting_piece(mv)][get_starting_piece(mv)] = score
                beta = score

            if beta <= alpha:
                if get_capture_piece(mv) != 12:
                    killers[1][c] = killers[0][c]
                    killers[0][c] = mv
                break
    if legal_moves == 0:
        if side == COLOR_WHITE:
            king_bb = board.bitboard[P_K]
        else:
            king_bb = board.bitboard[P_k]
        king_sq = get_lsb1_index(king_bb)
        if is_square_attacked(board, king_sq, 1 - side):
            return 10000000000000 - depth
        else:
            return 0
    return best_move if root_bool else best_score


@njit
def quiescence_max(board, alpha, beta, killers, history):
    stand_pat = Evaluation(board).evaluate()

    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    mvs = Capture_generator(board)
    sort_moves(mvs, board.halfmove)
    a, b, c = board.castle, board.enpassant, board.halfmove
    legal_moves = 0
    for i in range(mvs.counter):
        mv = mvs.moves[i]
        if mv == 0:
            continue
        if Move(board, mv):
            score = quiescence_min(board, alpha, beta, killers, history)
            unmove(board, mv, a, b, c)
            legal_moves += 1
            if score >= beta:
                return beta
            if score > alpha:
                history[get_starting_piece(mv)][get_starting_piece(mv)] = score
                alpha = score
    side = board.side
    if legal_moves == 0:
        if side == COLOR_WHITE:
            king_bb = board.bitboard[P_K]
        else:
            king_bb = board.bitboard[P_k]
        king_sq = get_lsb1_index(king_bb)
        if is_square_attacked(board, king_sq, 1 - side):
            return -10000000000000
        else:
            return 0

    return alpha


@njit
def quiescence_min(board, alpha, beta, killers, history):
    stand_pat = Evaluation(board).evaluate()

    if stand_pat <= alpha:
        return alpha
    if beta > stand_pat:
        beta = stand_pat

    mvs = Capture_generator(board)
    sort_moves(mvs, board.halfmove)
    a, b, c = board.castle, board.enpassant, board.halfmove
    legal_moves = 0
    for i in range(mvs.counter):

        mv = mvs.moves[i]
        if mv == 0:
            continue
        if Move(board, mv):
            score = quiescence_max(board, alpha, beta, killers, history)
            unmove(board, mv, a, b, c)
            legal_moves += 1
            if score <= alpha:
                return alpha
            if score < beta:
                history[get_starting_piece(mv)][get_starting_piece(mv)] = score
                beta = score
    side = board.side
    if legal_moves == 0:
        if side == COLOR_WHITE:
            king_bb = board.bitboard[P_K]
        else:
            king_bb = board.bitboard[P_k]
        king_sq = get_lsb1_index(king_bb)
        if is_square_attacked(board, king_sq, 1 - side):
            return 10000000000000
        else:
            return 0
    return beta


minimax_max.compile(sig)
minimax_min.compile(sig)
quiescence_max.compile(i8(Board.class_type.instance_type, i8, i8, u8[:, :], u8[:, :]))
quiescence_min.compile(i8(Board.class_type.instance_type, i8, i8, u8[:, :], u8[:, :]))
