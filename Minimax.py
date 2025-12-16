import numpy as np
from numba import int64, njit, uint64

from Board import is_square_attacked
from Board_Move_gen import (
    Capture_generator,
    Move,
    Move_generator,
    get_capture_piece,
    get_starting_piece,
    get_target_square,
    move_to_uci,
    unmove,
)
from Constants import Color, Pieces
from Evaluation import Evaluation
from pregen.Utilities import get_lsb1_index

P_K = int(Pieces.K)
P_k = int(Pieces.k)
COLOR_WHITE = int(Color.WHITE)

MATE_SCORE = int64(100000)
MATE_THRESHOLD = int64(90000)
ASPIRATION_WINDOW = int64(50)
MAX_PLY = 64


@njit
def is_in_check(board, side):
    if side == COLOR_WHITE:
        king_bb = board.bitboard[P_K]
    else:
        king_bb = board.bitboard[P_k]

    if king_bb == 0:
        return False

    king_sq = get_lsb1_index(king_bb)
    return is_square_attacked(board, king_sq, 1 - side)


@njit
def has_non_pawn_material(board, side):
    if side == COLOR_WHITE:
        for piece in range(1, 6):
            if board.bitboard[piece] != 0:
                return True
    else:
        for piece in range(7, 12):
            if board.bitboard[piece] != 0:
                return True
    return False


@njit
def score_move(mv, ply, killers, history):
    score = int64(0)
    captured = get_capture_piece(mv)

    if captured != 12:
        piece = get_starting_piece(mv)
        victim_value = int64([0, 3, 3, 5, 9, 0, 0, 3, 3, 5, 9, 0, 0][captured])
        attacker_value = int64([0, 3, 3, 5, 9, 0, 0, 3, 3, 5, 9, 0, 0][piece])
        score = 10000000 + victim_value * 10 - attacker_value
    else:
        if ply < MAX_PLY:
            if mv == killers[0][ply]:
                score = 9000000
            elif mv == killers[1][ply]:
                score = 8000000
            else:
                piece = get_starting_piece(mv)
                target = get_target_square(mv)
                if piece < 12 and target < 64:
                    score = history[piece][target]

    return score


@njit
def sort_moves_with_scores(mvs, ply, killers, history):
    scores = np.empty(mvs.counter, dtype=np.int64)
    for i in range(mvs.counter):
        if mvs.moves[i] != 0:
            scores[i] = score_move(mvs.moves[i], ply, killers, history)
        else:
            scores[i] = -999999999

    for i in range(1, mvs.counter):
        key_move = mvs.moves[i]
        key_score = scores[i]
        j = i - 1
        while j >= 0 and scores[j] < key_score:
            mvs.moves[j + 1] = mvs.moves[j]
            scores[j + 1] = scores[j]
            j -= 1
        mvs.moves[j + 1] = key_move
        scores[j + 1] = key_score


@njit
def negamax(board, depth, ply, alpha, beta, killers, history):
    if board.halfmove >= 50:
        return int64(0)

    alpha_mate = max(alpha, -MATE_SCORE + ply)
    beta_mate = min(beta, MATE_SCORE - ply - 1)
    if alpha_mate >= beta_mate:
        return alpha_mate

    is_pv_node = (beta - alpha) > 1
    in_check = is_in_check(board, board.side)

    if (
        not is_pv_node
        and depth >= 3
        and not in_check
        and has_non_pawn_material(board, board.side)
    ):

        board.side = 1 - board.side
        old_enpassant = board.enpassant
        board.enpassant = 0

        R = 3 if depth > 6 else 2
        score = -negamax(
            board, depth - 1 - R, ply + 1, -beta, -beta + 1, killers, history
        )

        board.side = 1 - board.side
        board.enpassant = old_enpassant

        if score >= beta:
            return beta

    mvs = Move_generator(board)
    sort_moves_with_scores(mvs, ply, killers, history)

    a = board.castle
    b = board.enpassant
    c = board.halfmove

    legal_moves = 0
    best_score = -MATE_SCORE - 1
    pv_found = False

    for i in range(mvs.counter):
        mv = mvs.moves[i]
        if mv == 0:
            continue

        if not Move(board, mv):
            continue

        legal_moves += 1

        if depth <= 1:
            score = -quiescence(board, ply + 1, -beta, -alpha, killers, history)
        else:
            if not pv_found:
                score = -negamax(
                    board, depth - 1, ply + 1, -beta, -alpha, killers, history
                )
            else:
                reduction = int64(0)
                is_quiet = get_capture_piece(mv) == 12

                if legal_moves > 4 and depth >= 3 and is_quiet and not in_check:
                    reduction = int64(1)
                    if legal_moves > 8 and depth >= 5:
                        reduction = int64(2)

                if reduction > 0:
                    score = -negamax(
                        board,
                        depth - 1 - reduction,
                        ply + 1,
                        -alpha - 1,
                        -alpha,
                        killers,
                        history,
                    )
                    if score > alpha:
                        score = -negamax(
                            board,
                            depth - 1,
                            ply + 1,
                            -alpha - 1,
                            -alpha,
                            killers,
                            history,
                        )
                else:
                    score = -negamax(
                        board, depth - 1, ply + 1, -alpha - 1, -alpha, killers, history
                    )

                if alpha < score < beta:
                    score = -negamax(
                        board, depth - 1, ply + 1, -beta, -alpha, killers, history
                    )

        unmove(board, mv, a, b, c)

        if score > best_score:
            best_score = score

            if score > alpha:
                alpha = score
                pv_found = True

                if get_capture_piece(mv) == 12:
                    piece = get_starting_piece(mv)
                    target = get_target_square(mv)
                    if piece < 12 and target < 64:
                        history[piece][target] += depth * depth

        if alpha >= beta:
            if get_capture_piece(mv) == 12 and ply < MAX_PLY:
                killers[1][ply] = killers[0][ply]
                killers[0][ply] = mv
            break

    if legal_moves == 0:
        if in_check:
            return -MATE_SCORE + ply
        else:
            return int64(0)

    return best_score


@njit
def quiescence(board, ply, alpha, beta, killers, history):
    stand_pat = Evaluation(board).evaluate()

    if board.side != COLOR_WHITE:
        stand_pat = -stand_pat

    if stand_pat >= beta:
        return beta

    if stand_pat > alpha:
        alpha = stand_pat

    BIG_DELTA = int64(900)
    if stand_pat + BIG_DELTA < alpha:
        return alpha

    mvs = Capture_generator(board)
    sort_moves_with_scores(mvs, ply, killers, history)

    a = board.castle
    b = board.enpassant
    c = board.halfmove

    legal_moves = 0

    for i in range(mvs.counter):
        mv = mvs.moves[i]
        if mv == 0:
            continue

        if not Move(board, mv):
            continue

        legal_moves += 1

        score = -quiescence(board, ply + 1, -beta, -alpha, killers, history)

        unmove(board, mv, a, b, c)

        if score >= beta:
            return beta

        if score > alpha:
            alpha = score
            piece = get_starting_piece(mv)
            target = get_target_square(mv)
            if piece < 12 and target < 64:
                history[piece][target] += 1

    if legal_moves == 0 and is_in_check(board, board.side):
        return -MATE_SCORE + ply

    return alpha


@njit
def search_root(board, depth, killers, history):
    mvs = Move_generator(board)
    sort_moves_with_scores(mvs, 0, killers, history)

    a = board.castle
    b = board.enpassant
    c = board.halfmove

    best_move = uint64(0)
    best_score = -MATE_SCORE - 1
    alpha = -MATE_SCORE
    beta = MATE_SCORE
    legal_moves = 0

    for i in range(mvs.counter):
        mv = mvs.moves[i]
        if mv == 0:
            continue

        if not Move(board, mv):
            continue

        legal_moves += 1

        if depth <= 1:
            score = -quiescence(board, 1, -beta, -alpha, killers, history)
        else:
            score = -negamax(board, depth - 1, 1, -beta, -alpha, killers, history)

        unmove(board, mv, a, b, c)

        if score > best_score:
            best_score = score
            best_move = mv

            if score > alpha:
                alpha = score

    if legal_moves == 0:
        return uint64(0)

    return best_move


@njit
def search_with_aspiration(board, depth, prev_score, killers, history):
    if depth <= 2:
        return search_root(board, depth, killers, history)

    window = ASPIRATION_WINDOW
    alpha = prev_score - window
    beta = prev_score + window

    mvs = Move_generator(board)
    sort_moves_with_scores(mvs, 0, killers, history)

    a = board.castle
    b = board.enpassant
    c = board.halfmove

    best_move = uint64(0)
    best_score = -MATE_SCORE - 1
    legal_moves = 0

    for i in range(mvs.counter):
        mv = mvs.moves[i]
        if mv == 0:
            continue

        if not Move(board, mv):
            continue

        legal_moves += 1

        score = -negamax(board, depth - 1, 1, -beta, -alpha, killers, history)

        unmove(board, mv, a, b, c)

        if score > best_score:
            best_score = score
            best_move = mv

            if score > alpha:
                alpha = score

    if best_score <= prev_score - window or best_score >= prev_score + window:
        return search_root(board, depth, killers, history)

    return best_move


def AI(board, color, depth):
    killers = np.zeros((2, MAX_PLY), dtype=np.uint64)
    history = np.zeros((12, 64), dtype=np.int64)

    print(
        f"\nSearching at depth {depth} for {'White' if color == COLOR_WHITE else 'Black'}"
    )

    board.side = color

    best_move = uint64(0)
    prev_score = int64(0)

    for d in range(1, depth + 1):
        print(f"\nDepth {d}...")

        if d <= 2:
            move = search_root(board, d, killers, history)
        else:
            move = search_with_aspiration(board, d, prev_score, killers, history)

        if move != 0:
            best_move = move
            print(f"Best move at depth {d}: {move_to_uci(best_move)}")
        else:
            print(f"No legal moves found at depth {d}")
            break

    return best_move
