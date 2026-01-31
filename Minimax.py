import time

import numpy as np
from numba import int16, int64, njit as _njit, uint8, uint64
from numba.experimental import jitclass

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
from Evaluation import evaluate_board, popcount
from Move_gen_pieces import (
    get_bishop_attacks,
    get_king_attacks,
    get_knight_attacks,
    get_pawn_attacks,
    get_queen_attacks,
    get_rook_attacks,
)
from pregen.Utilities import count_bits, get_lsb1_index, pop_bit

def njit(*args, **kwargs):
    kwargs["fastmath"] = True if "fastmath" not in kwargs else kwargs["fastmath"]
    kwargs["nogil"] = True if "nogil" not in kwargs else kwargs["nogil"]
    kwargs["boundscheck"] = False
    kwargs["error_model"] = "numpy"
    return _njit(*args, **kwargs)

@njit(inline="always")
def eval_inline(board):
    return evaluate_board(board)

PIECE_VALUE = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int32)
P_K = int(Pieces.K)
P_k = int(Pieces.k)
COLOR_WHITE = int(Color.WHITE)
COLOR_BLACK = 1 - COLOR_WHITE
MATE_SCORE = int64(100000000)
MATE_THRESHOLD = int64(90000000)
ASPIRATION_WINDOW = int64(50)
MAX_PLY = 64
MAX_SEARCH_DEPTH = 50
PIECE_VALUES = np.array(
    [100, 320, 330, 500, 900, 0, 100, 320, 330, 500, 900, 0, 0], dtype=np.int64
)
TT_SIZE = np.int64(1 << 26)
TT_EMPTY = np.uint8(255)

FUTILITY_MARGIN = np.array([0, 200, 350, 500], dtype=np.int64)
RFP_MARGIN = int64(75)
LMP_DEPTH = 6

PROBCUT_MARGIN = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)

spec = [
    ("keys", uint64[:]),
    ("depth", int16[:]),
    ("score", int64[:]),
    ("flags", uint8[:]),
    ("move", uint64[:]),
    ("size", int64),
    ("mask", int64),
]


@jitclass(spec)
class TranspositionTable:
    def __init__(self, size_mb):
        self.size = TT_SIZE
        self.mask = self.size - 1
        self.keys = np.zeros(self.size, dtype=np.uint64)
        self.depth = np.zeros(self.size, dtype=np.int16)
        self.score = np.zeros(self.size, dtype=np.int64)
        self.flags = np.full(self.size, TT_EMPTY, dtype=np.uint8)
        self.move = np.zeros(self.size, dtype=np.uint64)

    def probe(self, key, depth, alpha, beta):
        idx = int64(key & self.mask)
        if idx < 0 or idx >= self.size:
            return False, int64(0), int64(0), uint8(0), uint64(0)

        if self.keys[idx] != key or self.flags[idx] == TT_EMPTY:
            return False, int64(0), int64(0), uint8(0), uint64(0)

        d = self.depth[idx]
        s = self.score[idx]
        f = self.flags[idx]
        m = self.move[idx]

        if d >= depth and (f == 0 or (f == 1 and s >= beta) or (f == 2 and s <= alpha)):
            return True, d, s, f, m

        return False, d, s, f, m

    def store(self, key, depth, score, flag, move):
        idx = int64(key & self.mask)
        if idx < 0 or idx >= self.size:
            return

        if (
                self.flags[idx] == TT_EMPTY
                or depth >= self.depth[idx]
                or (flag == 0 and self.flags[idx] != 0)
        ):
            self.keys[idx] = key
            self.depth[idx] = np.int16(depth)
            self.score[idx] = score
            self.flags[idx] = np.uint8(flag)
            self.move[idx] = move

@njit(boundscheck=True, error_model="python")
def safe_array_access(arr, idx, default_val):
    if idx < 0 or idx >= len(arr):
        return default_val
    return arr[idx]


@njit(boundscheck=True, error_model="python")
def is_in_check(board, side):
    king_piece = P_K if side == COLOR_WHITE else P_k
    if king_piece < 0 or king_piece >= len(board.bitboard):
        return False
    king_bb = board.bitboard[king_piece]
    if king_bb == 0:
        return False
    king_sq = get_lsb1_index(king_bb)
    if king_sq < 0 or king_sq >= 64:
        return False
    return is_square_attacked(board, king_sq, 1 - side)


@njit(boundscheck=True, error_model="python")
def has_non_pawn_material(board, side):
    start, end = (1, 5) if side == COLOR_WHITE else (7, 11)
    if start < 0 or end > len(board.bitboard):
        return False
    for piece in range(start, end):
        if board.bitboard[piece] != 0:
            return True
    return False


@njit(boundscheck=True, error_model="python")
def count_total_material(board):
    total = 0
    for i in range(min(12, len(board.bitboard))):
        if i != 5 and i != 11:
            total += count_bits(board.bitboard[i])
    return total


@njit(boundscheck=True, error_model="python")
def is_basic_endgame(board):
    return count_total_material(board) <= 4


@njit(boundscheck=True, error_model="python")
def quick_phase(board):
    if len(board.bitboard) < 11:
        return 0
    bb = board.bitboard
    p = count_bits(bb[1]) + count_bits(bb[7]) + count_bits(bb[2]) + count_bits(bb[8])
    p += 2 * (count_bits(bb[3]) + count_bits(bb[9])) + 4 * (
            count_bits(bb[4]) + count_bits(bb[10])
    )
    return min(p, 24)


@njit(boundscheck=True, error_model="python")
def attackers_to_square(board, sq):
    if sq < 0 or sq >= 64:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), 0, 0
    occ, aw, bw, cw, cb = (
        board.occupancy[2],
        np.zeros(16, np.int32),
        np.zeros(16, np.int32),
        0,
        0,
    )
    for p in range(min(12, len(board.bitboard))):
        bb = board.bitboard[p]
        iter_count = 0
        while bb != 0 and iter_count < 64:
            iter_count += 1
            s = get_lsb1_index(bb)
            if s < 0 or s >= 64:
                break
            if p == 0:
                attacks = get_pawn_attacks(s, 0)
            elif p == 6:
                attacks = get_pawn_attacks(s, 1)
            elif p in (1, 7):
                attacks = get_knight_attacks(s)
            elif p in (2, 8):
                attacks = get_bishop_attacks(s, occ)
            elif p in (3, 9):
                attacks = get_rook_attacks(s, occ)
            elif p in (4, 10):
                attacks = get_queen_attacks(s, occ)
            else:
                attacks = get_king_attacks(s)
            if (attacks >> sq) & 1:
                piece_idx = p % 6
                if piece_idx < len(PIECE_VALUE):
                    val = PIECE_VALUE[piece_idx]
                    if p < 6 and cw < len(aw):
                        aw[cw], cw = val, cw + 1
                    elif cb < len(bw):
                        bw[cb], cb = val, cb + 1
            bb = pop_bit(bb, s)
    return aw, bw, cw, cb


@njit(boundscheck=True, error_model="python")
def see_gain(target_value, aw, bw, cw, cb, side):
    gains = np.zeros(32, np.int32)
    gains[0], iw, ib, g, turn = target_value, 0, 0, 0, side
    for _ in range(32):
        if turn == COLOR_WHITE:
            if iw >= cw or g + 1 >= len(gains) or iw >= len(aw):
                break
            gains[g + 1], iw = gains[g] - aw[iw], iw + 1
        else:
            if ib >= cb or g + 1 >= len(gains) or ib >= len(bw):
                break
            gains[g + 1], ib = gains[g] - bw[ib], ib + 1
        g, turn = g + 1, turn ^ 1
    for i in range(min(g, len(gains) - 1) - 1, -1, -1):
        if i + 1 < len(gains) and -gains[i + 1] > gains[i]:
            gains[i] = -gains[i + 1]
    return gains[0]


@njit(boundscheck=True, error_model="python")
def see_ge(board, mv, threshold):
    cap = get_capture_piece(mv)
    if cap == 12 or cap >= len(PIECE_VALUES):
        return cap == 12

    sq = get_target_square(mv)
    if sq < 0 or sq >= 64:
        return False

    attacker_piece = get_starting_piece(mv)
    if attacker_piece >= len(PIECE_VALUES):
        return False

    victim_value = PIECE_VALUES[cap]
    attacker_value = PIECE_VALUES[attacker_piece % 6]

    if victim_value >= attacker_value:
        return True

    if victim_value < attacker_value and threshold > int64(0):
        return False

    aw, bw, cw, cb = attackers_to_square(board, sq)

    if board.side == COLOR_WHITE:
        cw = remove_one(aw, cw, attacker_piece)
    else:
        cb = remove_one(bw, cb, attacker_piece)

    gain = victim_value
    side = 1 - board.side

    for depth in range(16):
        if side == COLOR_WHITE:
            if cw == 0:
                break
            p, cw = pop_least(aw, cw)
            if p >= len(PIECE_VALUES):
                break
            piece_val = PIECE_VALUES[p % 6]
            gain = gain - piece_val
            if gain < threshold:
                return False
        else:
            if cb == 0:
                break
            p, cb = pop_least(bw, cb)
            if p >= len(PIECE_VALUES):
                break
            piece_val = PIECE_VALUES[p % 6]
            gain = gain + piece_val

        side = 1 - side

    return gain >= threshold


@njit(boundscheck=True, error_model="python")
def pop_least(arr, count):
    if count <= 0 or len(arr) == 0:
        return np.uint8(12), 0
    min_idx, piece_idx = 0, int(arr[0]) % 12
    if piece_idx >= len(PIECE_VALUES):
        return np.uint8(12), count
    min_val = PIECE_VALUES[piece_idx]
    for i in range(1, min(count, len(arr))):
        piece_idx = int(arr[i]) % 12
        if piece_idx < len(PIECE_VALUES):
            v = PIECE_VALUES[piece_idx]
            if v < min_val:
                min_val, min_idx = v, i
    if min_idx >= len(arr):
        return np.uint8(12), count
    val = arr[min_idx]
    for i in range(min_idx, min(count - 1, len(arr) - 1)):
        arr[i] = arr[i + 1]
    return val, count - 1


@njit(boundscheck=True, error_model="python")
def remove_one(arr, count, piece):
    if count <= 0 or len(arr) == 0:
        return count
    idx = -1
    for i in range(min(count, len(arr))):
        if arr[i] == piece:
            idx = i
            break
    if idx == -1:
        return count
    for i in range(idx, min(count - 1, len(arr) - 1)):
        arr[i] = arr[i + 1]
    return count - 1


@njit(boundscheck=True, error_model="python")
def is_promotion_move(mv):
    promo = (mv >> 20) & 3
    return promo != 0


@njit()
def score_move(board, mv, ply, killers, history, tt_move, capture_history, is_endgame):
    if mv == tt_move:
        return int64(30000000)

    if is_promotion_move(mv):
        return int64(25000000)

    captured = get_capture_piece(mv)
    piece = get_starting_piece(mv)
    target = get_target_square(mv)

    endgame_king_pressure = int64(0)
    if is_endgame and piece < 13:

        opp_king_piece = P_k if board.side == COLOR_WHITE else P_K
        if opp_king_piece < len(board.bitboard):
            opp_king_bb = board.bitboard[opp_king_piece]
            if opp_king_bb != 0:
                opp_king_sq = get_lsb1_index(opp_king_bb)
                if opp_king_sq >= 0 and opp_king_sq < 64 and target >= 0 and target < 64:

                    king_rank = opp_king_sq >> 3
                    king_file = opp_king_sq & 7
                    target_rank = target >> 3
                    target_file = target & 7

                    rank_dist = abs(king_rank - target_rank)
                    file_dist = abs(king_file - target_file)
                    chebyshev_dist = max(rank_dist, file_dist)
                    manhattan_dist = rank_dist + file_dist

                    if chebyshev_dist == 0:
                        endgame_king_pressure = int64(50000)
                    elif chebyshev_dist == 1:
                        endgame_king_pressure = int64(30000)
                    elif chebyshev_dist == 2:
                        endgame_king_pressure = int64(15000)
                    elif chebyshev_dist == 3:
                        endgame_king_pressure = int64(6000)
                    elif chebyshev_dist == 4:
                        endgame_king_pressure = int64(2000)

                    piece_type = piece % 6
                    attacks_king_zone = False

                    if piece_type == 1:
                        attacks = get_knight_attacks(target)
                        if (attacks >> opp_king_sq) & 1:
                            endgame_king_pressure += int64(20000)
                            attacks_king_zone = True
                    elif piece_type == 2:
                        attacks = get_bishop_attacks(target, board.occupancy[2])
                        if (attacks >> opp_king_sq) & 1:
                            endgame_king_pressure += int64(25000)
                            attacks_king_zone = True
                    elif piece_type == 3:
                        attacks = get_rook_attacks(target, board.occupancy[2])
                        if (attacks >> opp_king_sq) & 1:
                            endgame_king_pressure += int64(30000)
                            attacks_king_zone = True
                    elif piece_type == 4:
                        attacks = get_queen_attacks(target, board.occupancy[2])
                        if (attacks >> opp_king_sq) & 1:
                            endgame_king_pressure += int64(40000)
                            attacks_king_zone = True

                    if attacks_king_zone:
                        king_attacks = get_king_attacks(opp_king_sq)
                        controlled_escape_squares = 0
                        temp_king_attacks = king_attacks
                        while temp_king_attacks:
                            escape_sq = get_lsb1_index(temp_king_attacks)
                            if (attacks >> escape_sq) & 1:
                                controlled_escape_squares += 1
                            temp_king_attacks = pop_bit(temp_king_attacks, escape_sq)

                        endgame_king_pressure += int64(controlled_escape_squares * 4000)

                    if piece_type == 3 or piece_type == 4:
                        if target_rank == king_rank or target_file == king_file:
                            endgame_king_pressure += int64(8000)

                    target_center_dist = max(abs(3 - target_rank), abs(3 - target_file))
                    if target_center_dist <= 2:
                        endgame_king_pressure += int64((2 - target_center_dist) * 1000)

    if captured != 12 and captured < 13:
        if piece >= 13:
            return int64(0)

        victim = PIECE_VALUES[captured] if captured < len(PIECE_VALUES) else int64(0)
        attacker = PIECE_VALUES[piece] if piece < len(PIECE_VALUES) else int64(0)

        score = int64(15000) + victim * int64(100) - attacker

        if is_endgame:
            score += endgame_king_pressure

        if attacker > victim:
            if not see_ge(board, mv, int64(-100 if is_endgame else 0)):
                score -= int64(8000)

                if is_endgame and endgame_king_pressure > int64(2000):
                    score += int64(4000)

        if (
                piece < 12
                and target < 64
                and piece < len(capture_history)
                and target < len(capture_history[0])
        ):
            score += capture_history[piece][target] >> 2

        return score
    else:

        score = int64(0)

        if is_endgame:
            score += endgame_king_pressure

        if ply < MAX_PLY and ply < len(killers[0]):
            if mv == killers[0][ply]:
                score += int64(9000)
            elif mv == killers[1][ply]:
                score += int64(8000)

        if (
                piece < 12
                and target < 64
                and piece < len(history)
                and target < len(history[0])
        ):
            score += history[piece][target]

        return score


@njit(boundscheck=True, error_model="python")
def insertion_sort_moves(mvs, scores, n):
    for i in range(1, n):
        key_score = scores[i]
        key_move = mvs.moves[i]

        j = i - 1
        while j >= 0 and scores[j] < key_score:
            scores[j + 1] = scores[j]
            mvs.moves[j + 1] = mvs.moves[j]
            j -= 1

        scores[j + 1] = key_score
        mvs.moves[j + 1] = key_move


@njit(boundscheck=True, error_model="python")
def sort_moves_with_scores(
        board, mvs, ply, killers, history, tt_move, capture_history, is_endgame
):
    if mvs.counter <= 1 or mvs.counter > len(mvs.moves):
        return

    if mvs.counter <= 5:
        for i in range(mvs.counter):
            if i >= len(mvs.moves):
                break
            score_i = score_move(
                board, mvs.moves[i], ply, killers, history,
                tt_move, capture_history, is_endgame
            ) if mvs.moves[i] != 0 else int64(-999999999)

            for j in range(i + 1, mvs.counter):
                if j >= len(mvs.moves):
                    break
                score_j = score_move(
                    board, mvs.moves[j], ply, killers, history,
                    tt_move, capture_history, is_endgame
                ) if mvs.moves[j] != 0 else int64(-999999999)

                if score_j > score_i:
                    mvs.moves[i], mvs.moves[j] = mvs.moves[j], mvs.moves[i]
                    score_i = score_j
        return

    scores = np.empty(mvs.counter, dtype=np.int64)
    for i in range(mvs.counter):
        if i >= len(mvs.moves):
            break
        scores[i] = (
            score_move(
                board,
                mvs.moves[i],
                ply,
                killers,
                history,
                tt_move,
                capture_history,
                is_endgame,
            )
            if mvs.moves[i] != 0
            else int64(-999999999)
        )
    insertion_sort_moves(mvs, scores, mvs.counter)


@njit(boundscheck=True, error_model="python")
def is_tactical_position(board):
    if is_in_check(board, board.side):
        return True

    occ = board.occupancy[2]
    our_start = 0 if board.side == COLOR_WHITE else 6
    enemy_start = 6 if board.side == COLOR_WHITE else 0

    for piece_type in range(1, 5):
        piece = our_start + piece_type
        if piece >= len(board.bitboard):
            continue
        bb = board.bitboard[piece]

        while bb != 0:
            sq = get_lsb1_index(bb)
            if sq < 0 or sq >= 64:
                break

            if is_square_attacked(board, sq, 1 - board.side):
                return True

            bb = pop_bit(bb, sq)

    return False


@njit(boundscheck=True, error_model="python")
def quiescence_negamax(
        board, ply, alpha, beta, killers, history, capture_history, stats, tt, depth_limit=10
):

    if len(stats) > 0:
        stats[0] += 1

    in_check = is_in_check(board, board.side)


    stand_pat = eval_inline(board)
    if len(stats) > 3:
        stats[3] += 1
    if board.side != COLOR_WHITE:
        stand_pat = -stand_pat

    if ply >= depth_limit:
        return stand_pat

    if not in_check:
        if stand_pat >= beta:
            return stand_pat

        if stand_pat > alpha:
            alpha = stand_pat

        delta = int64(900)
        if stand_pat + delta < alpha:
            if popcount(board.bitboard[0] & uint64(0xFF000000000000)) == 0 and \
                    popcount(board.bitboard[6] & uint64(0xFF00)) == 0:
                return alpha

    mvs = Move_generator(board) if in_check else Capture_generator(board)
    is_endgame = is_basic_endgame(board)
    sort_moves_with_scores(
        board, mvs, ply, killers, history, uint64(0), capture_history, is_endgame
    )

    a, b, c = board.castle, board.enpassant, board.halfmove
    best_score = stand_pat
    legal_count = 0

    for i in range(min(mvs.counter, len(mvs.moves))):
        mv = mvs.moves[i]

        if mv == 0:
            continue

        is_promo = is_promotion_move(mv)

        if not in_check and not is_promo:
            captured = get_capture_piece(mv)

            if captured < 7 or captured == 12:
                if not see_ge(board, mv, int64(-30)):
                    continue

        if not Move(board, mv):
            continue

        legal_count += 1

        score = -quiescence_negamax(
            board,
            ply + 1,
            -beta,
            -alpha,
            killers,
            history,
            capture_history,
            stats,
            tt,
            depth_limit,
        )
        unmove(board, mv, a, b, c)

        if score > best_score:
            best_score = score

        if score >= beta:
            return score
        if score > alpha:
            alpha = score

    if in_check and legal_count == 0:
        return -MATE_SCORE + int64(ply)

    return best_score


@njit(boundscheck=True, error_model="python")
def negamax(
        board,
        depth,
        ply,
        alpha,
        beta,
        killers,
        history,
        capture_history,
        allow_null,
        tt,
        stop_flag,
        game_history,
        stats,
):
    if len(stats) > 0:
        stats[0] += 1



    if ply > 0:

        reps = count_repetitions(game_history, board.hash, board.halfmove)
        if reps >= 2:
            return int64(0)

    if len(stop_flag) > 0 and stop_flag[0] != 0:
        return int64(0)



    if board.halfmove >= 100:
        return int64(0)

    if ply >= MAX_SEARCH_DEPTH:
        eval_score = eval_inline(board)
        if len(stats) > 3:
            stats[3] += 1
        if board.side != COLOR_WHITE:
            eval_score = -eval_score
        return eval_score

    alpha = max(alpha, -MATE_SCORE + int64(ply))
    beta = min(beta, MATE_SCORE - int64(ply) - int64(1))
    if alpha >= beta:
        return alpha

    is_pv_node = (beta - alpha) > int64(1)
    in_check = is_in_check(board, board.side)
    phase = quick_phase(board)
    is_endgame = phase <= 8
    is_basic_eg = is_basic_endgame(board)

    if in_check:
        depth += int64(1)

    if depth <= int64(0):
        return quiescence_negamax(
            board, ply, alpha, beta, killers, history, capture_history, stats, tt
        )

    tt_found, tt_d, tt_s, tt_f, tt_m = tt.probe(board.hash, depth, alpha, beta)
    if tt_found and tt_d >= depth and not is_pv_node:
        if len(stats) > 1:
            stats[1] += 1
        if (
                (tt_f == uint8(0))
                or (tt_f == uint8(1) and tt_s >= beta)
                or (tt_f == uint8(2) and tt_s <= alpha)
        ):
            if len(stats) > 2:
                stats[2] += 1
            return tt_s

    tt_move = tt_m
    static_eval = int64(0)

    if not in_check:
        if tt_found and tt_f != TT_EMPTY:
            static_eval = tt_s
        else:
            static_eval = eval_inline(board)
            if len(stats) > 3:
                stats[3] += 1
            if board.side != COLOR_WHITE:
                static_eval = -static_eval

    is_tactical = False
    if not in_check and not is_pv_node and not is_basic_eg:
        if depth <= int64(2) or depth >= int64(4):
            is_tactical = is_tactical_position(board)

    if (
            not is_pv_node
            and not in_check
            and not is_tactical
            and depth <= int64(2)
            and not is_endgame
            and not is_basic_eg
    ):
        margin = int64(200) if depth == int64(1) else int64(400)

        if static_eval + margin < alpha:
            qscore = quiescence_negamax(
                board,
                ply,
                alpha - margin,
                alpha - margin + int64(1),
                killers,
                history,
                capture_history,
                stats,
                tt,
            )
            if qscore + margin < alpha:
                return qscore

    if (
            allow_null
            and not is_pv_node
            and depth >= int64(4)
            and not in_check
            and not is_tactical
            and not is_basic_eg
            and has_non_pawn_material(board, board.side)
            and static_eval >= beta + int64(100)
    ):
        board.side, old_ep, board.enpassant = 1 - board.side, board.enpassant, 64

        R = int64(2) if depth >= int64(6) else int64(1)

        score = -negamax(
            board,
            depth - R - int64(1),
            ply + 1,
            -beta,
            -beta + int64(1),
            killers,
            history,
            capture_history,
            False,
            tt,
            stop_flag,
            game_history,
            stats,
        )
        board.side, board.enpassant = 1 - board.side, old_ep

        if score >= beta:
            if depth >= int64(12):
                verify = negamax(
                    board,
                    depth - R,
                    ply,
                    beta - int64(1),
                    beta,
                    killers,
                    history,
                    capture_history,
                    False,
                    tt,
                    stop_flag,
                    game_history,
                    stats,
                )
                if verify >= beta:
                    return beta
            else:
                return beta

    mvs = Move_generator(board)
    sort_moves_with_scores(
        board, mvs, ply, killers, history, tt_move, capture_history, is_basic_eg
    )


    a, b, c = board.castle, board.enpassant, board.halfmove
    legal_moves = 0
    best_score = -MATE_SCORE - int64(1)
    best_move = uint64(0)
    orig_alpha = alpha

    for i in range(min(mvs.counter, len(mvs.moves))):
        mv = mvs.moves[i]

        if mv == uint64(0) or not Move(board, mv):
            continue

        legal_moves += 1

        reduction = int64(0)
        is_capture = get_capture_piece(mv) != uint8(12)
        is_promo = is_promotion_move(mv)
        gives_check = False
        gives_check_known = False

        lmr_threshold = 8 if is_basic_eg else (6 if is_endgame else 4)

        if (
                not is_pv_node
                and depth >= int64(4)
                and legal_moves > lmr_threshold
                and not is_capture
                and not is_promo
                and not is_tactical
                and not is_basic_eg
        ):
            gives_check = is_in_check(board, 1 - board.side)
            gives_check_known = True
            if not gives_check:
                reduction = int64(1)
                threshold = 24 if is_endgame else 16
                if depth <= int64(LMP_DEPTH) and legal_moves > threshold and not is_basic_eg:
                    unmove(board, mv, a, b, c)
                    continue

        if (
                not is_pv_node
                and not in_check
                and depth <= int64(2)
                and not is_capture
                and not is_promo
                and not is_basic_eg
                and not is_endgame
        ):
            if not gives_check_known:
                gives_check = is_in_check(board, 1 - board.side)
                gives_check_known = True
            if not gives_check:
                margin_idx = int(depth)
                if margin_idx < len(FUTILITY_MARGIN):
                    if static_eval + FUTILITY_MARGIN[margin_idx] <= alpha:
                        unmove(board, mv, a, b, c)
                        continue

        if legal_moves == 1:
            score = -negamax(
                board,
                depth - int64(1) - reduction,
                ply + 1,
                -beta,
                -alpha,
                killers,
                history,
                capture_history,
                True,
                tt,
                stop_flag,
                game_history,
                stats,
            )
        else:
            score = -negamax(
                board,
                depth - int64(1) - reduction,
                ply + 1,
                -alpha - int64(1),
                -alpha,
                killers,
                history,
                capture_history,
                True,
                tt,
                stop_flag,
                game_history,
                stats,
            )

            if score > alpha and reduction > int64(0):
                score = -negamax(
                    board,
                    depth - int64(1),
                    ply + 1,
                    -alpha - int64(1),
                    -alpha,
                    killers,
                    history,
                    capture_history,
                    True,
                    tt,
                    stop_flag,
                    game_history,
                    stats,
                )

            if score > alpha and score < beta:
                score = -negamax(
                    board,
                    depth - int64(1),
                    ply + 1,
                    -beta,
                    -alpha,
                    killers,
                    history,
                    capture_history,
                    True,
                    tt,
                    stop_flag,
                    game_history,
                    stats,
                )

        unmove(board, mv, a, b, c)

        if score > best_score:
            best_score = score
            best_move = mv

            if score > alpha:
                alpha = score

                if score >= beta:
                    if get_capture_piece(mv) == uint8(12):
                        if ply >= 0 and ply < MAX_PLY and ply < len(killers[0]):
                            if mv != killers[0][ply]:
                                killers[1][ply] = killers[0][ply]
                                killers[0][ply] = mv

                        piece = get_starting_piece(mv)
                        target = get_target_square(mv)
                        if (
                                piece < 12
                                and target < 64
                                and piece < len(history)
                                and target < len(history[0])
                        ):
                            bonus = depth * depth
                            history[piece][target] += bonus

                            if history[piece][target] > int64(8000):
                                history[piece][target] = int64(8000)

                    tt.store(board.hash, depth, score, uint8(1), mv)
                    break

    if legal_moves == 0:
        return -MATE_SCORE + int64(ply) if in_check else int64(0)

    tt_flag = (
        uint8(1)
        if best_score >= beta
        else (uint8(2) if best_score <= orig_alpha else uint8(0))
    )
    tt.store(board.hash, depth, best_score, tt_flag, best_move)

    return best_score

@njit(boundscheck=True, error_model="python")
def count_repetitions(game_history, current_hash, halfmove):
    if halfmove < 4:
        return 0

    count = 0

    lookback = min(halfmove, len(game_history) - 1)

    for i in range(0, lookback, 2):
        if i >= len(game_history):
            break
        if game_history[len(game_history) - 1 - i] == current_hash:
            count += 1
            if count >= 2:
                return count

    return count

@njit(boundscheck=True, error_model="python")
def search_root(
        board, depth, killers, history, capture_history, tt, stop_flag, game_history, stats
):
    mvs = Move_generator(board)
    alpha, beta = -MATE_SCORE, MATE_SCORE

    tt_found, tt_d, tt_s, tt_f, tt_m = tt.probe(board.hash, depth, alpha, beta)
    is_basic_eg = is_basic_endgame(board)
    sort_moves_with_scores(
        board,
        mvs,
        0,
        killers,
        history,
        tt_m if tt_found else uint64(0),
        capture_history,
        is_basic_eg,
    )

    a, b, c = board.castle, board.enpassant, board.halfmove
    best_move = uint64(0)
    best_score = -MATE_SCORE - int64(1)

    for i in range(mvs.counter):
        mv = mvs.moves[i]

        if mv == uint64(0) or not Move(board, mv):
            continue

        score = -negamax(
            board,
            depth - int64(1),
            1,
            -beta,
            -alpha,
            killers,
            history,
            capture_history,
            True,
            tt,
            stop_flag,
            game_history,
            stats,
        )

        unmove(board, mv, a, b, c)

        if stop_flag[0] != 0:
            return uint64(0), int64(0)

        if score > best_score:
            best_score = score
            best_move = mv

            if score > alpha:
                alpha = score

    return best_move, best_score


@njit(boundscheck=True, error_model="python")
def search_with_aspiration(
        board,
        depth,
        prev_score,
        killers,
        history,
        capture_history,
        tt,
        stop_flag,
        game_history,
        stats,
):
    window = ASPIRATION_WINDOW
    alpha = max(prev_score - window, -MATE_SCORE)
    beta = min(prev_score + window, MATE_SCORE)

    best_move, best_score = search_root_internal(
        board, depth, alpha, beta, killers, history,
        capture_history, tt, stop_flag, game_history, stats
    )

    if stop_flag[0] != 0:
        return best_move, best_score

    if best_score <= alpha or best_score >= beta:

        for i in range(4):
            if best_score <= alpha:
                alpha = max(alpha - window * (2 ** i), -MATE_SCORE)
            if best_score >= beta:
                beta = min(beta + window * (2 ** i), MATE_SCORE)

            best_move, best_score = search_root_internal(
                board, depth, alpha, beta, killers, history,
                capture_history, tt, stop_flag, game_history, stats
            )

            if stop_flag[0] != 0:
                return best_move, best_score

            if alpha < best_score < beta:
                return best_move, best_score

        return search_root(
            board, depth, killers, history, capture_history,
            tt, stop_flag, game_history, stats
        )

    return best_move, best_score


@njit(boundscheck=True, error_model="python")
def search_root_internal(
        board, depth, alpha, beta, killers, history,
        capture_history, tt, stop_flag, game_history, stats
):
    mvs = Move_generator(board)
    tt_found, tt_d, tt_s, tt_f, tt_m = tt.probe(board.hash, depth, alpha, beta)
    is_basic_eg = is_basic_endgame(board)
    sort_moves_with_scores(
        board, mvs, 0, killers, history,
        tt_m if tt_found else uint64(0), capture_history, is_basic_eg
    )

    a, b, c = board.castle, board.enpassant, board.halfmove
    best_move = uint64(0)
    best_score = -MATE_SCORE - int64(1)

    for i in range(mvs.counter):
        mv = mvs.moves[i]
        if mv == uint64(0) or not Move(board, mv):
            continue

        score = -negamax(
            board, depth - int64(1), 1, -beta, -alpha,
            killers, history, capture_history, True,
            tt, stop_flag, game_history, stats
        )
        unmove(board, mv, a, b, c)

        if stop_flag[0] != 0:
            return uint64(0), int64(0)

        if score > best_score:
            best_score = score
            best_move = mv
            if score > alpha:
                alpha = score

    return best_move, best_score


class TimeManager:
    def __init__(self, time_limit, increment, movestogo):
        self.time_limit = time_limit
        self.increment = increment
        self.movestogo = movestogo if movestogo > 0 else 30
        self.allocated_time = (
            min(
                (time_limit + (self.movestogo - 1) * increment) / self.movestogo,
                time_limit * 0.8,
            )
            if time_limit > 0
            else 0
        )
        self.start_time = time.time()

    def is_expired(self):
        return (
                self.allocated_time > 0
                and (time.time() - self.start_time) >= self.allocated_time
        )


class SingleSearch:
    def __init__(
            self,
            board,
            depth=64,
            time_limit=0,
            increment=0,
            movestogo=30,
            game_history=None,
            info_callback=None,
    ):
        self.board = board
        self.depth = depth if depth > 0 else 64
        self.best_move = 0
        self.tt = TranspositionTable(32)
        self.stop_flag = np.array([0], dtype=np.int64)
        self.time_manager = TimeManager(time_limit, increment, movestogo)
        self.game_history = (
            game_history if game_history is not None else np.zeros(512, dtype=np.uint64)
        )
        self.info_callback = info_callback

    def search(self):
        killers = np.zeros((2, MAX_PLY), dtype=np.uint64)
        history = np.zeros((12, 64), dtype=np.int64)
        capture_history = np.zeros((12, 64), dtype=np.int64)
        best_score = 0

        for d in range(1, self.depth + 1):
            if d > 1:
                history //= 2
                capture_history //= 2
            if self.stop_flag[0]:
                break

            if self.time_manager.allocated_time > 0 and self.time_manager.is_expired():
                self.stop_flag[0] = 1
                break

            stats = np.zeros(5, dtype=np.int64)
            t0 = time.time()

            if d == 1:
                mv, score = search_root(
                    self.board,
                    d,
                    killers,
                    history,
                    capture_history,
                    self.tt,
                    self.stop_flag,
                    self.game_history,
                    stats,
                )
            else:
                mv, score = search_with_aspiration(
                    self.board,
                    d,
                    best_score,
                    killers,
                    history,
                    capture_history,
                    self.tt,
                    self.stop_flag,
                    self.game_history,
                    stats,
                )

            if self.stop_flag[0]:
                break

            best_score = score
            self.best_move = mv

            ps = score if self.board.side == COLOR_WHITE else -score
            ss = f"cp {ps}"
            if abs(ps) > MATE_THRESHOLD:
                mi = (MATE_SCORE - abs(ps) + 1) // 2
                if ps < 0:
                    mi = -mi
                ss = f"mate {mi}"

            msg = f"info depth {d} score {ss} pv {move_to_uci(mv)}"
            if self.info_callback:
                self.info_callback(msg)
            else:
                print(msg)

            elapsed = max(1e-6, time.time() - t0)
            nps = int(int(stats[0]) / elapsed)
            msg_stats = f"info stats depth {d} nodes {int(stats[0])} tt_hits {int(stats[1])} tt_cutoffs {int(stats[2])} evals {int(stats[3])} nps {nps} DEBUG {stats[4]}"
            if self.info_callback:
                self.info_callback(msg_stats)
            else:
                print(msg_stats)

        return self.best_move


def AI(board, side, depth):
    searcher = SingleSearch(board, depth)
    result = searcher.search()
    return result
