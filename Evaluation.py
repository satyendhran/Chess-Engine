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

ENDGAME_CONTRIB = np.array(
    [0, 1, 1, 2, 4, 0, 0, 1, 1, 2, 4, 0],
    dtype=np.int32,
)

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
    [0, 8, 15, 9, -6, -2, 0, -8, -15, -9, 6, 2],
    dtype=np.int32,
)

MOBILITY_CONTRIB_END = np.array(
    [0, 6, 11, 15, 12, 15, 0, -6, -11, -15, -12, -15],
    dtype=np.int32,
)


ISOLATED_PAWN_MG = np.int32(-12)
ISOLATED_PAWN_EG = np.int32(-18)
DOUBLED_PAWN_MG = np.int32(-8)
DOUBLED_PAWN_EG = np.int32(-20)
BACKWARD_PAWN_MG = np.int32(-10)
BACKWARD_PAWN_EG = np.int32(-12)
PAWN_ISLAND_MG = np.int32(-8)
PAWN_ISLAND_EG = np.int32(-10)

PASSED_PAWN_MG = np.array([0, 5, 10, 20, 40, 70, 110, 0], dtype=np.int32)
PASSED_PAWN_EG = np.array([0, 10, 20, 40, 70, 120, 180, 0], dtype=np.int32)
CANDIDATE_PAWN_MG = np.array([0, 3, 6, 12, 24, 40, 0, 0], dtype=np.int32)
CANDIDATE_PAWN_EG = np.array([0, 5, 10, 20, 35, 55, 0, 0], dtype=np.int32)

PASSED_BLOCKADED_MG = np.int32(-15)
PASSED_BLOCKADED_EG = np.int32(-25)
PASSED_KING_DIST_EG = np.int32(5)
OUTSIDE_PASSED_EG = np.int32(35)


PAWN_SHIELD_MG = np.int32(25)
PAWN_SHIELD_MISSING_MG = np.int32(-35)
OPEN_FILE_KING_MG = np.int32(-40)
SEMI_OPEN_FILE_KING_MG = np.int32(-20)
KING_ATTACKER_WEIGHT = np.array([0, 20, 20, 30, 50, 0], dtype=np.int32)
KING_DEFENDER_BONUS = np.int32(8)
KING_DANGER_MAX = np.int32(400)


BISHOP_PAIR_MG = np.int32(30)
BISHOP_PAIR_EG = np.int32(50)
BAD_BISHOP_MG = np.int32(-3)
BAD_BISHOP_EG = np.int32(-4)

KNIGHT_OUTPOST_MG = np.int32(20)
KNIGHT_OUTPOST_EG = np.int32(15)
KNIGHT_RIM_MG = np.int32(-15)
KNIGHT_RIM_EG = np.int32(-10)

TRAPPED_MINOR_MG = np.int32(-150)
TRAPPED_MINOR_EG = np.int32(-100)
LOOSE_PIECE_MG = np.int32(-8)
LOOSE_PIECE_EG = np.int32(-5)


ROOK_OPEN_FILE_MG = np.int32(25)
ROOK_OPEN_FILE_EG = np.int32(15)
ROOK_SEMI_OPEN_MG = np.int32(12)
ROOK_SEMI_OPEN_EG = np.int32(8)
ROOK_ON_7TH_MG = np.int32(20)
ROOK_ON_7TH_EG = np.int32(30)
CONNECTED_ROOKS_MG = np.int32(10)
CONNECTED_ROOKS_EG = np.int32(15)
ROOK_BEHIND_PASSER_MG = np.int32(15)
ROOK_BEHIND_PASSER_EG = np.int32(25)


SPACE_BONUS_MG = np.int32(2)
CENTRAL_PAWN_MG = np.int32(15)
CENTRAL_PAWN_EG = np.int32(10)
CENTRAL_CONTROL_MG = np.int32(5)
CRAMPED_PENALTY_MG = np.int32(-5)


ACTIVE_ROOK_EG = np.int32(20)
OPPOSITION_EG = np.int32(15)


FILE_MASKS = np.array(
    [np.uint64(0x0101010101010101 << i) for i in range(8)], dtype=np.uint64
)
RANK_MASKS = np.array([np.uint64(0xFF << (i * 8)) for i in range(8)], dtype=np.uint64)
CENTRAL_SQUARES = np.uint64((1 << 27) | (1 << 28) | (1 << 35) | (1 << 36))
EXTENDED_CENTER = np.uint64(0x00003C3C3C3C0000)


ADJACENT_FILE_MASKS = np.zeros(8, dtype=np.uint64)
for f in range(8):
    mask = np.uint64(0)
    if f > 0:
        mask |= FILE_MASKS[f - 1]
    if f < 7:
        mask |= FILE_MASKS[f + 1]
    ADJACENT_FILE_MASKS[f] = mask


@njit(inline="always")
def shift_up(bb):
    return (bb << np.uint64(8)) & np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(inline="always")
def shift_down(bb):
    return bb >> np.uint64(8)


@njit(inline="always")
def distance(sq1, sq2):
    r1, f1 = sq1 >> 3, sq1 & 7
    r2, f2 = sq2 >> 3, sq2 & 7
    return max(abs(r1 - r2), abs(f1 - f2))


@njit(inline="always")
def manhattan_distance(sq1, sq2):
    r1, f1 = sq1 >> 3, sq1 & 7
    r2, f2 = sq2 >> 3, sq2 & 7
    return abs(r1 - r2) + abs(f1 - f2)


@njit(u8(u4, u4), inline="always")
def manhattan_center(x, y):
    ix = np.uint16(3 - x if x < 4 else x - 4)
    iy = np.uint16(3 - y if y < 4 else y - 4)
    return ix + iy


@njit(inline="always")
def same_quadrant(x1, y1, x2, y2):
    return (x1 >= 4) == (x2 >= 4) and (y1 >= 4) == (y2 >= 4)


spec = [("board", Board.class_type.instance_type)]


@jitclass(spec)
class Evaluation:
    def __init__(self, board):
        self.board = board

    def endgame_score_calc(self):
        """Calculate endgame weight (0-256)"""
        bb_arr = self.board.bitboard
        total = np.int32(0)
        for p in range(12):
            total += np.int32(count_bits(bb_arr[p])) * ENDGAME_CONTRIB[p]
        eg = MAX_ENDGAME_CONTRIB - total
        return np.uint8((eg * 256) // MAX_ENDGAME_CONTRIB)

    def get_piece_attacks(self, sq, piece, occ):
        """Get attacks for a piece"""
        if piece == 1 or piece == 7:
            return get_knight_attacks(sq)
        elif piece == 2 or piece == 8:
            return get_bishop_attacks(sq, occ)
        elif piece == 3 or piece == 9:
            return get_rook_attacks(sq, occ)
        elif piece == 4 or piece == 10:
            return get_queen_attacks(sq, occ)
        elif piece == 5 or piece == 11:
            return get_knight_attacks(sq)
        return np.uint64(0)

    def evaluate_all_pieces(self, eg_weight):
        """
        Consolidated piece evaluation - material, mobility, PST,
        piece quality, and rook-specific features in ONE PASS
        """
        bb_arr = self.board.bitboard
        occ_all = self.board.occupancy[2]

        inv_weight = np.int64(256 - eg_weight)
        e_weight = np.int64(eg_weight)

        score_material = np.int64(0)
        score_mg = np.int32(0)
        score_eg = np.int32(0)

        w_bishop_count = count_bits(bb_arr[2])
        b_bishop_count = count_bits(bb_arr[8])

        w_rook_sqs = []
        b_rook_sqs = []

        w_pawns = bb_arr[0]
        b_pawns = bb_arr[6]

        w_defended = np.uint64(0)
        b_defended = np.uint64(0)

        for piece in range(12):
            bb = bb_arr[piece]
            if bb == 0:
                continue

            is_white = piece < 6
            piece_type = piece if is_white else piece - 6

            pvalue = np.int64(PIECE_VALUES[piece])
            mid = np.int64(MOBILITY_CONTRIB_MID[piece])
            endv = np.int64(MOBILITY_CONTRIB_END[piece])
            mob_scale = (inv_weight * mid + e_weight * endv) >> 8

            while bb:
                sq = get_lsb1_index(bb)
                rank = sq >> 3
                file = sq & 7

                mg = np.int64(pst_eval(piece, sq, 0))
                eg = np.int64(pst_eval(piece, sq, 1))
                blended = ((inv_weight * mg) + (e_weight * eg)) >> 8
                score_material += pvalue + blended

                attacks = self.get_piece_attacks(sq, piece, occ_all)
                mob = count_bits(attacks)
                score_material += mob * mob_scale

                if is_white:
                    w_defended |= attacks
                else:
                    b_defended |= attacks

                if piece_type == 1:

                    if file == 0 or file == 7 or rank == 0 or rank == 7:
                        score_mg += KNIGHT_RIM_MG if is_white else -KNIGHT_RIM_MG
                        score_eg += KNIGHT_RIM_EG if is_white else -KNIGHT_RIM_EG

                    if is_white and 3 <= rank <= 5:
                        adj_m = ADJACENT_FILE_MASKS[file]
                        ahead = np.uint64(0xFFFFFFFFFFFFFFFF) >> np.uint64(
                            (7 - rank + 1) * 8
                        )
                        if (b_pawns & adj_m & ahead) == 0:
                            behind_ranks = (
                                RANK_MASKS[rank - 1] if rank > 0 else np.uint64(0)
                            )
                            if (w_pawns & adj_m & behind_ranks) != 0:
                                score_mg += KNIGHT_OUTPOST_MG
                                score_eg += KNIGHT_OUTPOST_EG
                    elif not is_white and 2 <= rank <= 4:
                        adj_m = ADJACENT_FILE_MASKS[file]
                        ahead = np.uint64(0xFFFFFFFFFFFFFFFF) << np.uint64(
                            (rank + 1) * 8
                        )
                        if (w_pawns & adj_m & ahead) == 0:
                            behind_ranks = (
                                RANK_MASKS[rank + 1] if rank < 7 else np.uint64(0)
                            )
                            if (b_pawns & adj_m & behind_ranks) != 0:
                                score_mg -= KNIGHT_OUTPOST_MG
                                score_eg -= KNIGHT_OUTPOST_EG

                    free_moves = count_bits(attacks & ~occ_all)
                    if free_moves <= 1:
                        score_mg += TRAPPED_MINOR_MG if is_white else -TRAPPED_MINOR_MG
                        score_eg += TRAPPED_MINOR_EG if is_white else -TRAPPED_MINOR_EG

                elif piece_type == 2:

                    free_moves = count_bits(attacks & ~occ_all)
                    if free_moves <= 2:
                        score_mg += TRAPPED_MINOR_MG if is_white else -TRAPPED_MINOR_MG
                        score_eg += TRAPPED_MINOR_EG if is_white else -TRAPPED_MINOR_EG

                    is_light = ((rank + file) & 1) == 1
                    blocked = np.int32(0)
                    pawn_bb = w_pawns if is_white else b_pawns
                    temp_bb = pawn_bb
                    while temp_bb:
                        p_sq = get_lsb1_index(temp_bb)
                        p_rank = p_sq >> 3
                        p_file = p_sq & 7
                        pawn_light = ((p_rank + p_file) & 1) == 1
                        if pawn_light == is_light and 2 <= p_file <= 5:
                            blocked += 1
                        temp_bb = pop_bit(temp_bb, p_sq)

                    bad_penalty = -blocked
                    score_mg += (
                        bad_penalty * BAD_BISHOP_MG
                        if is_white
                        else -bad_penalty * BAD_BISHOP_MG
                    )
                    score_eg += (
                        bad_penalty * BAD_BISHOP_EG
                        if is_white
                        else -bad_penalty * BAD_BISHOP_EG
                    )

                elif piece_type == 3:
                    fm = FILE_MASKS[file]
                    our_pawns = w_pawns if is_white else b_pawns
                    enemy_pawns = b_pawns if is_white else w_pawns

                    has_our = (our_pawns & fm) != 0
                    has_enemy = (enemy_pawns & fm) != 0

                    if not has_our and not has_enemy:
                        score_mg += (
                            ROOK_OPEN_FILE_MG if is_white else -ROOK_OPEN_FILE_MG
                        )
                        score_eg += (
                            ROOK_OPEN_FILE_EG if is_white else -ROOK_OPEN_FILE_EG
                        )
                    elif not has_our:
                        score_mg += (
                            ROOK_SEMI_OPEN_MG if is_white else -ROOK_SEMI_OPEN_MG
                        )
                        score_eg += (
                            ROOK_SEMI_OPEN_EG if is_white else -ROOK_SEMI_OPEN_EG
                        )

                    if (is_white and rank == 6) or (not is_white and rank == 1):
                        score_mg += ROOK_ON_7TH_MG if is_white else -ROOK_ON_7TH_MG
                        score_eg += ROOK_ON_7TH_EG if is_white else -ROOK_ON_7TH_EG

                        if eg_weight >= 180:
                            score_eg += ACTIVE_ROOK_EG if is_white else -ACTIVE_ROOK_EG

                    if is_white:
                        w_rook_sqs.append(sq)
                    else:
                        b_rook_sqs.append(sq)

                if 1 <= piece_type <= 4:
                    sq_bb = np.uint64(1) << np.uint64(sq)
                    if is_white:
                        if (sq_bb & w_defended) == 0:
                            score_mg += LOOSE_PIECE_MG
                            score_eg += LOOSE_PIECE_EG
                    else:
                        if (sq_bb & b_defended) == 0:
                            score_mg -= LOOSE_PIECE_MG
                            score_eg -= LOOSE_PIECE_EG

                bb = pop_bit(bb, sq)

        if w_bishop_count >= 2:
            score_mg += BISHOP_PAIR_MG
            score_eg += BISHOP_PAIR_EG
        if b_bishop_count >= 2:
            score_mg -= BISHOP_PAIR_MG
            score_eg -= BISHOP_PAIR_EG

        if len(w_rook_sqs) == 2:
            sq1, sq2 = w_rook_sqs[0], w_rook_sqs[1]
            r1, f1 = sq1 >> 3, sq1 & 7
            r2, f2 = sq2 >> 3, sq2 & 7
            if r1 == r2 or f1 == f2:
                between = np.uint64(0)
                if r1 == r2:
                    for f in range(min(f1, f2) + 1, max(f1, f2)):
                        between |= np.uint64(1) << np.uint64(r1 * 8 + f)
                else:
                    for r in range(min(r1, r2) + 1, max(r1, r2)):
                        between |= np.uint64(1) << np.uint64(r * 8 + f1)
                if (between & occ_all) == 0:
                    score_mg += CONNECTED_ROOKS_MG
                    score_eg += CONNECTED_ROOKS_EG

        if len(b_rook_sqs) == 2:
            sq1, sq2 = b_rook_sqs[0], b_rook_sqs[1]
            r1, f1 = sq1 >> 3, sq1 & 7
            r2, f2 = sq2 >> 3, sq2 & 7
            if r1 == r2 or f1 == f2:
                between = np.uint64(0)
                if r1 == r2:
                    for f in range(min(f1, f2) + 1, max(f1, f2)):
                        between |= np.uint64(1) << np.uint64(r1 * 8 + f)
                else:
                    for r in range(min(r1, r2) + 1, max(r1, r2)):
                        between |= np.uint64(1) << np.uint64(r * 8 + f1)
                if (between & occ_all) == 0:
                    score_mg -= CONNECTED_ROOKS_MG
                    score_eg -= CONNECTED_ROOKS_EG

        blended_piece_score = (
            (inv_weight * np.int64(score_mg)) + (e_weight * np.int64(score_eg))
        ) >> 8
        return score_material + blended_piece_score

    def evaluate_pawns_comprehensive(self, eg_weight):
        """
        Comprehensive pawn evaluation in ONE PASS:
        - Structure (isolated, doubled, backward, islands)
        - Passed pawns
        - Candidate pawns
        - Rook behind passer
        - Outside passers
        - Central pawns
        """
        bb_arr = self.board.bitboard
        w_pawns = bb_arr[0]
        b_pawns = bb_arr[6]
        w_king_sq = get_lsb1_index(bb_arr[5])
        b_king_sq = get_lsb1_index(bb_arr[11])

        inv_weight = np.int64(256 - eg_weight)
        e_weight = np.int64(eg_weight)

        score_mg = np.int32(0)
        score_eg = np.int32(0)

        w_passed_mask = np.uint64(0)
        b_passed_mask = np.uint64(0)

        w_island_count = np.int32(0)
        b_island_count = np.int32(0)
        prev_w_file_has_pawn = False
        prev_b_file_has_pawn = False

        all_pieces = self.board.occupancy[2]

        for file in range(8):
            fm = FILE_MASKS[file]
            adj_m = ADJACENT_FILE_MASKS[file]

            w_file_pawns = w_pawns & fm
            b_file_pawns = b_pawns & fm

            if w_file_pawns != 0:
                if not prev_w_file_has_pawn:
                    w_island_count += 1
                prev_w_file_has_pawn = True
            else:
                prev_w_file_has_pawn = False

            if b_file_pawns != 0:
                if not prev_b_file_has_pawn:
                    b_island_count += 1
                prev_b_file_has_pawn = True
            else:
                prev_b_file_has_pawn = False

            temp_w = w_file_pawns
            w_file_count = count_bits(temp_w)
            if w_file_count > 1:
                score_mg += DOUBLED_PAWN_MG
                score_eg += DOUBLED_PAWN_EG

            if w_file_pawns != 0 and (w_pawns & adj_m) == 0:
                score_mg += w_file_count * ISOLATED_PAWN_MG
                score_eg += w_file_count * ISOLATED_PAWN_EG

            while temp_w:
                sq = get_lsb1_index(temp_w)
                rank = sq >> 3

                ahead = np.uint64(0xFFFFFFFFFFFFFFFF) << np.uint64((rank + 1) * 8)
                if (b_pawns & (fm | adj_m) & ahead) == 0:
                    w_passed_mask |= np.uint64(1) << np.uint64(sq)
                    score_mg += PASSED_PAWN_MG[rank]
                    score_eg += PASSED_PAWN_EG[rank]

                    if rank < 7:
                        block_sq = sq + 8
                        if (all_pieces & (np.uint64(1) << np.uint64(block_sq))) != 0:
                            score_mg += PASSED_BLOCKADED_MG
                            score_eg += PASSED_BLOCKADED_EG

                    if eg_weight > 128:
                        our_dist = distance(sq, w_king_sq)
                        their_dist = distance(sq, b_king_sq)
                        score_eg += (their_dist - our_dist) * PASSED_KING_DIST_EG

                    if eg_weight >= 180 and (file <= 2 or file >= 5):
                        if manhattan_distance(sq, b_king_sq) > 4:
                            score_eg += OUTSIDE_PASSED_EG
                else:

                    if 2 <= rank <= 5:
                        if count_bits(b_pawns & (fm | adj_m) & ahead) <= count_bits(
                            w_pawns & adj_m & ahead
                        ):
                            score_mg += CANDIDATE_PAWN_MG[rank]
                            score_eg += CANDIDATE_PAWN_EG[rank]

                support_ranks = RANK_MASKS[rank - 1] if rank > 0 else np.uint64(0)
                has_support = (w_pawns & adj_m & support_ranks) != 0
                enemy_controls_advance = False
                if rank < 7:
                    adv_sq = sq + 8
                    enemy_controls_advance = (
                        get_pawn_attacks(adv_sq, 0) & b_pawns
                    ) != 0

                if not has_support and enemy_controls_advance:
                    if (w_pawns & adj_m & ahead) == 0:
                        score_mg += BACKWARD_PAWN_MG
                        score_eg += BACKWARD_PAWN_EG

                if sq == 27 or sq == 28 or sq == 35 or sq == 36:
                    score_mg += CENTRAL_PAWN_MG
                    score_eg += CENTRAL_PAWN_EG

                temp_w = pop_bit(temp_w, sq)

            temp_b = b_file_pawns
            b_file_count = count_bits(temp_b)
            if b_file_count > 1:
                score_mg -= DOUBLED_PAWN_MG
                score_eg -= DOUBLED_PAWN_EG

            if b_file_pawns != 0 and (b_pawns & adj_m) == 0:
                score_mg -= b_file_count * ISOLATED_PAWN_MG
                score_eg -= b_file_count * ISOLATED_PAWN_EG

            while temp_b:
                sq = get_lsb1_index(temp_b)
                rank = sq >> 3

                ahead = np.uint64(0xFFFFFFFFFFFFFFFF) >> np.uint64((7 - rank + 1) * 8)
                if (w_pawns & (fm | adj_m) & ahead) == 0:
                    b_passed_mask |= np.uint64(1) << np.uint64(sq)
                    black_rank = 7 - rank
                    score_mg -= PASSED_PAWN_MG[black_rank]
                    score_eg -= PASSED_PAWN_EG[black_rank]

                    if rank > 0:
                        block_sq = sq - 8
                        if (all_pieces & (np.uint64(1) << np.uint64(block_sq))) != 0:
                            score_mg -= PASSED_BLOCKADED_MG
                            score_eg -= PASSED_BLOCKADED_EG

                    if eg_weight > 128:
                        our_dist = distance(sq, b_king_sq)
                        their_dist = distance(sq, w_king_sq)
                        score_eg -= (their_dist - our_dist) * PASSED_KING_DIST_EG

                    if eg_weight >= 180 and (file <= 2 or file >= 5):
                        if manhattan_distance(sq, w_king_sq) > 4:
                            score_eg -= OUTSIDE_PASSED_EG
                else:
                    if 2 <= rank <= 5:
                        black_rank = 7 - rank
                        if count_bits(w_pawns & (fm | adj_m) & ahead) <= count_bits(
                            b_pawns & adj_m & ahead
                        ):
                            score_mg -= CANDIDATE_PAWN_MG[black_rank]
                            score_eg -= CANDIDATE_PAWN_EG[black_rank]

                support_ranks = RANK_MASKS[rank + 1] if rank < 7 else np.uint64(0)
                has_support = (b_pawns & adj_m & support_ranks) != 0
                enemy_controls_advance = False
                if rank > 0:
                    adv_sq = sq - 8
                    enemy_controls_advance = (
                        get_pawn_attacks(adv_sq, 1) & w_pawns
                    ) != 0

                if not has_support and enemy_controls_advance:
                    if (b_pawns & adj_m & ahead) == 0:
                        score_mg -= BACKWARD_PAWN_MG
                        score_eg -= BACKWARD_PAWN_EG

                if sq == 27 or sq == 28 or sq == 35 or sq == 36:
                    score_mg -= CENTRAL_PAWN_MG
                    score_eg -= CENTRAL_PAWN_EG

                temp_b = pop_bit(temp_b, sq)

        score_mg += w_island_count * PAWN_ISLAND_MG
        score_eg += w_island_count * PAWN_ISLAND_EG
        score_mg -= b_island_count * PAWN_ISLAND_MG
        score_eg -= b_island_count * PAWN_ISLAND_EG

        w_rooks = bb_arr[3]
        b_rooks = bb_arr[9]

        temp_r = w_rooks
        while temp_r:
            r_sq = get_lsb1_index(temp_r)
            r_rank = r_sq >> 3
            r_file = r_sq & 7
            fm = FILE_MASKS[r_file]
            ahead = np.uint64(0xFFFFFFFFFFFFFFFF) << np.uint64((r_rank + 1) * 8)

            if (w_passed_mask & fm & ahead) != 0:
                score_mg += ROOK_BEHIND_PASSER_MG
                score_eg += ROOK_BEHIND_PASSER_EG

            temp_r = pop_bit(temp_r, r_sq)

        temp_r = b_rooks
        while temp_r:
            r_sq = get_lsb1_index(temp_r)
            r_rank = r_sq >> 3
            r_file = r_sq & 7
            fm = FILE_MASKS[r_file]
            ahead = np.uint64(0xFFFFFFFFFFFFFFFF) >> np.uint64((7 - r_rank + 1) * 8)

            if (b_passed_mask & fm & ahead) != 0:
                score_mg -= ROOK_BEHIND_PASSER_MG
                score_eg -= ROOK_BEHIND_PASSER_EG

            temp_r = pop_bit(temp_r, r_sq)

        return (
            (inv_weight * np.int64(score_mg)) + (e_weight * np.int64(score_eg))
        ) >> 8

    def evaluate_king_safety_and_activity(self, eg_weight):
        """Combined king safety (middlegame) and king activity (endgame)"""
        bb_arr = self.board.bitboard
        w_king_sq = get_lsb1_index(bb_arr[5])
        b_king_sq = get_lsb1_index(bb_arr[11])

        score = np.int32(0)

        if eg_weight < 160:
            w_danger = self.king_danger_score(w_king_sq, True)
            b_danger = self.king_danger_score(b_king_sq, False)
            mg_scale = np.int64(160 - eg_weight)
            safety_score = (b_danger - w_danger) * mg_scale >> 8
            score += safety_score if self.board.side == 0 else -safety_score

        if eg_weight >= 128:
            wx, wy = w_king_sq >> 3, w_king_sq & 7
            bx, by = b_king_sq >> 3, b_king_sq & 7

            dw = manhattan_center(wx, wy)
            db = manhattan_center(bx, by)

            dx = np.abs(wx - bx)
            dy = np.abs(wy - by)

            sign = 1 if self.board.side == 0 else -1
            activity = np.int32(0)

            if wx == bx and dy == 2:
                activity += sign * OPPOSITION_EG
            elif wy == by and dx == 2:
                activity += sign * OPPOSITION_EG
            elif dx == 2 and dy == 2:
                activity += sign * (OPPOSITION_EG >> 1)

            if dx == 0:
                activity += sign * (7 - dy) * (10 if dy & 1 else -10)
            if dy == 0:
                activity += sign * (7 - dx) * (10 if dx & 1 else -10)
            if dx == dy:
                activity += sign * (7 - dx) * (10 if dx & 1 else -10)

            if (dx == 1 and dy == 2) or (dx == 2 and dy == 1):
                activity += 5 if dw < db else -5

            if same_quadrant(wx, wy, bx, by):
                activity += 10 * (db - dw)

            if dw == 6:
                activity -= 20
            if db == 6:
                activity += 20

            score += (np.int64(activity) * eg_weight) >> 8

        return score

    def king_danger_score(self, king_sq, is_white):
        """Calculate king danger score"""
        bb_arr = self.board.bitboard
        danger = np.int32(0)

        file = king_sq & 7
        rank = king_sq >> 3

        if is_white:
            pawn_bb = bb_arr[0]
            shield_ranks = [rank + 1, rank + 2] if rank < 6 else [6, 7]
        else:
            pawn_bb = bb_arr[6]
            shield_ranks = [rank - 1, rank - 2] if rank > 1 else [0, 1]

        shield_files = [file]
        if file > 0:
            shield_files.append(file - 1)
        if file < 7:
            shield_files.append(file + 1)

        shield_count = 0
        for sf in shield_files:
            for sr in shield_ranks[:1]:
                if 0 <= sr <= 7:
                    sq = sr * 8 + sf
                    if (pawn_bb & (np.uint64(1) << np.uint64(sq))) != 0:
                        shield_count += 1
                        danger -= PAWN_SHIELD_MG

        missing_shields = len(shield_files) - shield_count
        danger += missing_shields * PAWN_SHIELD_MISSING_MG

        enemy_pawn_bb = bb_arr[6] if is_white else bb_arr[0]
        for sf in shield_files:
            fm = FILE_MASKS[sf]
            has_our_pawn = (pawn_bb & fm) != 0
            has_enemy_pawn = (enemy_pawn_bb & fm) != 0

            if not has_our_pawn and not has_enemy_pawn:
                danger += OPEN_FILE_KING_MG
            elif not has_our_pawn:
                danger += SEMI_OPEN_FILE_KING_MG

        king_ring = self.get_king_ring(king_sq)
        occ_all = self.board.occupancy[2]

        attacker_count = 0
        defender_count = 0
        enemy_start = 6 if is_white else 0
        our_start = 0 if is_white else 6

        for piece in range(1, 5):

            enemy_piece = enemy_start + piece
            bb = bb_arr[enemy_piece]
            while bb:
                sq = get_lsb1_index(bb)
                attacks = self.get_piece_attacks(sq, enemy_piece, occ_all)
                if (attacks & king_ring) != 0:
                    attacker_count += 1
                    danger += KING_ATTACKER_WEIGHT[piece]
                bb = pop_bit(bb, sq)

            our_piece = our_start + piece
            bb = bb_arr[our_piece]
            while bb:
                sq = get_lsb1_index(bb)
                attacks = self.get_piece_attacks(sq, our_piece, occ_all)
                if (attacks & king_ring) != 0:
                    defender_count += 1
                bb = pop_bit(bb, sq)

        danger -= defender_count * KING_DEFENDER_BONUS

        if danger > KING_DANGER_MAX:
            danger = KING_DANGER_MAX
        elif danger < -KING_DANGER_MAX:
            danger = -KING_DANGER_MAX

        return danger

    def get_king_ring(self, king_sq):
        """Get 5x5 king ring"""
        ring = np.uint64(0)
        rank = king_sq >> 3
        file = king_sq & 7

        for dr in range(-2, 3):
            for df in range(-2, 3):
                r = rank + dr
                f = file + df
                if 0 <= r <= 7 and 0 <= f <= 7:
                    sq = r * 8 + f
                    ring |= np.uint64(1) << np.uint64(sq)

        return ring

    def evaluate_space(self, eg_weight):
        """Evaluate space control and crampedness"""
        if eg_weight > 200:
            return np.int32(0)

        bb_arr = self.board.bitboard
        w_pawns = bb_arr[0]
        b_pawns = bb_arr[6]

        score_mg = np.int32(0)

        w_controlled = np.uint64(0)
        b_controlled = np.uint64(0)

        temp = w_pawns
        while temp:
            sq = get_lsb1_index(temp)
            w_controlled |= get_pawn_attacks(sq, 0)
            temp = pop_bit(temp, sq)

        temp = b_pawns
        while temp:
            sq = get_lsb1_index(temp)
            b_controlled |= get_pawn_attacks(sq, 1)
            temp = pop_bit(temp, sq)

        enemy_half_w = np.uint64(0xFFFFFFFF00000000)
        enemy_half_b = np.uint64(0x00000000FFFFFFFF)

        w_space = count_bits(w_controlled & enemy_half_w)
        b_space = count_bits(b_controlled & enemy_half_b)
        score_mg += (w_space - b_space) * SPACE_BONUS_MG

        occ_all = self.board.occupancy[2]

        w_knights = bb_arr[1]
        b_knights = bb_arr[7]
        w_bishops = bb_arr[2]
        b_bishops = bb_arr[8]

        w_center_control = 0
        b_center_control = 0

        temp = w_knights
        while temp:
            sq = get_lsb1_index(temp)
            attacks = get_knight_attacks(sq)
            w_center_control += count_bits(attacks & CENTRAL_SQUARES)
            temp = pop_bit(temp, sq)

        temp = b_knights
        while temp:
            sq = get_lsb1_index(temp)
            attacks = get_knight_attacks(sq)
            b_center_control += count_bits(attacks & CENTRAL_SQUARES)
            temp = pop_bit(temp, sq)

        temp = w_bishops
        while temp:
            sq = get_lsb1_index(temp)
            attacks = get_bishop_attacks(sq, occ_all)
            w_center_control += count_bits(attacks & EXTENDED_CENTER)
            temp = pop_bit(temp, sq)

        temp = b_bishops
        while temp:
            sq = get_lsb1_index(temp)
            attacks = get_bishop_attacks(sq, occ_all)
            b_center_control += count_bits(attacks & EXTENDED_CENTER)
            temp = pop_bit(temp, sq)

        score_mg += (w_center_control - b_center_control) * CENTRAL_CONTROL_MG

        w_cramped_ranks = np.uint64(0x0000000000FFFFFF)
        b_cramped_ranks = np.uint64(0xFFFFFF0000000000)

        w_cramped_count = 0
        b_cramped_count = 0

        for i in range(1, 6):
            w_cramped_count += count_bits(bb_arr[i] & w_cramped_ranks)
            b_cramped_count += count_bits(bb_arr[6 + i] & b_cramped_ranks)

        if w_cramped_count >= 4:
            score_mg += CRAMPED_PENALTY_MG
        if b_cramped_count >= 4:
            score_mg -= CRAMPED_PENALTY_MG

        mg_scale = np.int64(200 - eg_weight) if eg_weight < 200 else np.int64(0)
        return (np.int64(score_mg) * mg_scale) >> 8

    def evaluate(self):
        """Main evaluation function - optimized single pass"""
        eg_weight = self.endgame_score_calc()

        score = self.evaluate_all_pieces(eg_weight)
        score += self.evaluate_pawns_comprehensive(eg_weight)
        score += self.evaluate_king_safety_and_activity(eg_weight)
        score += self.evaluate_space(eg_weight)

        return score
