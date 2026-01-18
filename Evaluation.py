import numpy as np
from numba import int32, int64, njit, uint8, uint32, uint64
from numba.experimental import jitclass

from Board import Board
from Move_gen_pieces import (
    get_bishop_attacks,
    get_king_attacks,
    get_knight_attacks,
    get_pawn_attacks,
    get_queen_attacks,
    get_rook_attacks,
)
from pregen.Utilities import get_lsb1_index as lsb
from PST import pst_eval

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

KING_RINGS = np.zeros(64, dtype=np.uint64)
for king_sq in range(64):
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
    KING_RINGS[king_sq] = ring

DISTANCE_TABLE = np.zeros((64, 64), dtype=np.uint8)
MANHATTAN_TABLE = np.zeros((64, 64), dtype=np.uint8)
for sq1 in range(64):
    r1, f1 = sq1 >> 3, sq1 & 7
    for sq2 in range(64):
        r2, f2 = sq2 >> 3, sq2 & 7
        DISTANCE_TABLE[sq1, sq2] = max(abs(r1 - r2), abs(f1 - f2))
        MANHATTAN_TABLE[sq1, sq2] = abs(r1 - r2) + abs(f1 - f2)

PIECE_VALUES = np.array(
    [100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000],
    dtype=np.int32,
)
MOBILITY_BONUS_MG = np.array(
    [0, 4, 3, 2, 1, 0, 0, 4, 3, 2, 1, 0],
    dtype=np.int32,
)
MOBILITY_BONUS_EG = np.array(
    [0, 6, 5, 4, 3, 0, 0, 6, 5, 4, 3, 0],
    dtype=np.int32,
)

PASSED_PAWN_MG = np.array([0, 10, 15, 30, 60, 110, 180, 0], dtype=np.int32)
PASSED_PAWN_EG = np.array([0, 20, 40, 70, 120, 200, 300, 0], dtype=np.int32)
CANDIDATE_PAWN_MG = np.array([0, 5, 8, 15, 30, 50, 0, 0], dtype=np.int32)
CANDIDATE_PAWN_EG = np.array([0, 8, 15, 30, 50, 80, 0, 0], dtype=np.int32)

KING_CENTRALIZATION_EG = np.array(
    [
        -40,
        -30,
        -20,
        -10,
        -10,
        -20,
        -30,
        -40,
        -30,
        -20,
        -10,
        0,
        0,
        -10,
        -20,
        -30,
        -20,
        -10,
        5,
        15,
        15,
        5,
        -10,
        -20,
        -10,
        0,
        15,
        25,
        25,
        15,
        0,
        -10,
        -10,
        0,
        15,
        25,
        25,
        15,
        0,
        -10,
        -20,
        -10,
        5,
        15,
        15,
        5,
        -10,
        -20,
        -30,
        -20,
        -10,
        0,
        0,
        -10,
        -20,
        -30,
        -40,
        -30,
        -20,
        -10,
        -10,
        -20,
        -30,
        -40,
    ],
    dtype=np.int32,
)

KING_ATTACKER_WEIGHT = np.array([0, 30, 30, 50, 80, 0], dtype=np.int32)

CENTRAL_SQUARES_MASK = np.uint64((1 << 27) | (1 << 28) | (1 << 35) | (1 << 36))

SHIELD_FILE_OFFSETS = np.zeros((8, 3), dtype=np.int32)
SHIELD_FILE_COUNTS = np.zeros(8, dtype=np.int32)
for f in range(8):
    count = 0
    SHIELD_FILE_OFFSETS[f, count] = f
    count += 1
    if f > 0:
        SHIELD_FILE_OFFSETS[f, count] = f - 1
        count += 1
    if f < 7:
        SHIELD_FILE_OFFSETS[f, count] = f + 1
        count += 1
    SHIELD_FILE_COUNTS[f] = count

AHEAD_MASKS_WHITE = np.zeros(64, dtype=np.uint64)
AHEAD_MASKS_BLACK = np.zeros(64, dtype=np.uint64)
for sq in range(64):
    rank = sq >> 3
    if rank < 7:
        AHEAD_MASKS_WHITE[sq] = np.uint64(0xFFFFFFFFFFFFFFFF) << np.uint64(
            (rank + 1) * 8
        )
    if rank > 0:
        AHEAD_MASKS_BLACK[sq] = np.uint64(0xFFFFFFFFFFFFFFFF) >> np.uint64(
            (7 - rank + 1) * 8
        )

SQUARE_RANK = np.array([sq >> 3 for sq in range(64)], dtype=np.int32)
SQUARE_FILE = np.array([sq & 7 for sq in range(64)], dtype=np.int32)

IS_EDGE_SQUARE = np.array(
    [
        (sq >> 3) == 0 or (sq >> 3) == 7 or (sq & 7) == 0 or (sq & 7) == 7
        for sq in range(64)
    ],
    dtype=np.bool_,
)

IS_SEVENTH_RANK_WHITE = np.array([(sq >> 3) == 6 for sq in range(64)], dtype=np.bool_)
IS_SEVENTH_RANK_BLACK = np.array([(sq >> 3) == 1 for sq in range(64)], dtype=np.bool_)

CORNER_DISTANCE = np.array(
    [
        min(
            DISTANCE_TABLE[sq, 0],
            DISTANCE_TABLE[sq, 7],
            DISTANCE_TABLE[sq, 56],
            DISTANCE_TABLE[sq, 63],
        )
        for sq in range(64)
    ],
    dtype=np.uint8,
)

EDGE_DISTANCE = np.array(
    [min(sq >> 3, 7 - (sq >> 3), sq & 7, 7 - (sq & 7)) for sq in range(64)],
    dtype=np.uint8,
)

CENTER_SQUARES = np.array([27, 28, 35, 36], dtype=np.int32)


@njit(uint8(uint64), inline="always", fastmath=True, cache=True, nogil=True)
def popcount(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return uint8((x * 0x0101010101010101) >> 56)


@njit(uint64(uint64, uint64), inline="always", fastmath=True, cache=True, nogil=True)
def pop_bit_inline(bb, sq):
    return bb & ~(uint64(1) << sq)


@njit(uint32(uint64[:]), inline="always", fastmath=True, cache=True, nogil=True)
def compute_phase(bb_arr):
    total = (
        popcount(bb_arr[1])
        + popcount(bb_arr[2])
        + popcount(bb_arr[7])
        + popcount(bb_arr[8])
        + (popcount(bb_arr[3]) << 1)
        + (popcount(bb_arr[9]) << 1)
        + (popcount(bb_arr[4]) << 2)
        + (popcount(bb_arr[10]) << 2)
    )
    phase = uint32((total * 256) // 24)
    return min(phase, uint32(256))


@njit(inline="always", fastmath=True, cache=True, nogil=True)
def compute_attack_maps_parallel(bb_arr, occ_all):
    w_pawn_atks = uint64(0)
    b_pawn_atks = uint64(0)
    w_minor_atks = uint64(0)
    b_minor_atks = uint64(0)
    w_major_atks = uint64(0)
    b_major_atks = uint64(0)

    temp = bb_arr[0]
    while temp:
        sq = lsb(temp)
        w_pawn_atks |= get_pawn_attacks(sq, 0)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[6]
    while temp:
        sq = lsb(temp)
        b_pawn_atks |= get_pawn_attacks(sq, 1)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[1]
    while temp:
        sq = lsb(temp)
        w_minor_atks |= get_knight_attacks(sq)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[7]
    while temp:
        sq = lsb(temp)
        b_minor_atks |= get_knight_attacks(sq)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[2]
    while temp:
        sq = lsb(temp)
        w_minor_atks |= get_bishop_attacks(sq, occ_all)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[8]
    while temp:
        sq = lsb(temp)
        b_minor_atks |= get_bishop_attacks(sq, occ_all)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[3]
    while temp:
        sq = lsb(temp)
        w_major_atks |= get_rook_attacks(sq, occ_all)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[9]
    while temp:
        sq = lsb(temp)
        b_major_atks |= get_rook_attacks(sq, occ_all)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[4]
    while temp:
        sq = lsb(temp)
        w_major_atks |= get_queen_attacks(sq, occ_all)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[10]
    while temp:
        sq = lsb(temp)
        b_major_atks |= get_queen_attacks(sq, occ_all)
        temp = pop_bit_inline(temp, sq)

    return (
        w_pawn_atks,
        b_pawn_atks,
        w_minor_atks,
        b_minor_atks,
        w_major_atks,
        b_major_atks,
    )


@njit(
    int64(int32, uint64[:], uint64, uint64, uint64, uint64, uint64, uint64),
    inline="always",
    fastmath=True,
    cache=True,
    nogil=True,
)
def eval_piece_safety(
    phase,
    bb_arr,
    w_all_atks,
    b_all_atks,
    w_pawn_atks,
    b_pawn_atks,
    w_minor_atks,
    b_minor_atks,
):
    inv_phase = 256 - phase
    score = int64(0)
    shift_val = 8

    for piece in range(1, 5):
        temp = bb_arr[piece]
        while temp:
            sq = lsb(temp)
            sq_bb = uint64(1) << sq
            attacked = (sq_bb & b_all_atks) != 0
            defended = (sq_bb & w_all_atks) != 0
            score_mg = int32(0)
            score_eg = int32(0)

            if attacked:
                pawn_attacked = (sq_bb & b_pawn_atks) != 0
                score_mg -= 30 * pawn_attacked
                score_eg -= 20 * pawn_attacked
                if not defended:
                    score_mg -= 100
                    score_eg -= 80
                elif piece >= 3:
                    minor_attacked = (sq_bb & b_minor_atks) != 0
                    score_mg -= 20 * minor_attacked
                    score_eg -= 15 * minor_attacked

            if defended and piece >= 2:
                pawn_defended = (sq_bb & w_pawn_atks) != 0
                score_mg += 10 * pawn_defended
                score_eg += 8 * pawn_defended
                if piece >= 3:
                    minor_defended = (sq_bb & w_minor_atks) != 0
                    score_mg += 5 * minor_defended
                    score_eg += 4 * minor_defended

            score += (
                (inv_phase * int64(score_mg)) + (phase * int64(score_eg))
            ) >> shift_val
            temp = pop_bit_inline(temp, sq)

    for piece in range(7, 11):
        temp = bb_arr[piece]
        piece_type = piece - 6
        while temp:
            sq = lsb(temp)
            sq_bb = uint64(1) << sq
            attacked = (sq_bb & w_all_atks) != 0
            defended = (sq_bb & b_all_atks) != 0
            score_mg = int32(0)
            score_eg = int32(0)

            if attacked:
                pawn_attacked = (sq_bb & w_pawn_atks) != 0
                score_mg += 30 * pawn_attacked
                score_eg += 20 * pawn_attacked
                if not defended:
                    score_mg += 100
                    score_eg += 80
                elif piece_type >= 3:
                    minor_attacked = (sq_bb & w_minor_atks) != 0
                    score_mg += 20 * minor_attacked
                    score_eg += 15 * minor_attacked

            if defended and piece_type >= 2:
                pawn_defended = (sq_bb & b_pawn_atks) != 0
                score_mg -= 10 * pawn_defended
                score_eg -= 8 * pawn_defended
                if piece_type >= 3:
                    minor_defended = (sq_bb & b_minor_atks) != 0
                    score_mg -= 5 * minor_defended
                    score_eg -= 4 * minor_defended

            score += (
                (inv_phase * int64(score_mg)) + (phase * int64(score_eg))
            ) >> shift_val
            temp = pop_bit_inline(temp, sq)

    return score



@njit(
    int64(int32, uint64[:], uint64, uint64, uint64),
    inline="always",
    fastmath=True,
    nogil=True,
)
def eval_pieces(phase, bb_arr, occ_all, w_pawn_atks, b_pawn_atks):
    inv_phase = int64(256 - phase)
    eg_phase = int64(phase)
    shift_val = 8
    score = int64(0)
    score_mg = int32(0)
    score_eg = int32(0)
    
    # Count material first
    for piece in range(12):
        if piece == 5 or piece == 11:  # Skip kings
            continue
        bb = bb_arr[piece]
        piece_count = popcount(bb)
        pvalue = int64(PIECE_VALUES[piece])
        
        if piece < 6:
            score += pvalue * int64(piece_count)
        else:
            score -= pvalue * int64(piece_count)
    
    w_bishop_count = popcount(bb_arr[2])
    b_bishop_count = popcount(bb_arr[8])

    # Mobility and positional evaluation
    for piece in range(12):
        bb = bb_arr[piece]
        if bb == 0 or piece == 5 or piece == 11:  # Skip empty and kings
            continue
            
        is_white = piece < 6
        piece_type = piece % 6
        
        if is_white:
            mid = int64(MOBILITY_BONUS_MG[piece])
            endv = int64(MOBILITY_BONUS_EG[piece])
        else:
            mid = -int64(MOBILITY_BONUS_MG[piece])
            endv = -int64(MOBILITY_BONUS_EG[piece])

        mob_scale = (inv_phase * mid + eg_phase * endv) >> shift_val
        enemy_pawn_atks = b_pawn_atks if is_white else w_pawn_atks

        while bb:
            sq = lsb(bb)

            pst_mg = int64(pst_eval(piece, sq, 0))
            pst_eg = int64(pst_eval(piece, sq, 1))
            score += (inv_phase * pst_mg + eg_phase * pst_eg) >> shift_val

            if piece_type < 5:
                attacks = (
                    get_pawn_attacks(sq, 0 if is_white else 1)
                    if piece_type == 0
                    else (
                        get_knight_attacks(sq)
                        if piece_type == 1
                        else (
                            get_bishop_attacks(sq, occ_all)
                            if piece_type == 2
                            else (
                                get_rook_attacks(sq, occ_all)
                                if piece_type == 3
                                else get_queen_attacks(sq, occ_all)
                            )
                        )
                    )
                )
                attack_targets = attacks & ~occ_all
                mob = popcount(attack_targets)
                score += int64(mob) * mob_scale

                if piece_type >= 1 and piece_type <= 4:
                    safe_targets = attack_targets & ~enemy_pawn_atks
                    safe_mob = popcount(safe_targets)
                    safe_scale = mob_scale >> 1
                    score += int64(safe_mob) * safe_scale

            bb = pop_bit_inline(bb, sq)

    # Bishop pair bonus
    w_has_pair = w_bishop_count >= 2
    b_has_pair = b_bishop_count >= 2
    score_mg += 40 * w_has_pair - 40 * b_has_pair
    score_eg += 65 * w_has_pair - 65 * b_has_pair
    blended_piece_score = (
        (inv_phase * int64(score_mg)) + (eg_phase * int64(score_eg))
    ) >> shift_val
    return score + blended_piece_score

@njit(
    int64(int32, uint64[:], uint64, uint64, uint64),
    inline="always",
    fastmath=True,
    nogil=True,
)
def eval_pawns(
    phase,
    bb_arr,
    occ_all,
    w_all_atks,
    b_all_atks,
):
    w_pawns = bb_arr[0]
    b_pawns = bb_arr[6]
    w_king_sq = lsb(bb_arr[5])
    b_king_sq = lsb(bb_arr[11])

    inv_phase = int64(256 - phase)
    eg_phase = int64(phase)
    shift = 8

    score_mg = int32(0)
    score_eg = int32(0)

    all_pieces = occ_all
    enemy_white = (
        bb_arr[6] | bb_arr[7] | bb_arr[8] | bb_arr[9] | bb_arr[10] | bb_arr[11]
    )
    enemy_black = bb_arr[0] | bb_arr[1] | bb_arr[2] | bb_arr[3] | bb_arr[4] | bb_arr[5]

    temp = w_pawns
    while temp:
        sq = lsb(temp)
        rank = SQUARE_RANK[sq]
        file = SQUARE_FILE[sq]

        forward = sq + 8 if rank < 7 else -1
        if forward != -1:
            free = (all_pieces & (uint64(1) << uint64(forward))) == 0
            score_mg += 4 * free

        if rank >= 3:
            score_mg += 6

        if rank == 1 or rank == 2:
            score_mg += 5

        atks = get_pawn_attacks(sq, 0)
        pressure = popcount(atks & enemy_white)
        score_mg += pressure * 12

        restrict = popcount(atks & b_all_atks)
        score_mg += restrict * 6

        score_eg += rank * 12

        temp = pop_bit_inline(temp, sq)

    temp = b_pawns
    while temp:
        sq = lsb(temp)
        rank = SQUARE_RANK[sq]
        file = SQUARE_FILE[sq]

        forward = sq - 8 if rank > 0 else -1
        if forward != -1:
            free = (all_pieces & (uint64(1) << uint64(forward))) == 0
            score_mg -= 4 * free

        if rank <= 4:
            score_mg -= 6

        if rank == 6 or rank == 5:
            score_mg -= 5

        atks = get_pawn_attacks(sq, 1)
        pressure = popcount(atks & enemy_black)
        score_mg -= pressure * 12

        restrict = popcount(atks & w_all_atks)
        score_mg -= restrict * 6

        score_eg -= (7 - rank) * 12

        temp = pop_bit_inline(temp, sq)

    return ((inv_phase * int64(score_mg)) + (eg_phase * int64(score_eg))) >> shift


@njit(
    int32(int32, int32, uint64[:], uint64), inline="always", fastmath=True, nogil=True
)
def king_danger(king_sq, is_white, bb_arr, occ_all):
    danger = int32(0)
    file = SQUARE_FILE[king_sq]
    rank = SQUARE_RANK[king_sq]
    pawn_bb = bb_arr[0] if is_white else bb_arr[6]

    if is_white:
        shield_rank1 = min(rank + 1, 6)
        shield_rank2 = min(rank + 2, 7)
    else:
        shield_rank1 = max(rank - 1, 1)
        shield_rank2 = max(rank - 2, 0)

    shield_count = 0
    num_files = SHIELD_FILE_COUNTS[file]

    for i in range(num_files):
        sf = SHIELD_FILE_OFFSETS[file, i]
        sq1 = shield_rank1 * 8 + sf
        has_shield = (pawn_bb & (uint64(1) << uint64(sq1))) != 0
        shield_count += has_shield
        danger -= 35 * has_shield

    missing = num_files - shield_count
    danger -= 50 * missing

    enemy_pawn_bb = bb_arr[6] if is_white else bb_arr[0]

    for i in range(num_files):
        sf = SHIELD_FILE_OFFSETS[file, i]
        fm = FILE_MASKS[sf]
        has_our = (pawn_bb & fm) != 0
        has_enemy = (enemy_pawn_bb & fm) != 0
        if not has_our:
            danger -= 60 if not has_enemy else 30

    king_ring = KING_RINGS[king_sq]
    attacker_count = 0
    defender_count = 0
    enemy_start = 6 if is_white else 0
    our_start = 0 if is_white else 6

    for piece in range(1, 5):
        temp = bb_arr[enemy_start + piece]
        while temp:
            sq = lsb(temp)
            if piece == 1:
                attacks = get_knight_attacks(sq)
            elif piece == 2:
                attacks = get_bishop_attacks(sq, occ_all)
            elif piece == 3:
                attacks = get_rook_attacks(sq, occ_all)
            else:
                attacks = get_queen_attacks(sq, occ_all)
            hits_ring = (attacks & king_ring) != 0
            attacker_count += hits_ring
            danger += KING_ATTACKER_WEIGHT[piece] * hits_ring
            temp = pop_bit_inline(temp, sq)

        temp = bb_arr[our_start + piece]
        while temp:
            sq = lsb(temp)
            if piece == 1:
                attacks = get_knight_attacks(sq)
            elif piece == 2:
                attacks = get_bishop_attacks(sq, occ_all)
            elif piece == 3:
                attacks = get_rook_attacks(sq, occ_all)
            else:
                attacks = get_queen_attacks(sq, occ_all)
            defender_count += (attacks & king_ring) != 0
            temp = pop_bit_inline(temp, sq)

    danger -= defender_count * 12

    if danger < -600:
        return int32(-600)
    if danger > 600:
        return int32(600)
    return danger


@njit(
    int32(int32, uint64[:], uint64, int32), inline="always", fastmath=True, nogil=True
)
def eval_king_safety(phase, bb_arr, occ_all, side):
    w_king_sq = lsb(bb_arr[5])
    b_king_sq = lsb(bb_arr[11])
    score = int32(0)

    if phase > 96:
        w_danger = king_danger(w_king_sq, True, bb_arr, occ_all)
        b_danger = king_danger(b_king_sq, False, bb_arr, occ_all)
        mg_scale = int64(phase - 96)
        score += ((b_danger - w_danger) * mg_scale) >> 8

        w_file = SQUARE_FILE[w_king_sq]
        b_file = SQUARE_FILE[b_king_sq]
        w_castled = (w_king_sq == 6) or (w_king_sq == 2)
        b_castled = (b_king_sq == 62) or (b_king_sq == 58)
        w_center = (not w_castled) and (3 <= w_file <= 4)
        b_center = (not b_castled) and (3 <= b_file <= 4)

        castling_bonus = (int64(80) * mg_scale) >> 8
        center_penalty = (int64(-60) * mg_scale) >> 8
        score += castling_bonus * (w_castled - b_castled)
        score += center_penalty * (w_center - b_center)

    if phase < 100:
        eg_scale = int64(256 - phase)
        w_central = (int64(KING_CENTRALIZATION_EG[w_king_sq]) * eg_scale) >> 8
        b_central = (int64(KING_CENTRALIZATION_EG[b_king_sq]) * eg_scale) >> 8
        score += w_central - b_central

        wx = SQUARE_RANK[w_king_sq]
        wy = SQUARE_FILE[w_king_sq]
        bx = SQUARE_RANK[b_king_sq]
        by = SQUARE_FILE[b_king_sq]

        dx = abs(wx - bx)
        dy = abs(wy - by)
        sign = 1 if side == 0 else -1
        activity = int32(0)

        same_file = (wx == bx) and (dy == 2)
        same_rank = (wy == by) and (dx == 2)
        diagonal = (dx == 2) and (dy == 2)
        activity += sign * (20 * (same_file or same_rank) + 10 * diagonal)

        if dx == 0:
            parity = -10 if (dy & 1) else 10
            activity += sign * (7 - dy) * parity
        if dy == 0:
            parity = -10 if (dx & 1) else 10
            activity += sign * (7 - dx) * parity
        if dx == dy:
            parity = -10 if (dx & 1) else 10
            activity += sign * (7 - dx) * parity

        score += (int64(activity) * eg_scale) >> 8

    return score


@njit(int64(int32, uint64[:], int32, int32), inline="always", fastmath=True, nogil=True)
def eval_lone_king_forced_loss(phase, bb_arr, w_king_sq, b_king_sq):
    if phase > 64:
        return int64(0)
    if bb_arr[0] != 0 or bb_arr[6] != 0:
        return int64(0)
    w_kn = popcount(bb_arr[1])
    w_bi = popcount(bb_arr[2])
    w_ro = popcount(bb_arr[3])
    w_q = popcount(bb_arr[4])
    b_kn = popcount(bb_arr[7])
    b_bi = popcount(bb_arr[8])
    b_ro = popcount(bb_arr[9])
    b_q = popcount(bb_arr[10])
    w_nonking = w_kn + w_bi + w_ro + w_q
    b_nonking = b_kn + b_bi + b_ro + b_q
    defender = int32(-1)
    attacker = int32(-1)
    if w_nonking == 0 and b_nonking > 0:
        if b_ro == 1 and b_kn == 0 and b_bi == 0 and b_q == 0:
            defender = int32(0)
            attacker = int32(1)
        elif b_q == 1 and b_kn == 0 and b_bi == 0 and b_ro == 0:
            defender = int32(0)
            attacker = int32(1)
        elif b_ro == 2 and b_kn == 0 and b_bi == 0 and b_q == 0:
            defender = int32(0)
            attacker = int32(1)
        elif b_q == 2 and b_kn == 0 and b_bi == 0 and b_ro == 0:
            defender = int32(0)
            attacker = int32(1)
        elif b_bi == 2 and b_kn == 0 and b_ro == 0 and b_q == 0:
            defender = int32(0)
            attacker = int32(1)
        elif b_kn == 1 and b_ro == 0 and b_bi == 0 and b_q == 0:
            defender = int32(0)
            attacker = int32(1)
    elif b_nonking == 0 and w_nonking > 0:
        if w_ro == 1 and w_kn == 0 and w_bi == 0 and w_q == 0:
            defender = int32(1)
            attacker = int32(0)
        elif w_q == 1 and w_kn == 0 and w_bi == 0 and w_ro == 0:
            defender = int32(1)
            attacker = int32(0)
        elif w_ro == 2 and w_kn == 0 and w_bi == 0 and w_q == 0:
            defender = int32(1)
            attacker = int32(0)
        elif w_q == 2 and w_kn == 0 and w_bi == 0 and w_ro == 0:
            defender = int32(1)
            attacker = int32(0)
        elif w_bi == 2 and w_kn == 0 and w_ro == 0 and w_q == 0:
            defender = int32(1)
            attacker = int32(0)
        elif w_kn == 1 and w_ro == 0 and w_bi == 0 and w_q == 0:
            defender = int32(1)
            attacker = int32(0)
    if defender == -1 or attacker == -1:
        return int64(0)
    lost_base = int32(-600)
    res_bonus = int32(0)
    mate_pressure = int32(0)
    def_is_white = defender == 0
    def_king = w_king_sq if def_is_white else b_king_sq
    att_king = b_king_sq if def_is_white else w_king_sq
    def_center = int32(8)
    att_center = int32(8)
    for i in range(4):
        c_sq = CENTER_SQUARES[i]
        d_def = int32(DISTANCE_TABLE[def_king, c_sq])
        d_att = int32(DISTANCE_TABLE[att_king, c_sq])
        if d_def < def_center:
            def_center = d_def
        if d_att < att_center:
            att_center = d_att
    center_pen = int32(12) * def_center
    mate_pressure += center_pen
    att_center_term = int32(8) - att_center
    if att_center_term > 0:
        mate_pressure += int32(4) * att_center_term
    kk_dist = int32(DISTANCE_TABLE[att_king, def_king])
    mate_pressure += int32(6) * kk_dist
    def_mob = popcount(get_king_attacks(def_king))
    if def_mob > 8:
        def_mob = uint8(8)
    lack_mob = int32(8) - int32(def_mob)
    mate_pressure += int32(20) * lack_mob
    att_rank = SQUARE_RANK[att_king]
    att_file = SQUARE_FILE[att_king]
    def_rank = SQUARE_RANK[def_king]
    def_file = SQUARE_FILE[def_king]
    opp = 0
    if att_rank == def_rank:
        df = att_file - def_file
        if df == 2 or df == -2:
            opp = 1
    if att_file == def_file:
        dr = att_rank - def_rank
        if dr == 2 or dr == -2:
            opp = 1
    if opp == 1:
        if def_center <= att_center:
            res_bonus += int32(25)
    att_coord = int32(0)
    if def_is_white:
        if w_ro + w_q + w_bi + w_kn > 0:
            if w_ro > 0:
                bb = bb_arr[3]
            elif w_q > 0:
                bb = bb_arr[4]
            elif w_bi > 0:
                bb = bb_arr[2]
            else:
                bb = bb_arr[1]
        else:
            bb = uint64(0)
    else:
        if b_ro + b_q + b_bi + b_kn > 0:
            if b_ro > 0:
                bb = bb_arr[9]
            elif b_q > 0:
                bb = bb_arr[10]
            elif b_bi > 0:
                bb = bb_arr[8]
            else:
                bb = bb_arr[7]
        else:
            bb = uint64(0)
    cutoff_active = 0
    cutoff_pen = int32(0)
    corner_pen = int32(0)
    if bb != 0:
        temp = bb
        best_coord = int32(0)
        best_cut = int32(0)
        while temp:
            p_sq = lsb(temp)
            pr = SQUARE_RANK[p_sq]
            pf = SQUARE_FILE[p_sq]
            d1 = int32(DISTANCE_TABLE[att_king, p_sq])
            d2 = int32(DISTANCE_TABLE[def_king, p_sq])
            c_val = 0
            v1 = int32(8) - d1
            v2 = int32(8) - d2
            if v1 > 0:
                c_val += v1
            if v2 > 0:
                c_val += v2
            if c_val > best_coord:
                best_coord = c_val
            file_cut = 0
            rank_cut = 0
            if pf > def_file and pf < att_file:
                file_cut = 1
            if pf < def_file and pf > att_file:
                file_cut = 1
            if pr > def_rank and pr < att_rank:
                rank_cut = 1
            if pr < def_rank and pr > att_rank:
                rank_cut = 1
            if file_cut == 1 or rank_cut == 1:
                cutoff_active = 1
                if file_cut == 1:
                    dist_line = abs(def_file - pf)
                else:
                    dist_line = abs(def_rank - pr)
                sup = int32(0)
                dkr = int32(DISTANCE_TABLE[att_king, p_sq])
                if dkr <= 3:
                    sup = int32(3) - dkr
                local_cut = int32(40) + int32(8) * int32(dist_line) + int32(4) * sup
                if local_cut > best_cut:
                    best_cut = local_cut
            temp = pop_bit_inline(temp, p_sq)
        att_coord = best_coord
        cutoff_pen = best_cut
    mate_pressure += int32(15) * att_coord
    if opp == 1:
        mate_pressure += int32(10)
    mate_pressure += cutoff_pen
    if CORNER_DISTANCE[def_king] == 0 and cutoff_active == 1:
        prox = int32(8) - int32(DISTANCE_TABLE[att_king, def_king])
        if prox < 0:
            prox = 0
        corner_pen = int32(80) + int32(15) * prox
        mate_pressure += corner_pen
    if res_bonus > int32(200):
        res_bonus = int32(200)
    final_def = int64(lost_base) + int64(res_bonus) - int64(mate_pressure)
    if def_is_white:
        return final_def
    else:
        return -final_def


@njit(
    int64(int32, uint64[:], uint64, uint64), inline="always", fastmath=True, nogil=True
)
def eval_basic_endgames(phase, bb_arr, w_king_sq, b_king_sq):
    if phase >= 32:
        return int64(0)
    special = eval_lone_king_forced_loss(
        phase, bb_arr, int32(w_king_sq), int32(b_king_sq)
    )
    if special != 0:
        return special
    score = int64(0)
    w_material = int32(0)
    b_material = int32(0)
    for i in range(1, 5):
        w_material += popcount(bb_arr[i])
        b_material += popcount(bb_arr[6 + i])
    w_has_pieces = w_material > 0
    b_has_pieces = b_material > 0
    if w_has_pieces and not b_has_pieces:
        ring = KING_RINGS[b_king_sq]
        corner_dist = int32(CORNER_DISTANCE[b_king_sq])
        edge_dist = int32(EDGE_DISTANCE[b_king_sq])
        king_dist = int32(DISTANCE_TABLE[w_king_sq, b_king_sq])
        score += int64(360 - corner_dist * 40)
        score += int64(220 - edge_dist * 28)
        score += int64(80 - king_dist * 10)
        ak_dist = int32(DISTANCE_TABLE[b_king_sq, w_king_sq])
        if ak_dist <= 6:
            score += int64(70 * (7 - ak_dist))
        ak_center = int32(EDGE_DISTANCE[w_king_sq])
        score += int64(ak_center * 10)
        ak_bb = uint64(1) << uint64(w_king_sq)
        if (ring & ak_bb) != 0:
            score += int64(200)
        if ak_center == 0 and ak_dist >= 3:
            score -= int64(90)
        b_mob = int32(popcount(get_king_attacks(int32(b_king_sq))))
        score += int64((8 - b_mob) * 14)
        wr = SQUARE_RANK[w_king_sq]
        wf = SQUARE_FILE[w_king_sq]
        br = SQUARE_RANK[b_king_sq]
        bf = SQUARE_FILE[b_king_sq]
        dr = wr - br
        if dr < 0:
            dr = -dr
        df = wf - bf
        if df < 0:
            df = -df
        if (wr == br and df == 2) or (wf == bf and dr == 2):
            score += int64(40)
        if wr == br and df >= 2:
            score += int64(20 * (8 - df))
        if wf == bf and dr >= 2:
            score += int64(20 * (8 - dr))
        for p in (3, 4):
            temp = bb_arr[p]
            while temp:
                sq = lsb(temp)
                sr = SQUARE_RANK[sq]
                sf = SQUARE_FILE[sq]
                cutoff = int32((sr == br) + (sf == bf))
                if cutoff > 0:
                    cd = int32(CORNER_DISTANCE[b_king_sq])
                    kd = int32(DISTANCE_TABLE[w_king_sq, b_king_sq])
                    score += int64(60 * cutoff + 20 * (7 - cd) + 20 * (7 - kd))
                temp = pop_bit_inline(temp, sq)
        for piece in range(1, 5):
            temp = bb_arr[piece]
            while temp:
                sq = lsb(temp)
                dist = int32(DISTANCE_TABLE[b_king_sq, sq])
                if dist <= 6:
                    score += int64(40 * (7 - dist))
                sq_bb = uint64(1) << uint64(sq)
                if (ring & sq_bb) != 0:
                    score += int64(120)
                center = int32(EDGE_DISTANCE[sq])
                score += int64(center * 10)
                if center == 0 and dist >= 3:
                    score -= int64(50)
                temp = pop_bit_inline(temp, sq)
        temp = bb_arr[0]
        while temp:
            sq = lsb(temp)
            dist = int32(DISTANCE_TABLE[b_king_sq, sq])
            if dist <= 6:
                score += int64(28 * (7 - dist))
            sq_bb = uint64(1) << uint64(sq)
            if (ring & sq_bb) != 0:
                score += int64(80)
            center = int32(EDGE_DISTANCE[sq])
            score += int64(center * 6)
            if center == 0 and dist >= 3:
                score -= int64(30)
            r = SQUARE_RANK[sq]
            score += int64(r * 4)
            temp = pop_bit_inline(temp, sq)
    elif b_has_pieces and not w_has_pieces:
        ring = KING_RINGS[w_king_sq]
        corner_dist = int32(CORNER_DISTANCE[w_king_sq])
        edge_dist = int32(EDGE_DISTANCE[w_king_sq])
        king_dist = int32(DISTANCE_TABLE[w_king_sq, b_king_sq])
        score -= int64(360 - corner_dist * 40)
        score -= int64(220 - edge_dist * 28)
        score -= int64(80 - king_dist * 10)
        ak_dist = int32(DISTANCE_TABLE[w_king_sq, b_king_sq])
        if ak_dist <= 6:
            score -= int64(70 * (7 - ak_dist))
        ak_center = int32(EDGE_DISTANCE[b_king_sq])
        score -= int64(ak_center * 10)
        ak_bb = uint64(1) << uint64(b_king_sq)
        if (ring & ak_bb) != 0:
            score -= int64(200)
        if ak_center == 0 and ak_dist >= 3:
            score += int64(90)
        w_mob = int32(popcount(get_king_attacks(int32(w_king_sq))))
        score -= int64((8 - w_mob) * 14)
        wr = SQUARE_RANK[w_king_sq]
        wf = SQUARE_FILE[w_king_sq]
        br = SQUARE_RANK[b_king_sq]
        bf = SQUARE_FILE[b_king_sq]
        dr = wr - br
        if dr < 0:
            dr = -dr
        df = wf - bf
        if df < 0:
            df = -df
        if (wr == br and df == 2) or (wf == bf and dr == 2):
            score -= int64(40)
        if wr == br and df >= 2:
            score -= int64(20 * (8 - df))
        if wf == bf and dr >= 2:
            score -= int64(20 * (8 - dr))
        for p in (9, 10):
            temp = bb_arr[p]
            while temp:
                sq = lsb(temp)
                sr = SQUARE_RANK[sq]
                sf = SQUARE_FILE[sq]
                cutoff = int32((sr == wr) + (sf == wf))
                if cutoff > 0:
                    cd = int32(CORNER_DISTANCE[w_king_sq])
                    kd = int32(DISTANCE_TABLE[w_king_sq, b_king_sq])
                    score -= int64(60 * cutoff + 20 * (7 - cd) + 20 * (7 - kd))
                temp = pop_bit_inline(temp, sq)
        for piece in range(7, 11):
            temp = bb_arr[piece]
            while temp:
                sq = lsb(temp)
                dist = int32(DISTANCE_TABLE[w_king_sq, sq])
                if dist <= 6:
                    score -= int64(40 * (7 - dist))
                sq_bb = uint64(1) << uint64(sq)
                if (ring & sq_bb) != 0:
                    score -= int64(120)
                center = int32(EDGE_DISTANCE[sq])
                score -= int64(center * 10)
                if center == 0 and dist >= 3:
                    score += int64(50)
                temp = pop_bit_inline(temp, sq)
        temp = bb_arr[6]
        while temp:
            sq = lsb(temp)
            dist = int32(DISTANCE_TABLE[w_king_sq, sq])
            if dist <= 6:
                score -= int64(28 * (7 - dist))
            sq_bb = uint64(1) << uint64(sq)
            if (ring & sq_bb) != 0:
                score -= int64(80)
            center = int32(EDGE_DISTANCE[sq])
            score -= int64(center * 6)
            if center == 0 and dist >= 3:
                score += int64(30)
            r = SQUARE_RANK[sq]
            score -= int64((7 - r) * 4)
            temp = pop_bit_inline(temp, sq)
    return score


@njit(fastmath=True)
def eval_space(phase, bb_arr, occ_all):
    if phase < 56:
        return np.int32(0)

    w_pawns = bb_arr[0]
    b_pawns = bb_arr[6]

    w_controlled = np.uint64(0)
    b_controlled = np.uint64(0)

    temp = w_pawns
    while temp:
        sq = lsb(temp)
        w_controlled |= get_pawn_attacks(sq, 0)
        temp = pop_bit_inline(temp, sq)

    temp = b_pawns
    while temp:
        sq = lsb(temp)
        b_controlled |= get_pawn_attacks(sq, 1)
        temp = pop_bit_inline(temp, sq)

    enemy_half_w = np.uint64(0xFFFFFFFF00000000)
    enemy_half_b = np.uint64(0x00000000FFFFFFFF)

    w_space = popcount(w_controlled & enemy_half_w)
    b_space = popcount(b_controlled & enemy_half_b)

    w_center_core = np.int32(0)
    b_center_core = np.int32(0)
    w_center_ext = np.int32(0)
    b_center_ext = np.int32(0)

    temp = bb_arr[1]
    while temp:
        sq = lsb(temp)
        w_center_core += popcount(get_knight_attacks(sq) & CENTRAL_SQUARES)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[7]
    while temp:
        sq = lsb(temp)
        b_center_core += popcount(get_knight_attacks(sq) & CENTRAL_SQUARES)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[2]
    while temp:
        sq = lsb(temp)
        w_center_ext += popcount(get_bishop_attacks(sq, occ_all) & EXTENDED_CENTER)
        temp = pop_bit_inline(temp, sq)

    temp = bb_arr[8]
    while temp:
        sq = lsb(temp)
        b_center_ext += popcount(get_bishop_attacks(sq, occ_all) & EXTENDED_CENTER)
        temp = pop_bit_inline(temp, sq)

    e_mg = (
        8 * (w_center_core - b_center_core)
        + 3 * (w_center_ext - b_center_ext)
        + 3 * (w_space - b_space)
    )

    w_space_val = w_space >> 2 if w_space > 0 else 0
    b_space_val = b_space >> 2 if b_space > 0 else 0

    w_f_mg = -((w_space_val * w_space_val) // 45)
    b_f_mg = -((b_space_val * b_space_val) // 45)
    e_mg += w_f_mg - b_f_mg

    w_f_eg = -(w_space_val // 5)
    b_f_eg = -(b_space_val // 5)

    e_eg = (
        4 * (w_center_core - b_center_core)
        + 2 * (w_center_ext - b_center_ext)
        + w_f_eg
        - b_f_eg
    )

    gamma = 2 * (w_center_core + b_center_core) + (w_center_ext + b_center_ext)
    abs_e_eg = e_eg if e_eg >= 0 else -e_eg
    boost = gamma if gamma >= abs_e_eg else abs_e_eg
    boost_sign = 1 if e_eg >= 0 else -1

    score_mg = e_mg
    score_eg = e_eg + boost_sign * boost

    w_cramped_ranks = np.uint64(0x0000000000FFFFFF)
    b_cramped_ranks = np.uint64(0xFFFFFF0000000000)

    w_cramped = 0
    b_cramped = 0

    for i in range(1, 6):
        w_cramped += popcount(bb_arr[i] & w_cramped_ranks)
        b_cramped += popcount(bb_arr[6 + i] & b_cramped_ranks)

    score_mg -= 8 * (w_cramped >= 4) - 8 * (b_cramped >= 4)

    mg_scale = np.int64(phase - 56)
    eg_scale = np.int64(160 - phase) if phase < 160 else np.int64(0)

    return ((np.int64(score_mg) * mg_scale) >> 8) + (
        (np.int64(score_eg) * eg_scale) >> 8
    )


spec = [("board", Board.class_type.instance_type)]


@jitclass(spec)
class Evaluation:
    def __init__(self, board):
        self.board = board

    def evaluate(self):
        bb_arr = self.board.bitboard
        occ_all = self.board.occupancy[2]
        side = self.board.side

        phase = compute_phase(bb_arr)

        (
            w_pawn_atks,
            b_pawn_atks,
            w_minor_atks,
            b_minor_atks,
            w_major_atks,
            b_major_atks,
        ) = compute_attack_maps_parallel(bb_arr, occ_all)

        w_all_atks = w_pawn_atks | w_minor_atks | w_major_atks
        b_all_atks = b_pawn_atks | b_minor_atks | b_major_atks

        score = eval_pieces(phase, bb_arr, occ_all, w_pawn_atks, b_pawn_atks)
        score += eval_piece_safety(
            phase,
            bb_arr,
            w_all_atks,
            b_all_atks,
            w_pawn_atks,
            b_pawn_atks,
            w_minor_atks,
            b_minor_atks,
        )
        score += eval_pawns(phase, bb_arr, occ_all, w_all_atks, b_all_atks)

        w_king_sq = lsb(bb_arr[5])
        b_king_sq = lsb(bb_arr[11])

        score += eval_king_safety(phase, bb_arr, occ_all, side)
        score += eval_basic_endgames(phase, bb_arr, w_king_sq, b_king_sq)
        score += eval_space(phase, bb_arr, occ_all)

        tempo = 15 if side == 0 else -15

        return score + tempo