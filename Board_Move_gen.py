import numpy as np
import tqdm
from numba import njit
from numba.experimental import jitclass
from numba.types import uint8 as u8  # type:ignore
from numba.types import uint32 as u32  # type:ignore

from Board import Board, is_square_attacked, parse_fen
from Constants import Castle, Color, Flag, Pieces, Square, square_to_coord
from Move_gen_pieces import (
    get_bishop_attacks,
    get_king_attacks,
    get_knight_attacks,
    get_pawn_attacks,
    get_queen_attacks,
    get_rook_attacks,
)
from pregen.Utilities import get_bit, get_lsb1_index, pop_bit, set_bit
from Zobrist import piece_keys,enpassant_keys,castle_keys,side_key

specs = [
    ("moves", u32[:]),
    ("counter", u8),
]


@njit(u8(u32), nogil=True)
def get_flag(move):
    return move >> 20


@njit(u8(u32), nogil=True)
def get_starting_piece(move):
    return (move >> 12) & 0b1111


@njit(u8(u32), nogil=True)
def get_capture_piece(move):
    return (move >> 16) & 0b1111


@njit
def get_start_square(move):
    return (move) & 0b111111


@njit
def get_target_square(move):
    return (move >> 6) & 0b111111


@jitclass(specs)  # type:ignore
class MoveList:
    def __init__(self):
        self.moves = np.zeros(218, dtype=np.uint32)
        self.counter = 0

    def add(self, move):
        self.moves[self.counter] = move
        self.counter += 1


@njit
def Move_maker(
    starting_square,
    end_square,
    flag,
    starting_piece,
    capture_piece,
):
    return (
        np.uint32(starting_square)
        | np.uint32(end_square) << 6
        | np.uint32(starting_piece) << 12
        | np.uint32(capture_piece) << 16
        | np.uint32(flag) << 20
    )


def move_to_uci(move):
    flag = get_flag(move)
    promo_piece = None
    if flag in (
        Flag.QUEEN_PROMOTION,
        Flag.CAPTURE_PROMOTION_QUEEN,
    ):
        promo_piece = "q"
    elif flag in (
        Flag.ROOK_PROMOTION,
        Flag.CAPTURE_PROMOTION_ROOK,
    ):
        promo_piece = "r"
    elif flag in (
        Flag.BISHOP_PROMOTION,
        Flag.CAPTURE_PROMOTION_BISHOP,
    ):
        promo_piece = "b"
    elif flag in (
        Flag.KNIGHT_PROMOTION,
        Flag.CAPTURE_PROMOTION_KNIGHT,
    ):
        promo_piece = "n"
    uci = (
        f"{square_to_coord[get_start_square(move)]}"
        f"{square_to_coord[get_target_square(move)]}"
        f"{promo_piece or ''}"
    )
    return uci


@njit()
def Move_generator(board: Board):
    moves = MoveList()

    if board.side == Color.WHITE:
        bitboard = board.bitboard[Pieces.P.value]
        while bitboard:
            source_square = get_lsb1_index(bitboard)
            target_square = source_square - 8
            if target_square >= 0 and not get_bit(
                board.occupancy[Color.BOTH.value],
                target_square,
            ):
                if Square.a7 <= source_square <= Square.h7:
                    moves.add(
                        Move_maker(
                            source_square,
                            target_square,
                            Flag.QUEEN_PROMOTION,
                            Pieces.P,
                            Pieces.NONE,
                        )
                    )
                    moves.add(
                        Move_maker(
                            source_square,
                            target_square,
                            Flag.ROOK_PROMOTION,
                            Pieces.P,
                            Pieces.NONE,
                        )
                    )
                    moves.add(
                        Move_maker(
                            source_square,
                            target_square,
                            Flag.BISHOP_PROMOTION,
                            Pieces.P,
                            Pieces.NONE,
                        )
                    )
                    moves.add(
                        Move_maker(
                            source_square,
                            target_square,
                            Flag.KNIGHT_PROMOTION,
                            Pieces.P,
                            Pieces.NONE,
                        )
                    )
                else:
                    moves.add(
                        Move_maker(
                            source_square,
                            target_square,
                            Flag.NONE,
                            Pieces.P,
                            Pieces.NONE,
                        )
                    )
                    if Square.a2 <= source_square <= Square.h2 and not get_bit(
                        board.occupancy[Color.BOTH.value],
                        target_square - 8,
                    ):
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square - 8,
                                Flag.DOUBLE_PUSH,
                                Pieces.P,
                                Pieces.NONE,
                            )
                        )
            attack_bitboard = get_pawn_attacks(
                source_square,
                Color.WHITE.value,
            )
            while attack_bitboard:
                target_square = get_lsb1_index(attack_bitboard)
                if get_bit(
                    board.occupancy[Color.BLACK.value],
                    target_square,
                ):
                    capture_piece = Pieces.NONE
                    for x in range(12):
                        if get_bit(
                            board.bitboard[x],
                            target_square,
                        ):
                            capture_piece = x
                            break
                    if Square.a7 <= source_square <= Square.h7:
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_QUEEN,
                                Pieces.P,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_ROOK,
                                Pieces.P,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_BISHOP,
                                Pieces.P,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_KNIGHT,
                                Pieces.P,
                                capture_piece,
                            )
                        )
                    else:
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE,
                                Pieces.P,
                                capture_piece,
                            )
                        )
                if target_square == board.enpassant:
                    if get_bit(
                        board.occupancy[Color.BLACK.value],
                        target_square + 8,
                    ):
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.ENPASSANT,
                                Pieces.P,
                                Pieces.p,
                            )
                        )
                attack_bitboard = pop_bit(
                    attack_bitboard,
                    target_square,
                )
            bitboard = pop_bit(bitboard, source_square)
    else:
        bitboard = board.bitboard[Pieces.p.value]
        while bitboard:
            source_square = get_lsb1_index(bitboard)
            target_square = source_square + 8
            if target_square < 64 and not get_bit(
                board.occupancy[Color.BOTH.value],
                target_square,
            ):
                if Square.a2 <= source_square <= Square.h2:
                    moves.add(
                        Move_maker(
                            source_square,
                            target_square,
                            Flag.QUEEN_PROMOTION,
                            Pieces.p,
                            Pieces.NONE,
                        )
                    )
                    moves.add(
                        Move_maker(
                            source_square,
                            target_square,
                            Flag.ROOK_PROMOTION,
                            Pieces.p,
                            Pieces.NONE,
                        )
                    )
                    moves.add(
                        Move_maker(
                            source_square,
                            target_square,
                            Flag.BISHOP_PROMOTION,
                            Pieces.p,
                            Pieces.NONE,
                        )
                    )
                    moves.add(
                        Move_maker(
                            source_square,
                            target_square,
                            Flag.KNIGHT_PROMOTION,
                            Pieces.p,
                            Pieces.NONE,
                        )
                    )
                else:
                    moves.add(
                        Move_maker(
                            source_square,
                            target_square,
                            Flag.NONE,
                            Pieces.p,
                            Pieces.NONE,
                        )
                    )
                    if Square.a7 <= source_square <= Square.h7 and not get_bit(
                        board.occupancy[Color.BOTH.value],
                        target_square + 8,
                    ):
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square + 8,
                                Flag.DOUBLE_PUSH,
                                Pieces.p,
                                Pieces.NONE,
                            )
                        )
            attack_bitboard = get_pawn_attacks(
                source_square,
                Color.BLACK.value,
            )
            while attack_bitboard:
                target_square = get_lsb1_index(attack_bitboard)
                if get_bit(
                    board.occupancy[Color.WHITE.value],
                    target_square,
                ):
                    capture_piece = Pieces.NONE
                    for x in range(12):
                        if get_bit(
                            board.bitboard[x],
                            target_square,
                        ):
                            capture_piece = x
                            break
                    if Square.a2 <= source_square <= Square.h2:
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_QUEEN,
                                Pieces.p,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_ROOK,
                                Pieces.p,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_BISHOP,
                                Pieces.p,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_KNIGHT,
                                Pieces.p,
                                capture_piece,
                            )
                        )
                    else:
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE,
                                Pieces.p,
                                capture_piece,
                            )
                        )
                if target_square == board.enpassant:
                    if get_bit(
                        board.occupancy[Color.WHITE.value],
                        target_square - 8,
                    ):
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.ENPASSANT,
                                Pieces.p,
                                Pieces.P,
                            )
                        )
                attack_bitboard = pop_bit(
                    attack_bitboard,
                    target_square,
                )
            bitboard = pop_bit(bitboard, source_square)

    knight_piece = Pieces.N.value if board.side == Color.WHITE else Pieces.n.value
    opp = Color.BLACK.value if board.side == Color.WHITE else Color.WHITE.value
    bitboard = board.bitboard[knight_piece]
    while bitboard:
        source_square = get_lsb1_index(bitboard)
        attack_bitboard = get_knight_attacks(source_square)
        while attack_bitboard:
            target_square = get_lsb1_index(attack_bitboard)
            if get_bit(
                board.occupancy[opp],
                target_square,
            ):
                capture_piece = Pieces.NONE
                for x in range(12):
                    if get_bit(
                        board.bitboard[x],
                        target_square,
                    ):
                        capture_piece = x
                        break
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.CAPTURE,
                        knight_piece,
                        capture_piece,
                    )
                )
            elif not get_bit(
                board.occupancy[Color.BOTH.value],
                target_square,
            ):
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.NONE,
                        knight_piece,
                        Pieces.NONE,
                    )
                )
            attack_bitboard = pop_bit(
                attack_bitboard,
                target_square,
            )
        bitboard = pop_bit(bitboard, source_square)

    bishop_piece = Pieces.B.value if board.side == Color.WHITE else Pieces.b.value
    bitboard = board.bitboard[bishop_piece]
    while bitboard:
        source_square = get_lsb1_index(bitboard)
        attack_bitboard = get_bishop_attacks(
            source_square,
            board.occupancy[Color.BOTH.value],
        )
        while attack_bitboard:
            target_square = get_lsb1_index(attack_bitboard)
            if get_bit(
                board.occupancy[opp],
                target_square,
            ):
                capture_piece = Pieces.NONE
                for x in range(12):
                    if get_bit(
                        board.bitboard[x],
                        target_square,
                    ):
                        capture_piece = x
                        break
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.CAPTURE,
                        bishop_piece,
                        capture_piece,
                    )
                )
            elif not get_bit(
                board.occupancy[Color.BOTH.value],
                target_square,
            ):
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.NONE,
                        bishop_piece,
                        Pieces.NONE,
                    )
                )
            attack_bitboard = pop_bit(
                attack_bitboard,
                target_square,
            )
        bitboard = pop_bit(bitboard, source_square)

    rook_piece = Pieces.R.value if board.side == Color.WHITE else Pieces.r.value
    bitboard = board.bitboard[rook_piece]
    while bitboard:
        source_square = get_lsb1_index(bitboard)
        attack_bitboard = get_rook_attacks(
            source_square,
            board.occupancy[Color.BOTH.value],
        )
        while attack_bitboard:
            target_square = get_lsb1_index(attack_bitboard)
            if get_bit(
                board.occupancy[opp],
                target_square,
            ):
                capture_piece = Pieces.NONE
                for x in range(12):
                    if get_bit(
                        board.bitboard[x],
                        target_square,
                    ):
                        capture_piece = x
                        break
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.CAPTURE,
                        rook_piece,
                        capture_piece,
                    )
                )
            elif not get_bit(
                board.occupancy[Color.BOTH.value],
                target_square,
            ):
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.NONE,
                        rook_piece,
                        Pieces.NONE,
                    )
                )
            attack_bitboard = pop_bit(
                attack_bitboard,
                target_square,
            )
        bitboard = pop_bit(bitboard, source_square)

    queen_piece = Pieces.Q.value if board.side == Color.WHITE else Pieces.q.value
    bitboard = board.bitboard[queen_piece]
    while bitboard:
        source_square = get_lsb1_index(bitboard)
        attack_bitboard = get_queen_attacks(
            source_square,
            board.occupancy[Color.BOTH.value],
        )
        while attack_bitboard:
            target_square = get_lsb1_index(attack_bitboard)
            if get_bit(
                board.occupancy[opp],
                target_square,
            ):
                capture_piece = Pieces.NONE
                for x in range(12):
                    if get_bit(
                        board.bitboard[x],
                        target_square,
                    ):
                        capture_piece = x
                        break
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.CAPTURE,
                        queen_piece,
                        capture_piece,
                    )
                )
            elif not get_bit(
                board.occupancy[Color.BOTH.value],
                target_square,
            ):
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.NONE,
                        queen_piece,
                        Pieces.NONE,
                    )
                )
            attack_bitboard = pop_bit(
                attack_bitboard,
                target_square,
            )
        bitboard = pop_bit(bitboard, source_square)

    king_piece = Pieces.K.value if board.side == Color.WHITE else Pieces.k.value
    bitboard = board.bitboard[king_piece]
    while bitboard:
        source_square = get_lsb1_index(bitboard)
        attack_bitboard = get_king_attacks(source_square)
        while attack_bitboard:
            target_square = get_lsb1_index(attack_bitboard)
            if get_bit(
                board.occupancy[opp],
                target_square,
            ):
                capture_piece = Pieces.NONE
                for x in range(12):
                    if get_bit(
                        board.bitboard[x],
                        target_square,
                    ):
                        capture_piece = x
                        break
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.CAPTURE,
                        king_piece,
                        capture_piece,
                    )
                )
            elif not get_bit(
                board.occupancy[Color.BOTH.value],
                target_square,
            ):
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.NONE,
                        king_piece,
                        Pieces.NONE,
                    )
                )
            attack_bitboard = pop_bit(
                attack_bitboard,
                target_square,
            )
        bitboard = pop_bit(bitboard, source_square)

    if board.side == Color.WHITE:
        bitboardK = board.bitboard[Pieces.K.value]
        bitboardR = board.bitboard[Pieces.R.value]

        if (
            get_bit(bitboardK, Square.e1)
            and get_bit(bitboardR, Square.h1)
            and not get_bit(
                board.occupancy[Color.BOTH.value],
                Square.f1,
            )
            and not get_bit(
                board.occupancy[Color.BOTH.value],
                Square.g1,
            )
            and (board.castle & Castle.WK)
            and (
                not (
                    is_square_attacked(
                        board,
                        Square.f1,
                        Color.BLACK,
                    )
                    or is_square_attacked(
                        board,
                        Square.g1,
                        Color.BLACK,
                    )
                    or is_square_attacked(
                        board,
                        Square.e1,
                        Color.BLACK,
                    )
                )
            )
        ):
            moves.add(
                Move_maker(
                    Square.e1,
                    Square.g1,
                    Flag.CASTLE,
                    Pieces.K,
                    Pieces.NONE,
                )
            )

        if (
            get_bit(bitboardK, Square.e1)
            and get_bit(bitboardR, Square.a1)
            and not get_bit(
                board.occupancy[Color.BOTH.value],
                Square.d1,
            )
            and not get_bit(
                board.occupancy[Color.BOTH.value],
                Square.c1,
            )
            and not get_bit(
                board.occupancy[Color.BOTH.value],
                Square.b1,
            )
            and (board.castle & Castle.WQ)
            and not (
                is_square_attacked(
                    board,
                    Square.d1,
                    Color.BLACK,
                )
                or is_square_attacked(
                    board,
                    Square.c1,
                    Color.BLACK,
                )
                or is_square_attacked(
                    board,
                    Square.e1,
                    Color.BLACK,
                )
            )
        ):
            moves.add(
                Move_maker(
                    Square.e1,
                    Square.c1,
                    Flag.CASTLE,
                    Pieces.K,
                    Pieces.NONE,
                )
            )
    else:
        bitboardK = board.bitboard[Pieces.k.value]
        bitboardR = board.bitboard[Pieces.r.value]
        if (
            get_bit(bitboardK, Square.e8)
            and get_bit(bitboardR, Square.h8)
            and not get_bit(
                board.occupancy[Color.BOTH.value],
                Square.f8,
            )
            and not get_bit(
                board.occupancy[Color.BOTH.value],
                Square.g8,
            )
            and (board.castle & Castle.BK)
            and (
                not (
                    is_square_attacked(
                        board,
                        Square.f8,
                        Color.WHITE,
                    )
                    or is_square_attacked(
                        board,
                        Square.g8,
                        Color.WHITE,
                    )
                    or is_square_attacked(
                        board,
                        Square.e8,
                        Color.WHITE,
                    )
                )
            )
        ):
            moves.add(
                Move_maker(
                    Square.e8,
                    Square.g8,
                    Flag.CASTLE,
                    Pieces.k,
                    Pieces.NONE,
                )
            )

        if (
            get_bit(bitboardK, Square.e8)
            and get_bit(bitboardR, Square.a8)
            and not get_bit(
                board.occupancy[Color.BOTH.value],
                Square.d8,
            )
            and not get_bit(
                board.occupancy[Color.BOTH.value],
                Square.c8,
            )
            and not get_bit(
                board.occupancy[Color.BOTH.value],
                Square.b8,
            )
            and (board.castle & Castle.BQ)
            and (
                not (
                    is_square_attacked(
                        board,
                        Square.d8,
                        Color.WHITE,
                    )
                    or is_square_attacked(
                        board,
                        Square.c8,
                        Color.WHITE,
                    )
                    or is_square_attacked(
                        board,
                        Square.e8,
                        Color.WHITE,
                    )
                )
            )
        ):
            moves.add(
                Move_maker(
                    Square.e8,
                    Square.c8,
                    Flag.CASTLE,
                    Pieces.k,
                    Pieces.NONE,
                )
            )

    return moves


@njit()
def Capture_generator(board: Board):
    moves = MoveList()

    if board.side == Color.WHITE:
        bitboard = board.bitboard[Pieces.P.value]
        while bitboard:
            source_square = get_lsb1_index(bitboard)
            attack_bitboard = get_pawn_attacks(source_square, Color.WHITE.value)

            while attack_bitboard:
                target_square = get_lsb1_index(attack_bitboard)

                if get_bit(board.occupancy[Color.BLACK.value], target_square):
                    capture_piece = Pieces.NONE
                    for x in range(12):
                        if get_bit(board.bitboard[x], target_square):
                            capture_piece = x
                            break

                    if Square.a7 <= source_square <= Square.h7:
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_QUEEN,
                                Pieces.P,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_ROOK,
                                Pieces.P,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_BISHOP,
                                Pieces.P,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_KNIGHT,
                                Pieces.P,
                                capture_piece,
                            )
                        )
                    else:
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE,
                                Pieces.P,
                                capture_piece,
                            )
                        )

                if target_square == board.enpassant:
                    if get_bit(board.occupancy[Color.BLACK.value], target_square + 8):
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.ENPASSANT,
                                Pieces.P,
                                Pieces.p,
                            )
                        )

                attack_bitboard = pop_bit(attack_bitboard, target_square)
            bitboard = pop_bit(bitboard, source_square)
    else:
        bitboard = board.bitboard[Pieces.p.value]
        while bitboard:
            source_square = get_lsb1_index(bitboard)
            attack_bitboard = get_pawn_attacks(source_square, Color.BLACK.value)

            while attack_bitboard:
                target_square = get_lsb1_index(attack_bitboard)

                if get_bit(board.occupancy[Color.WHITE.value], target_square):
                    capture_piece = Pieces.NONE
                    for x in range(12):
                        if get_bit(board.bitboard[x], target_square):
                            capture_piece = x
                            break

                    if Square.a2 <= source_square <= Square.h2:
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_QUEEN,
                                Pieces.p,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_ROOK,
                                Pieces.p,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_BISHOP,
                                Pieces.p,
                                capture_piece,
                            )
                        )
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE_PROMOTION_KNIGHT,
                                Pieces.p,
                                capture_piece,
                            )
                        )
                    else:
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.CAPTURE,
                                Pieces.p,
                                capture_piece,
                            )
                        )

                if target_square == board.enpassant:
                    if get_bit(board.occupancy[Color.WHITE.value], target_square - 8):
                        moves.add(
                            Move_maker(
                                source_square,
                                target_square,
                                Flag.ENPASSANT,
                                Pieces.p,
                                Pieces.P,
                            )
                        )

                attack_bitboard = pop_bit(attack_bitboard, target_square)
            bitboard = pop_bit(bitboard, source_square)

    knight_piece = Pieces.N.value if board.side == Color.WHITE else Pieces.n.value
    opp = Color.BLACK.value if board.side == Color.WHITE else Color.WHITE.value
    bitboard = board.bitboard[knight_piece]
    while bitboard:
        source_square = get_lsb1_index(bitboard)
        attack_bitboard = get_knight_attacks(source_square)
        while attack_bitboard:
            target_square = get_lsb1_index(attack_bitboard)
            if get_bit(board.occupancy[opp], target_square):
                capture_piece = Pieces.NONE
                for x in range(12):
                    if get_bit(board.bitboard[x], target_square):
                        capture_piece = x
                        break
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.CAPTURE,
                        knight_piece,
                        capture_piece,
                    )
                )
            attack_bitboard = pop_bit(attack_bitboard, target_square)
        bitboard = pop_bit(bitboard, source_square)

    bishop_piece = Pieces.B.value if board.side == Color.WHITE else Pieces.b.value
    bitboard = board.bitboard[bishop_piece]
    while bitboard:
        source_square = get_lsb1_index(bitboard)
        attack_bitboard = get_bishop_attacks(
            source_square, board.occupancy[Color.BOTH.value]
        )
        while attack_bitboard:
            target_square = get_lsb1_index(attack_bitboard)
            if get_bit(board.occupancy[opp], target_square):
                capture_piece = Pieces.NONE
                for x in range(12):
                    if get_bit(board.bitboard[x], target_square):
                        capture_piece = x
                        break
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.CAPTURE,
                        bishop_piece,
                        capture_piece,
                    )
                )
            attack_bitboard = pop_bit(attack_bitboard, target_square)
        bitboard = pop_bit(bitboard, source_square)

    rook_piece = Pieces.R.value if board.side == Color.WHITE else Pieces.r.value
    bitboard = board.bitboard[rook_piece]
    while bitboard:
        source_square = get_lsb1_index(bitboard)
        attack_bitboard = get_rook_attacks(
            source_square, board.occupancy[Color.BOTH.value]
        )
        while attack_bitboard:
            target_square = get_lsb1_index(attack_bitboard)
            if get_bit(board.occupancy[opp], target_square):
                capture_piece = Pieces.NONE
                for x in range(12):
                    if get_bit(board.bitboard[x], target_square):
                        capture_piece = x
                        break
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.CAPTURE,
                        rook_piece,
                        capture_piece,
                    )
                )
            attack_bitboard = pop_bit(attack_bitboard, target_square)
        bitboard = pop_bit(bitboard, source_square)

    queen_piece = Pieces.Q.value if board.side == Color.WHITE else Pieces.q.value
    bitboard = board.bitboard[queen_piece]
    while bitboard:
        source_square = get_lsb1_index(bitboard)
        attack_bitboard = get_queen_attacks(
            source_square, board.occupancy[Color.BOTH.value]
        )
        while attack_bitboard:
            target_square = get_lsb1_index(attack_bitboard)
            if get_bit(board.occupancy[opp], target_square):
                capture_piece = Pieces.NONE
                for x in range(12):
                    if get_bit(board.bitboard[x], target_square):
                        capture_piece = x
                        break
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.CAPTURE,
                        queen_piece,
                        capture_piece,
                    )
                )
            attack_bitboard = pop_bit(attack_bitboard, target_square)
        bitboard = pop_bit(bitboard, source_square)

    king_piece = Pieces.K.value if board.side == Color.WHITE else Pieces.k.value
    bitboard = board.bitboard[king_piece]
    while bitboard:
        source_square = get_lsb1_index(bitboard)
        attack_bitboard = get_king_attacks(source_square)
        while attack_bitboard:
            target_square = get_lsb1_index(attack_bitboard)
            if get_bit(board.occupancy[opp], target_square):
                capture_piece = Pieces.NONE
                for x in range(12):
                    if get_bit(board.bitboard[x], target_square):
                        capture_piece = x
                        break
                moves.add(
                    Move_maker(
                        source_square,
                        target_square,
                        Flag.CAPTURE,
                        king_piece,
                        capture_piece,
                    )
                )
            attack_bitboard = pop_bit(attack_bitboard, target_square)
        bitboard = pop_bit(bitboard, source_square)

    return moves


@njit
def Move(board: Board, move):
    castle_cpy = board.castle
    enpassant_cpy = board.enpassant
    castle_shifter = np.array(
        [
            7,
            15,
            15,
            15,
            3,
            15,
            15,
            11,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            15,
            13,
            15,
            15,
            15,
            12,
            15,
            15,
            14,
        ],
        dtype=np.uint8,
    )
    source_square = get_start_square(move)
    target_square = get_target_square(move)
    piece = get_starting_piece(move)
    capture = get_capture_piece(move)
    flag = get_flag(move)
    board.bitboard[piece] = pop_bit(
        board.bitboard[piece],
        source_square,
    )
    board.enpassant = 64
    if flag == Flag.CAPTURE.value:
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            target_square,
        )
        board.bitboard[capture] = pop_bit(
            board.bitboard[capture],
            target_square,
        )

    elif flag == Flag.NONE.value:
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            target_square,
        )

    elif flag == Flag.DOUBLE_PUSH.value:
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            target_square,
        )
        if board.side == Color.BLACK.value:
            board.enpassant = target_square - 8
        else:
            board.enpassant = target_square + 8

    elif flag == Flag.ENPASSANT.value:
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            target_square,
        )
        if board.side == Color.BLACK.value:
            board.bitboard[capture] = pop_bit(
                board.bitboard[capture],
                target_square - 8,
            )
        else:
            board.bitboard[capture] = pop_bit(
                board.bitboard[capture],
                target_square + 8,
            )

    if flag == Flag.BISHOP_PROMOTION.value:
        if board.side == Color.WHITE:
            board.bitboard[Pieces.B.value] = set_bit(
                board.bitboard[Pieces.B.value],
                target_square,
            )

        else:
            board.bitboard[Pieces.b.value] = set_bit(
                board.bitboard[Pieces.b.value],
                target_square,
            )

    if flag == Flag.KNIGHT_PROMOTION.value:
        if board.side == Color.WHITE.value:
            board.bitboard[Pieces.N.value] = set_bit(
                board.bitboard[Pieces.N.value],
                target_square,
            )

        else:
            board.bitboard[Pieces.n.value] = set_bit(
                board.bitboard[Pieces.n.value],
                target_square,
            )

    if flag == Flag.ROOK_PROMOTION.value:
        if board.side == Color.WHITE.value:
            board.bitboard[Pieces.R.value] = set_bit(
                board.bitboard[Pieces.R.value],
                target_square,
            )

        else:
            board.bitboard[Pieces.r.value] = set_bit(
                board.bitboard[Pieces.r.value],
                target_square,
            )

    if flag == Flag.QUEEN_PROMOTION.value:
        if board.side == Color.WHITE.value:
            board.bitboard[Pieces.Q.value] = set_bit(
                board.bitboard[Pieces.Q.value],
                target_square,
            )

        else:
            board.bitboard[Pieces.q.value] = set_bit(
                board.bitboard[Pieces.q.value],
                target_square,
            )

    if flag == Flag.CAPTURE_PROMOTION_QUEEN.value:
        if board.side == Color.WHITE.value:
            board.bitboard[Pieces.Q.value] = set_bit(
                board.bitboard[Pieces.Q.value],
                target_square,
            )

        else:
            board.bitboard[Pieces.q.value] = set_bit(
                board.bitboard[Pieces.q.value],
                target_square,
            )
        board.bitboard[capture] = pop_bit(
            board.bitboard[capture],
            target_square,
        )

    if flag == Flag.CAPTURE_PROMOTION_BISHOP.value:
        if board.side == Color.WHITE.value:
            board.bitboard[Pieces.B.value] = set_bit(
                board.bitboard[Pieces.B.value],
                target_square,
            )

        else:
            board.bitboard[Pieces.b.value] = set_bit(
                board.bitboard[Pieces.b.value],
                target_square,
            )
        board.bitboard[capture] = pop_bit(
            board.bitboard[capture],
            target_square,
        )

    if flag == Flag.CAPTURE_PROMOTION_KNIGHT.value:
        if board.side == Color.WHITE.value:
            board.bitboard[Pieces.N.value] = set_bit(
                board.bitboard[Pieces.N.value],
                target_square,
            )

        else:
            board.bitboard[Pieces.n.value] = set_bit(
                board.bitboard[Pieces.n.value],
                target_square,
            )
        board.bitboard[capture] = pop_bit(
            board.bitboard[capture],
            target_square,
        )

    if flag == Flag.CAPTURE_PROMOTION_ROOK.value:
        if board.side == Color.WHITE.value:
            board.bitboard[Pieces.R.value] = set_bit(
                board.bitboard[Pieces.R.value],
                target_square,
            )

        else:
            board.bitboard[Pieces.r.value] = set_bit(
                board.bitboard[Pieces.r.value],
                target_square,
            )
        board.bitboard[capture] = pop_bit(
            board.bitboard[capture],
            target_square,
        )

    if flag == Flag.CASTLE.value:
        if source_square == Square.e1 and target_square == Square.g1:
            board.bitboard[Pieces.K.value] = set_bit(
                board.bitboard[Pieces.K.value],
                target_square,
            )
            board.bitboard[Pieces.R.value] = set_bit(
                board.bitboard[Pieces.R.value],
                Square.f1,
            )
            board.bitboard[Pieces.R.value] = pop_bit(
                board.bitboard[Pieces.R.value],
                Square.h1,
            )

        if source_square == Square.e1 and target_square == Square.c1:
            board.bitboard[Pieces.K.value] = set_bit(
                board.bitboard[Pieces.K.value],
                target_square,
            )
            board.bitboard[Pieces.R.value] = set_bit(
                board.bitboard[Pieces.R.value],
                Square.d1,
            )
            board.bitboard[Pieces.R.value] = pop_bit(
                board.bitboard[Pieces.R.value],
                Square.a1,
            )

        if source_square == Square.e8 and target_square == Square.g8:
            board.bitboard[Pieces.k.value] = set_bit(
                board.bitboard[Pieces.k.value],
                target_square,
            )
            board.bitboard[Pieces.r.value] = set_bit(
                board.bitboard[Pieces.r.value],
                Square.f8,
            )
            board.bitboard[Pieces.r.value] = pop_bit(
                board.bitboard[Pieces.r.value],
                Square.h8,
            )

        if source_square == Square.e8 and target_square == Square.c8:
            board.bitboard[Pieces.k.value] = set_bit(
                board.bitboard[Pieces.k.value],
                target_square,
            )
            board.bitboard[Pieces.r.value] = set_bit(
                board.bitboard[Pieces.r.value],
                Square.d8,
            )
            board.bitboard[Pieces.r.value] = pop_bit(
                board.bitboard[Pieces.r.value],
                Square.a8,
            )

    board.castle &= castle_shifter[source_square]

    board.occupancy[0] = (
        board.bitboard[0]
        | board.bitboard[1]
        | board.bitboard[2]
        | board.bitboard[3]
        | board.bitboard[4]
        | board.bitboard[5]
    )
    board.occupancy[1] = (
        board.bitboard[6]
        | board.bitboard[7]
        | board.bitboard[8]
        | board.bitboard[9]
        | board.bitboard[10]
        | board.bitboard[11]
    )
    board.occupancy[2] = board.occupancy[0] | board.occupancy[1]
    Wking = get_lsb1_index(board.bitboard[Pieces.K.value])
    Bking = get_lsb1_index(board.bitboard[Pieces.k.value])
    if (
        is_square_attacked(board, Wking, Color.BLACK) and board.side == Color.WHITE
    ) or (is_square_attacked(board, Bking, Color.WHITE) and board.side == Color.BLACK):
        unmove(board, move, castle_cpy, enpassant_cpy, board.halfmove)
        return False
    board.side = Color.WHITE if board.side == Color.BLACK else Color.BLACK

    if (
        piece == Pieces.P
        or flag == Flag.CAPTURE_PROMOTION_ROOK
        or flag == Flag.CAPTURE
        or flag == Flag.CAPTURE_PROMOTION_QUEEN
        or flag == Flag.CAPTURE_PROMOTION_KNIGHT
        or flag == Flag.CAPTURE_PROMOTION_BISHOP
        or flag == Flag.ENPASSANT
    ):
        board.halfmove = 0
    else:
        board.halfmove += 1

    board.hash ^= piece_keys[get_starting_piece(move),get_start_square(move)]
    board.hash ^= piece_keys[get_starting_piece(move),get_target_square(move)]

    if (flag == Flag.CAPTURE_PROMOTION_ROOK
        or flag == Flag.CAPTURE
        or flag == Flag.CAPTURE_PROMOTION_QUEEN
        or flag == Flag.CAPTURE_PROMOTION_KNIGHT
        or flag == Flag.CAPTURE_PROMOTION_BISHOP
        or flag == Flag.ENPASSANT):
        board.hash ^= piece_keys[get_capture_piece(move),get_target_square(move)]

    if enpassant_cpy != 64:
        board.hash ^= enpassant_keys[enpassant_cpy]

    if board.enpassant != 64:
        board.hash ^= enpassant_keys[board.enpassant]

    board.hash ^= castle_keys[castle_cpy]
    board.hash ^= castle_keys[board.castle]

    board.hash ^= side_key
    return True


@njit
def unmove(board, move, prev_castle, prev_enpassant, ply):
    source_square = get_start_square(move)
    target_square = get_target_square(move)
    piece = get_starting_piece(move)
    capture = get_capture_piece(move)
    flag = get_flag(move)

    board.side = piece > Pieces.K
    mover = board.side

    if flag == Flag.QUEEN_PROMOTION:
        promo = Pieces.Q.value if mover == Color.WHITE else Pieces.q.value
        board.bitboard[promo] = pop_bit(
            board.bitboard[promo],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )

    elif flag == Flag.ROOK_PROMOTION:
        promo = Pieces.R.value if mover == Color.WHITE else Pieces.r.value
        board.bitboard[promo] = pop_bit(
            board.bitboard[promo],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )

    elif flag == Flag.BISHOP_PROMOTION:
        promo = Pieces.B.value if mover == Color.WHITE else Pieces.b.value
        board.bitboard[promo] = pop_bit(
            board.bitboard[promo],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )

    elif flag == Flag.KNIGHT_PROMOTION:
        promo = Pieces.N.value if mover == Color.WHITE else Pieces.n.value
        board.bitboard[promo] = pop_bit(
            board.bitboard[promo],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )

    elif flag == Flag.CAPTURE_PROMOTION_QUEEN:
        promo = Pieces.Q.value if mover == Color.WHITE else Pieces.q.value
        board.bitboard[promo] = pop_bit(
            board.bitboard[promo],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )
        board.bitboard[capture] = set_bit(
            board.bitboard[capture],
            target_square,
        )

    elif flag == Flag.CAPTURE_PROMOTION_ROOK:
        promo = Pieces.R.value if mover == Color.WHITE else Pieces.r.value
        board.bitboard[promo] = pop_bit(
            board.bitboard[promo],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )
        board.bitboard[capture] = set_bit(
            board.bitboard[capture],
            target_square,
        )

    elif flag == Flag.CAPTURE_PROMOTION_BISHOP:
        promo = Pieces.B.value if mover == Color.WHITE else Pieces.b.value
        board.bitboard[promo] = pop_bit(
            board.bitboard[promo],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )
        board.bitboard[capture] = set_bit(
            board.bitboard[capture],
            target_square,
        )

    elif flag == Flag.CAPTURE_PROMOTION_KNIGHT:
        promo = Pieces.N.value if mover == Color.WHITE else Pieces.n.value
        board.bitboard[promo] = pop_bit(
            board.bitboard[promo],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )
        board.bitboard[capture] = set_bit(
            board.bitboard[capture],
            target_square,
        )

    elif flag == Flag.CASTLE:
        if source_square == Square.e1 and target_square == Square.g1:
            board.bitboard[Pieces.K.value] = pop_bit(
                board.bitboard[Pieces.K.value],
                Square.g1,
            )
            board.bitboard[Pieces.K.value] = set_bit(
                board.bitboard[Pieces.K.value],
                Square.e1,
            )
            board.bitboard[Pieces.R.value] = pop_bit(
                board.bitboard[Pieces.R.value],
                Square.f1,
            )
            board.bitboard[Pieces.R.value] = set_bit(
                board.bitboard[Pieces.R.value],
                Square.h1,
            )
        elif source_square == Square.e1 and target_square == Square.c1:
            board.bitboard[Pieces.K.value] = pop_bit(
                board.bitboard[Pieces.K.value],
                Square.c1,
            )
            board.bitboard[Pieces.K.value] = set_bit(
                board.bitboard[Pieces.K.value],
                Square.e1,
            )
            board.bitboard[Pieces.R.value] = pop_bit(
                board.bitboard[Pieces.R.value],
                Square.d1,
            )
            board.bitboard[Pieces.R.value] = set_bit(
                board.bitboard[Pieces.R.value],
                Square.a1,
            )
        elif source_square == Square.e8 and target_square == Square.g8:
            board.bitboard[Pieces.k.value] = pop_bit(
                board.bitboard[Pieces.k.value],
                Square.g8,
            )
            board.bitboard[Pieces.k.value] = set_bit(
                board.bitboard[Pieces.k.value],
                Square.e8,
            )
            board.bitboard[Pieces.r.value] = pop_bit(
                board.bitboard[Pieces.r.value],
                Square.f8,
            )
            board.bitboard[Pieces.r.value] = set_bit(
                board.bitboard[Pieces.r.value],
                Square.h8,
            )
        elif source_square == Square.e8 and target_square == Square.c8:
            board.bitboard[Pieces.k.value] = pop_bit(
                board.bitboard[Pieces.k.value],
                Square.c8,
            )
            board.bitboard[Pieces.k.value] = set_bit(
                board.bitboard[Pieces.k.value],
                Square.e8,
            )
            board.bitboard[Pieces.r.value] = pop_bit(
                board.bitboard[Pieces.r.value],
                Square.d8,
            )
            board.bitboard[Pieces.r.value] = set_bit(
                board.bitboard[Pieces.r.value],
                Square.a8,
            )

    elif flag == Flag.ENPASSANT:
        board.bitboard[piece] = pop_bit(
            board.bitboard[piece],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )
        if mover == Color.WHITE:
            captured_sq = target_square + 8
        else:
            captured_sq = target_square - 8
        board.bitboard[capture] = set_bit(
            board.bitboard[capture],
            captured_sq,
        )

    elif flag == Flag.DOUBLE_PUSH or flag == Flag.NONE:
        board.bitboard[piece] = pop_bit(
            board.bitboard[piece],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )

    elif flag == Flag.CAPTURE:
        board.bitboard[piece] = pop_bit(
            board.bitboard[piece],
            target_square,
        )
        board.bitboard[piece] = set_bit(
            board.bitboard[piece],
            source_square,
        )
        board.bitboard[capture] = set_bit(
            board.bitboard[capture],
            target_square,
        )

    board.castle = prev_castle
    board.enpassant = prev_enpassant

    occ_white = 0
    occ_black = 0
    occ_white |= board.bitboard[Pieces.P.value]
    occ_white |= board.bitboard[Pieces.N.value]
    occ_white |= board.bitboard[Pieces.B.value]
    occ_white |= board.bitboard[Pieces.R.value]
    occ_white |= board.bitboard[Pieces.Q.value]
    occ_white |= board.bitboard[Pieces.K.value]

    occ_black |= board.bitboard[Pieces.p.value]
    occ_black |= board.bitboard[Pieces.n.value]
    occ_black |= board.bitboard[Pieces.b.value]
    occ_black |= board.bitboard[Pieces.r.value]
    occ_black |= board.bitboard[Pieces.q.value]
    occ_black |= board.bitboard[Pieces.k.value]

    occ_all = occ_white | occ_black

    board.occupancy[Color.WHITE.value] = occ_white
    board.occupancy[Color.BLACK.value] = occ_black
    board.occupancy[2] = occ_all
    board.halfmove = ply


@njit
def perft_divide(board, depth: int) -> int:
    if depth == 0:
        return 1

    if depth == 1:
        ml = Move_generator(board)
        cnt = 0
        for i in range(ml.counter):
            saved_castle = board.castle
            saved_enpassant = board.enpassant
            saved_ply = board.halfmove
            if Move(board, ml.moves[i]):
                cnt += 1
                unmove(board, ml.moves[i], saved_castle, saved_enpassant, saved_ply)
        return cnt

    MAX_STACK = 15000
    MAX_MOVES = 128

    stack_depth = np.empty(MAX_STACK, np.int64)
    stack_move_index = np.empty(MAX_STACK, np.int64)
    stack_move_count = np.empty(MAX_STACK, np.int64)
    stack_enter_move = np.empty(MAX_STACK, np.int64)
    stack_enter_castle = np.empty(MAX_STACK, np.int64)
    stack_enter_enpassant = np.empty(MAX_STACK, np.int64)
    stack_moves = np.empty((MAX_STACK, MAX_MOVES), np.int64)

    total_nodes = 0
    sp = 0

    ml = Move_generator(board)
    cnt = ml.counter
    if cnt > MAX_MOVES:
        return -1

    stack_move_count[0] = cnt
    stack_move_index[0] = 0
    stack_depth[0] = depth
    stack_enter_move[0] = -1
    stack_enter_castle[0] = board.castle
    stack_enter_enpassant[0] = board.enpassant

    for j in range(cnt):
        stack_moves[0, j] = ml.moves[j]

    sp = 1

    while sp > 0:
        frame = sp - 1
        cur_depth = stack_depth[frame]
        move_idx = stack_move_index[frame]
        move_cnt = stack_move_count[frame]

        if move_idx >= move_cnt:
            sp -= 1
            if stack_enter_move[frame] != -1:
                unmove(
                    board,
                    stack_enter_move[frame],
                    stack_enter_castle[frame],
                    stack_enter_enpassant[frame],
                )
            continue

        mv = stack_moves[frame, move_idx]
        stack_move_index[frame] = move_idx + 1

        saved_castle = board.castle
        saved_enpassant = board.enpassant

        if not Move(board, mv):
            continue

        if cur_depth == 1:
            total_nodes += 1
            unmove(
                board,
                mv,
                saved_castle,
                saved_enpassant,
            )
        else:
            ml2 = Move_generator(board)
            cnt2 = ml2.counter

            if cnt2 > MAX_MOVES or sp >= MAX_STACK:
                return -1

            if cnt2 > 0:
                stack_enter_move[sp] = mv
                stack_enter_castle[sp] = saved_castle
                stack_enter_enpassant[sp] = saved_enpassant
                stack_depth[sp] = cur_depth - 1
                stack_move_count[sp] = cnt2
                stack_move_index[sp] = 0

                for k in range(cnt2):
                    stack_moves[sp, k] = ml2.moves[k]

                sp += 1
            else:
                unmove(
                    board,
                    mv,
                    saved_castle,
                    saved_enpassant,
                )

    return total_nodes


def perft(board, depth, root=True):
    if depth == 0:
        return 1

    mvs = Move_generator(board=board)
    perft_r = 0
    a, b = board.castle, board.enpassant

    for mv in mvs.moves:
        if mv == 0:
            break
        if Move(board, mv):
            n = perft(board, depth - 1, False)
            unmove(board, mv, a, b)
            perft_r += n
            if root:
                print(
                    f"{square_to_coord[get_start_square(mv)]}{square_to_coord[get_target_square(mv)]}: {n}"
                )

    if root:
        print(f"Nodes searched: {perft_r}")
    return perft_r


def run_perft(fen: str, depth: int):
    from time import perf_counter as pf

    board = Board(*parse_fen(fen.encode()))
    a = pf()
    total = perft_divide(board, depth)

    b = pf()
    return total / (b - a)
    print(f"\nTotal nodes at depth {depth}: {total}")
    print(f"Total nodes per second : {total // (b - a):,}")


def test_perft(
    fen,
    perft_results,
    error_filename="errors.txt",
    report_filename="report.txt",
):
    for depth in tqdm.tqdm(sorted(perft_results.keys())):
        try:
            total = run_perft(fen, depth)
        except Exception as e:
            with open(error_filename, "a") as f:
                f.write(f"fen : {fen} exception at depth : {depth}; exception: {e}\n")
            return depth

        try:
            total_int = int(total)  # type : ignore
        except Exception:
            with open(error_filename, "a") as f:
                f.write(
                    f"fen : {fen} non-integer result at depth : {depth}; mine : {total}\n"
                )
            return depth

        if total_int != int(perft_results[depth]):
            with open(error_filename, "a") as f:
                f.write(
                    f"fen : {fen} wrong at depth : {depth}; original : {perft_results[depth]} ; mine : {total_int}\n"
                )
            return depth
    return False


# if __name__ == "__main__":
# # FOR DEBUG
# tests = [
#     (
#         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
#         {
#             1: 20,
#             2: 400,
#             3: 8902,
#             4: 197281,
#             5: 4865609,
#             6: 119060324,
#             7: 3195901860,
#             8: 84998978956,


#         },
#     ),
#     (
#         "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
#         {1: 48, 2: 2039, 3: 97862, 4: 4085603, 5: 193690690, 6: 8031647685},
#     ),
#     (
#         "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w ---- 0 1",
#         {
#             1: 14,
#             2: 191,
#             3: 2812,
#             4: 43238,
#             5: 674624,
#             6: 11030083,
#             7: 178633661,
#             8: 3009794393,
#         },
#     ),
#     (
#         "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w --kq - 0 1",
#         {1: 6, 2: 264, 3: 9467, 4: 422333, 5: 15833292, 6: 706045033},
#     ),
#     (
#         "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ-- - 0 1",
#         {1: 44, 2: 1486, 3: 62379, 4: 2103487, 5: 89941194},
#     ),
#     (
#         "r4rk1/1pp1qppp/p1n1bn2/3pN3/1b1P4/2N1B3/PPP2PPP/R2Q1RK1 w ---- 0 1",
#         {1: 41, 2: 1769, 3: 71340, 4: 3027100, 5: 120980532, 6: 5111125376},
#     ),
# ]
# for fen, results in tests:
#     mismatch = test_perft(fen, results)
#     if mismatch:
#         print(f"Testing {fen} — first mismatch depth:{mismatch}")
#     else:
#         print(f"{fen} Passed")


# # FOR NPS Calculation
# from multiprocessing import Pool, Manager, cpu_count
# from pprint import pprint

# def task(args):
#     fen, depth, log_queue = args
#     result = run_perft(fen, depth)
#     log_queue.put(f"[Completed] FEN='{fen}', Depth={depth}")

#     return fen, depth, result

# if __name__ == "__main__":
#     from SAT_MINE.helpers import create_plot

#     fens_list = [
#         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
#         "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
#         "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w --kq - 0 1",
#         "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ-- - 0 1",
#         "r4rk1/1pp1qppp/p1n1bn2/3pN3/1b1P4/2N1B3/PPP2PPP/R2Q1RK1 w ---- 0 1"
#     ]

#     depth_values = range(1, 7)  # 1 → 8
#     fens = {fen: [] for fen in fens_list}

#     print("STARTED")
#     manager = Manager()
#     log_queue = manager.Queue()

#     jobs = []
#     for fen in fens_list:
#         for depth in depth_values:
#             jobs.append((fen, depth, log_queue))

#     TOTAL_JOBS = len(jobs)

#     CORE_COUNT = 6

#     with Pool(processes=CORE_COUNT) as pool:
#         results_async = pool.map_async(task, jobs)

#         completed = 0

#         while completed < TOTAL_JOBS:
#             msg = log_queue.get()
#             print(msg)
#             completed += 1

#         results = results_async.get()

#     for fen, depth, result in results:
#         fens[fen].append(result)

#     print("ALL TASKS COMPLETED")
#     pprint(fens)

#     create_plot("NPS", "Depth", "NPS", fens, "NPS.png", True)
