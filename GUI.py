import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from enum import Enum

import pygame

from Board import Board, is_square_attacked, parse_fen
from Board_Move_gen import (
    Move,
    Move_generator,
    get_flag,
    get_start_square,
    get_target_square,
    move_to_uci,
    unmove,
)
from Constants import Color, Flag, Pieces
from Minimax import AI
from pregen.Utilities import get_lsb1_index


class FENS:
    QUEEN = b"3k4/3q4/8/8/8/8/8/3K4 w ---- 0 1"
    DoubleRook = b"3r4/3r4/3k4/8/8/8/8/3K4 w ---- 0 1"
    ROOK = b"3r4/3k4/8/8/8/8/8/3K4 w ---- 0 1"
    DoubleBishop = b"3bb3/3k4/8/8/8/8/8/3K4 w ---- 0 1"
    RookKnight = b"8/3k4/2bn4/8/8/8/8/3K4 w ---- 0 1"
    DoublePawn = b"8/3k4/3pp3/8/8/8/8/3K4 w ---- 0 1"
    M1 = b"1q6/8/8/8/8/2k5/8/K7 w - - 0 1"
    STARTING = b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


FEN = FENS.STARTING


class GameMode(Enum):
    MENU = 0
    PVP = 1
    AI = 2


class GameState(Enum):
    PLAYING = 0
    CHECKMATE = 1
    STALEMATE = 2
    DRAW = 3


class PieceImageCache:
    """Efficient piece image caching with fallback"""

    def __init__(self, piece_size: int = 80):
        self.piece_size = piece_size
        self.cache: dict[str, pygame.Surface] = {}
        self.piece_names = {
            0: "pawn",
            1: "knight",
            2: "bishop",
            3: "rook",
            4: "queen",
            5: "king",
            6: "pawn",
            7: "knight",
            8: "bishop",
            9: "rook",
            10: "queen",
            11: "king",
        }
        self.colors = {
            0: "white",
            1: "white",
            2: "white",
            3: "white",
            4: "white",
            5: "white",
            6: "black",
            7: "black",
            8: "black",
            9: "black",
            10: "black",
            11: "black",
        }

    def get_piece(self, piece_idx: int) -> pygame.Surface:
        color = self.colors[piece_idx]
        piece = self.piece_names[piece_idx]
        key = f"{color}_{piece}"

        if key not in self.cache:
            path = f"images/{color}_{piece}.png"
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path)
                    img = pygame.transform.smoothscale(
                        img, (self.piece_size, self.piece_size)
                    )
                    self.cache[key] = img
                except:
                    self.cache[key] = self._create_fallback(piece, color)
            else:
                self.cache[key] = self._create_fallback(piece, color)

        return self.cache[key]

    def _create_fallback(self, name, color_name):
        s = pygame.Surface((self.piece_size, self.piece_size), pygame.SRCALPHA)
        try:
            font = pygame.font.SysFont("Arial", int(self.piece_size / 2), bold=True)
        except:
            font = pygame.font.Font(None, int(self.piece_size / 2))

        text_color = (255, 255, 255) if color_name == "white" else (0, 0, 0)
        bg_color = (220, 220, 220) if color_name == "white" else (60, 60, 60)
        stroke = (0, 0, 0) if color_name == "white" else (255, 255, 255)

        pygame.draw.circle(
            s,
            bg_color,
            (self.piece_size // 2, self.piece_size // 2),
            self.piece_size // 2 - 5,
        )
        pygame.draw.circle(
            s,
            stroke,
            (self.piece_size // 2, self.piece_size // 2),
            self.piece_size // 2 - 5,
            2,
        )

        txt = font.render(name[0].upper(), True, text_color)
        s.blit(
            txt,
            (
                self.piece_size // 2 - txt.get_width() // 2,
                self.piece_size // 2 - txt.get_height() // 2,
            ),
        )
        return s


class Slider:
    """Interactive slider widget"""

    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        self.handle_radius = height // 2 + 2

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            handle_x = (
                self.rect.x
                + ((self.value - self.min_val) / (self.max_val - self.min_val))
                * self.rect.width
            )
            if (
                (mx - handle_x) ** 2 + (my - self.rect.centery) ** 2
            ) <= self.handle_radius**2:
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx = event.pos[0]
            ratio = max(0, min(1, (mx - self.rect.x) / self.rect.width))
            self.value = int(self.min_val + ratio * (self.max_val - self.min_val))
            return True
        return False

    def draw(self, surface, fonts, colors):
        # Track
        track_rect = pygame.Rect(self.rect.x, self.rect.centery - 3, self.rect.width, 6)
        pygame.draw.rect(surface, (60, 60, 60), track_rect, border_radius=3)

        # Filled portion
        fill_width = (
            (self.value - self.min_val) / (self.max_val - self.min_val)
        ) * self.rect.width
        fill_rect = pygame.Rect(self.rect.x, self.rect.centery - 3, fill_width, 6)
        pygame.draw.rect(surface, colors["accent"], fill_rect, border_radius=3)

        # Handle
        handle_x = self.rect.x + fill_width
        pygame.draw.circle(
            surface, (40, 40, 40), (handle_x, self.rect.centery + 2), self.handle_radius
        )
        pygame.draw.circle(
            surface,
            colors["accent"] if self.dragging else (200, 200, 200),
            (handle_x, self.rect.centery),
            self.handle_radius,
        )

        # Label
        label_text = fonts["small"].render(
            f"{self.label}: {self.value}", True, colors["text"]
        )
        surface.blit(label_text, (self.rect.x, self.rect.y - 25))


class ChessGUI:
    COLORS = {
        "light_sq": (240, 217, 181),
        "dark_sq": (181, 136, 99),
        "bg": (28, 28, 32),
        "panel": (38, 38, 42),
        "panel_dark": (28, 28, 32),
        "highlight": (255, 255, 0, 100),
        "selected": (130, 151, 105, 200),
        "last_move": (205, 210, 106, 140),
        "text": (240, 240, 245),
        "text_muted": (160, 160, 170),
        "btn_default": (55, 55, 60),
        "btn_hover": (75, 75, 80),
        "accent": (100, 181, 246),
        "accent_hover": (120, 201, 255),
        "overlay": (0, 0, 0, 200),
        "white_piece": (255, 255, 255),
        "black_piece": (50, 50, 50),
        "check_red": (220, 70, 70, 180),
    }

    def __init__(
        self, width: int = 1400, height: int = 900, ai_func: Callable | None = None
    ):
        pygame.init()
        self.display = pygame.display.set_mode(
            (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Chess Engine - Advanced")

        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()

        self.fonts = {
            "header": pygame.font.SysFont("Segoe UI", 72, bold=True),
            "subheader": pygame.font.SysFont("Segoe UI", 48, bold=True),
            "title": pygame.font.SysFont("Segoe UI", 32, bold=True),
            "normal": pygame.font.SysFont("Segoe UI", 22),
            "small": pygame.font.SysFont("Segoe UI", 16),
            "tiny": pygame.font.SysFont("Segoe UI", 14),
            "coords": pygame.font.SysFont("Segoe UI", 13, bold=True),
        }

        self.board_size = min(height - 60, width - 400)
        self.square_size = self.board_size // 8
        self.offset_x = 40
        self.offset_y = (height - self.board_size) // 2

        self.piece_cache = PieceImageCache(self.square_size)
        self.ai_func = ai_func

        self.ai_queue = queue.Queue()
        self.thinking = False
        self.ai_start_time = 0

        self.mode = GameMode.MENU
        self.board = None
        self.selected = None
        self.valid_moves = []
        self.move_log = []
        self.history_states = []
        self.last_move = None

        self.game_state = GameState.PLAYING
        self.winner_text = ""
        self.end_reason = ""

        # New features
        self.player_color = Color.WHITE  # Player's chosen color
        self.ai_depth = 5  # AI search depth
        self.board_flipped = False  # Whether board is flipped

        # Menu state
        self.menu_selected_color = Color.WHITE
        self.depth_slider = None

        self.warmup_engine()

    def warmup_engine(self):
        """Forces Numba compilation"""
        self.display.fill(self.COLORS["bg"])

        # Fancy loading animation
        for i in range(3):
            self.display.fill(self.COLORS["bg"])
            dots = "." * (i + 1)
            text = self.fonts["title"].render(
                f"Initializing Engine{dots}", True, self.COLORS["accent"]
            )
            self.display.blit(
                text, (self.width // 2 - text.get_width() // 2, self.height // 2)
            )
            pygame.display.flip()
            time.sleep(0.2)

        pieces, occ, side, castle, ep, ply = parse_fen(FEN)
        dummy_board = Board(pieces, occ, side, castle, ep, ply)
        Move_generator(dummy_board)
        is_square_attacked(dummy_board, 0, 0)

        self.display.fill(self.COLORS["bg"])
        text = self.fonts["title"].render("Ready!", True, self.COLORS["accent"])
        self.display.blit(
            text, (self.width // 2 - text.get_width() // 2, self.height // 2)
        )
        pygame.display.flip()
        time.sleep(0.3)

    def pixel_to_square(self, x: int, y: int) -> int | None:
        x -= self.offset_x
        y -= self.offset_y
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            file = x // self.square_size
            rank = y // self.square_size

            if self.board_flipped:
                rank = 7 - rank
                file = 7 - file

            return rank * 8 + file
        return None

    def square_to_pixel(self, square: int) -> tuple[int, int]:
        rank, file = divmod(square, 8)

        if self.board_flipped:
            rank = 7 - rank
            file = 7 - file

        return (
            self.offset_x + file * self.square_size + self.square_size // 2,
            self.offset_y + rank * self.square_size + self.square_size // 2,
        )

    def get_legal_moves_for_square(self, square: int) -> list:
        moves = Move_generator(self.board)
        prev_c, prev_ep, prev_ply = (
            self.board.castle,
            self.board.enpassant,
            self.board.halfmove,
        )
        legal_moves = []
        for i in range(moves.counter):
            m = moves.moves[i]
            if get_start_square(m) == square:
                if Move(self.board, m):
                    unmove(self.board, m, prev_c, prev_ep, prev_ply)
                    legal_moves.append(m)
        return legal_moves

    def check_game_over(self):
        moves = Move_generator(self.board)
        prev_c, prev_ep, prev_ply = (
            self.board.castle,
            self.board.enpassant,
            self.board.halfmove,
        )
        has_legal_move = False

        for i in range(moves.counter):
            m = moves.moves[i]
            if Move(self.board, m):
                has_legal_move = True
                unmove(self.board, m, prev_c, prev_ep, prev_ply)
                break

        if not has_legal_move:
            king_piece = (
                Pieces.K.value if self.board.side == Color.WHITE else Pieces.k.value
            )
            king_sq = get_lsb1_index(self.board.bitboard[king_piece])
            attacker = Color.BLACK if self.board.side == Color.WHITE else Color.WHITE

            if is_square_attacked(self.board, king_sq, attacker):
                self.game_state = GameState.CHECKMATE
                winner = "White" if self.board.side == Color.BLACK else "Black"
                self.winner_text = f"{winner} Wins!"
                self.end_reason = "Checkmate"
            elif self.board.halfmove == 50:
                self.game_state = GameState.DRAW
            else:
                self.game_state = GameState.STALEMATE
                self.winner_text = "Draw"
                self.end_reason = "Stalemate"

    def push_move(self, move):
        self.history_states.append(
            {
                "board": self.board.copy(),
                "last_move": self.last_move,
                "log": self.move_log[:],
                "state": self.game_state,
            }
        )

        if Move(self.board, move):
            self.last_move = (get_start_square(move), get_target_square(move))
            self.move_log.append(move_to_uci(move))
            self.selected = None
            self.valid_moves = []
            self.check_game_over()
            return True
        return False

    def undo_move(self):
        if not self.history_states:
            return
        state = self.history_states.pop()
        self.board = state["board"]
        self.last_move = state["last_move"]
        self.move_log = state["log"]
        self.game_state = state.get("state", GameState.PLAYING)
        self.selected = None
        self.valid_moves = []
        self.thinking = False

    def start_ai_turn(self):
        if self.thinking or self.game_state != GameState.PLAYING:
            return
        self.thinking = True
        self.ai_start_time = time.time()
        threading.Thread(target=self._run_ai_logic, daemon=True).start()

    def _run_ai_logic(self):
        try:
            board_copy = self.board.copy()
            move = AI(board_copy, board_copy.side, self.ai_depth)
            self.ai_queue.put(move)
        except Exception as e:
            print(f"AI Error: {e}")
            self.ai_queue.put(None)

    def draw_board(self):
        for r in range(8):
            for f in range(8):
                # Determine actual square based on flip
                actual_r = (7 - r) if self.board_flipped else r
                actual_f = (7 - f) if self.board_flipped else f

                color = (
                    self.COLORS["light_sq"]
                    if (actual_r + actual_f) % 2 == 0
                    else self.COLORS["dark_sq"]
                )
                x = self.offset_x + f * self.square_size
                y = self.offset_y + r * self.square_size
                pygame.draw.rect(
                    self.display, color, (x, y, self.square_size, self.square_size)
                )

                # Coordinates
                coord_color = (
                    self.COLORS["dark_sq"]
                    if color == self.COLORS["light_sq"]
                    else self.COLORS["light_sq"]
                )

                if f == 0:
                    rank_num = 8 - actual_r
                    t = self.fonts["coords"].render(str(rank_num), True, coord_color)
                    self.display.blit(t, (x + 4, y + 4))

                if r == 7:
                    file_letter = chr(97 + actual_f)
                    t = self.fonts["coords"].render(file_letter, True, coord_color)
                    self.display.blit(
                        t, (x + self.square_size - 14, y + self.square_size - 18)
                    )

        # Highlight last move
        if self.last_move:
            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            s.fill(self.COLORS["last_move"])
            for sq in self.last_move:
                rank, file = divmod(sq, 8)
                if self.board_flipped:
                    rank = 7 - rank
                    file = 7 - file
                self.display.blit(
                    s,
                    (
                        self.offset_x + file * self.square_size,
                        self.offset_y + rank * self.square_size,
                    ),
                )

        # Highlight selected square
        if self.selected is not None:
            rank, file = divmod(self.selected, 8)
            if self.board_flipped:
                rank = 7 - rank
                file = 7 - file
            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            s.fill(self.COLORS["selected"])
            self.display.blit(
                s,
                (
                    self.offset_x + file * self.square_size,
                    self.offset_y + rank * self.square_size,
                ),
            )

        # Draw move indicators
        for move in self.valid_moves:
            target = get_target_square(move)
            rank, file = divmod(target, 8)
            if self.board_flipped:
                rank = 7 - rank
                file = 7 - file

            is_capture = any((self.board.bitboard[i] >> target) & 1 for i in range(12))

            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            if is_capture:
                pygame.draw.circle(
                    s,
                    (90, 90, 90, 120),
                    (self.square_size // 2, self.square_size // 2),
                    self.square_size // 2 - 4,
                    5,
                )
            else:
                pygame.draw.circle(
                    s,
                    (70, 70, 70, 120),
                    (self.square_size // 2, self.square_size // 2),
                    self.square_size // 6,
                )

            self.display.blit(
                s,
                (
                    self.offset_x + file * self.square_size,
                    self.offset_y + rank * self.square_size,
                ),
            )

    def draw_pieces(self):
        for idx in range(12):
            bb = self.board.bitboard[idx]
            square = 0
            while bb:
                if bb & 1:
                    x, y = self.square_to_pixel(square)
                    img = self.piece_cache.get_piece(idx)
                    rect = img.get_rect(center=(x, y))
                    self.display.blit(img, rect)
                bb >>= 1
                square += 1

    def draw_panel(self):
        panel_x = self.offset_x + self.board_size + 25
        panel_w = self.width - panel_x - 30

        # Main panel background
        pygame.draw.rect(
            self.display,
            self.COLORS["panel"],
            (panel_x, self.offset_y, panel_w, self.board_size),
            border_radius=12,
        )

        y_cursor = self.offset_y + 20

        # Game status section
        if self.game_state == GameState.PLAYING:
            turn_str = (
                "White to Move" if self.board.side == Color.WHITE else "Black to Move"
            )
            if self.thinking:
                elapsed = time.time() - self.ai_start_time
                turn_str = f"AI Thinking... ({elapsed:.1f}s)"
            col = self.COLORS["accent"] if self.thinking else self.COLORS["text"]
        else:
            turn_str = "Game Over"
            col = (220, 100, 100)

        txt = self.fonts["title"].render(turn_str, True, col)
        self.display.blit(txt, (panel_x + 20, y_cursor))
        y_cursor += 50

        # Divider
        pygame.draw.line(
            self.display,
            (70, 70, 70),
            (panel_x + 15, y_cursor),
            (panel_x + panel_w - 15, y_cursor),
            2,
        )
        y_cursor += 15

        # Game info section
        if self.mode == GameMode.AI:
            mode_text = f"Mode: Player ({self.get_color_name(self.player_color)}) vs AI"
            depth_text = f"AI Depth: {self.ai_depth}"
        else:
            mode_text = "Mode: Player vs Player"
            depth_text = ""

        info1 = self.fonts["small"].render(mode_text, True, self.COLORS["text_muted"])
        self.display.blit(info1, (panel_x + 20, y_cursor))
        y_cursor += 25

        if depth_text:
            info2 = self.fonts["small"].render(
                depth_text, True, self.COLORS["text_muted"]
            )
            self.display.blit(info2, (panel_x + 20, y_cursor))
            y_cursor += 25

        moves_text = self.fonts["small"].render(
            f"Moves: {len(self.move_log)}", True, self.COLORS["text_muted"]
        )
        self.display.blit(moves_text, (panel_x + 20, y_cursor))
        y_cursor += 35

        # Move history section
        pygame.draw.line(
            self.display,
            (70, 70, 70),
            (panel_x + 15, y_cursor),
            (panel_x + panel_w - 15, y_cursor),
            2,
        )
        y_cursor += 15

        hist_title = self.fonts["normal"].render(
            "Move History", True, self.COLORS["text"]
        )
        self.display.blit(hist_title, (panel_x + 20, y_cursor))
        y_cursor += 35

        # Display moves in two columns
        history_start = max(0, len(self.move_log) - 24)
        moves_to_show = self.move_log[history_start:]

        col1_x = panel_x + 20
        col2_x = panel_x + panel_w // 2 + 10

        for i, m in enumerate(moves_to_show):
            num = history_start + i + 1
            x_pos = col1_x if i % 2 == 0 else col2_x

            if i % 2 == 0:
                row_color = self.COLORS["text"]
            else:
                row_color = self.COLORS["text_muted"]

            t = self.fonts["tiny"].render(f"{num}. {m}", True, row_color)
            self.display.blit(t, (x_pos, y_cursor))

            if i % 2 == 1:
                y_cursor += 22

        # Control buttons at bottom
        mx, my = pygame.mouse.get_pos()
        btn_y = self.offset_y + self.board_size - 190
        btn_w = panel_w - 40

        flip_rect = pygame.Rect(panel_x + 20, btn_y, btn_w, 45)
        undo_rect = pygame.Rect(panel_x + 20, btn_y + 55, btn_w, 45)
        menu_rect = pygame.Rect(panel_x + 20, btn_y + 110, btn_w, 45)

        self._draw_btn(flip_rect, "Flip Board", mx, my)
        self._draw_btn(undo_rect, "Undo Move", mx, my)
        self._draw_btn(menu_rect, "Main Menu", mx, my)

        return flip_rect, undo_rect, menu_rect

    def draw_game_over_overlay(self):
        if self.game_state == GameState.PLAYING:
            return

        s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        s.fill(self.COLORS["overlay"])
        self.display.blit(s, (0, 0))

        box_w, box_h = 500, 320
        cx, cy = self.width // 2, self.height // 2
        box_rect = pygame.Rect(cx - box_w // 2, cy - box_h // 2, box_w, box_h)

        pygame.draw.rect(self.display, self.COLORS["panel"], box_rect, border_radius=16)
        pygame.draw.rect(
            self.display, self.COLORS["accent"], box_rect, 3, border_radius=16
        )

        # Trophy or draw icon
        icon_y = cy - 100
        if "Win" in self.winner_text:
            # Trophy
            pygame.draw.circle(self.display, (255, 215, 0), (cx, icon_y), 35)
            pygame.draw.rect(
                self.display,
                (255, 215, 0),
                (cx - 15, icon_y + 20, 30, 25),
                border_radius=3,
            )
        else:
            # Handshake for draw
            pygame.draw.circle(self.display, (150, 150, 150), (cx - 15, icon_y), 20)
            pygame.draw.circle(self.display, (150, 150, 150), (cx + 15, icon_y), 20)

        win_col = (255, 215, 0) if "Win" in self.winner_text else self.COLORS["text"]
        t1 = self.fonts["subheader"].render(self.winner_text, True, win_col)
        t2 = self.fonts["title"].render(
            self.end_reason, True, self.COLORS["text_muted"]
        )

        self.display.blit(t1, (cx - t1.get_width() // 2, cy - 20))
        self.display.blit(t2, (cx - t2.get_width() // 2, cy + 30))

        # New game button
        mx, my = pygame.mouse.get_pos()
        new_game_rect = pygame.Rect(cx - 100, cy + 90, 200, 50)
        self._draw_btn(new_game_rect, "New Game", mx, my)

        return new_game_rect

    def _draw_btn(self, rect, text, mx, my, active=False, icon=None):
        hover = rect.collidepoint(mx, my)
        color = (
            self.COLORS["accent_hover"]
            if active or hover
            else self.COLORS["btn_default"]
        )

        # Shadow
        shadow_rect = pygame.Rect(rect.x, rect.y + 3, rect.w, rect.h)
        pygame.draw.rect(self.display, (20, 20, 20), shadow_rect, border_radius=8)

        # Button
        pygame.draw.rect(self.display, color, rect, border_radius=8)

        if hover:
            pygame.draw.rect(
                self.display, self.COLORS["accent"], rect, 2, border_radius=8
            )

        # Text with icon
        if icon:
            full_text = f"{icon}  {text}"
        else:
            full_text = text

        t = self.fonts["normal"].render(full_text, True, self.COLORS["text"])
        self.display.blit(
            t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2)
        )

    def draw_menu(self):
        self.display.fill(self.COLORS["bg"])

        # Title with gradient effect
        title_y = 80
        t1 = self.fonts["header"].render("CHESS", True, self.COLORS["accent"])
        t2 = self.fonts["subheader"].render("ENGINE", True, self.COLORS["text_muted"])
        self.display.blit(t1, (self.width // 2 - t1.get_width() // 2, title_y))
        self.display.blit(t2, (self.width // 2 - t2.get_width() // 2, title_y + 80))

        mx, my = pygame.mouse.get_pos()

        # Mode selection buttons
        bw, bh = 350, 70
        cx = self.width // 2
        cy = self.height // 2 - 20

        pvp_rect = pygame.Rect(cx - bw // 2, cy - 60, bw, bh)
        ai_rect = pygame.Rect(cx - bw // 2, cy + 40, bw, bh)

        self._draw_btn(pvp_rect, "Player vs Player", mx, my)
        self._draw_btn(ai_rect, "Player vs AI", mx, my)

        # AI Settings (only show when hovering AI button)
        if ai_rect.collidepoint(mx, my) or hasattr(self, "_show_ai_settings"):
            self._show_ai_settings = True
            settings_y = cy + 140

            # Color selection
            color_label = self.fonts["normal"].render(
                "Play as:", True, self.COLORS["text"]
            )
            self.display.blit(color_label, (cx - bw // 2, settings_y))

            white_rect = pygame.Rect(cx - bw // 2 + 100, settings_y - 5, 100, 40)
            black_rect = pygame.Rect(cx - bw // 2 + 210, settings_y - 5, 100, 40)

            white_active = self.menu_selected_color == Color.WHITE
            black_active = self.menu_selected_color == Color.BLACK

            self._draw_btn(white_rect, "White", mx, my, white_active)
            self._draw_btn(black_rect, "Black", mx, my, black_active)

            # Depth slider
            if self.depth_slider is None:
                self.depth_slider = Slider(
                    cx - bw // 2, settings_y + 60, bw, 20, 1, 8, 5, "AI Depth"
                )

            self.depth_slider.draw(self.display, self.fonts, self.COLORS)

            return pvp_rect, ai_rect, white_rect, black_rect

        return pvp_rect, ai_rect, None, None

    def get_color_name(self, color):
        return "White" if color == Color.WHITE else "Black"

    def init_game(self):
        pieces, occ, side, castle, ep, ply = parse_fen(FEN)
        self.board = Board(pieces, occ, side, castle, ep, ply)
        self.selected = None
        self.valid_moves = []
        self.move_log = []
        self.history_states = []
        self.last_move = None
        self.thinking = False
        self.game_state = GameState.PLAYING

        # Set board flip based on player color
        if self.mode == GameMode.AI:
            self.board_flipped = self.player_color == Color.BLACK
        else:
            self.board_flipped = False

        with self.ai_queue.mutex:
            self.ai_queue.queue.clear()

    def handle_click(self, pos):
        if self.thinking or self.game_state != GameState.PLAYING:
            return

        sq = self.pixel_to_square(pos[0], pos[1])
        if sq is None:
            return

        if self.selected is None:
            is_white = self.board.side == Color.WHITE
            start, end = (0, 6) if is_white else (6, 12)
            has_piece = False
            for i in range(start, end):
                if (self.board.bitboard[i] >> sq) & 1:
                    has_piece = True
                    break

            if has_piece:
                self.selected = sq
                self.valid_moves = self.get_legal_moves_for_square(sq)
        else:
            if sq == self.selected:
                self.selected = None
                self.valid_moves = []
            else:
                chosen_move = None
                possible_moves = [
                    m for m in self.valid_moves if get_target_square(m) == sq
                ]

                if possible_moves:
                    chosen_move = possible_moves[0]
                    for pm in possible_moves:
                        if get_flag(pm) in [
                            Flag.QUEEN_PROMOTION,
                            Flag.CAPTURE_PROMOTION_QUEEN,
                        ]:
                            chosen_move = pm
                            break

                    self.push_move(chosen_move)
                else:
                    self.selected = None
                    self.valid_moves = []
                    self.handle_click(pos)

    def run(self):
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()

        running = True
        new_game_rect = None

        while running:
            # Handle AI move completion
            if self.mode == GameMode.AI and self.thinking:
                try:
                    m = self.ai_queue.get_nowait()
                    self.thinking = False

                    if m is None:
                        if self.game_state == GameState.PLAYING:
                            self.check_game_over()
                    else:
                        self.push_move(m)
                except queue.Empty:
                    pass

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Handle slider events
                if self.mode == GameMode.MENU and self.depth_slider:
                    if self.depth_slider.handle_event(event):
                        self.ai_depth = self.depth_slider.value

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.mode == GameMode.MENU:
                        result = self.draw_menu()
                        pvp_rect, ai_rect = result[0], result[1]

                        if pvp_rect.collidepoint(event.pos):
                            self.mode = GameMode.PVP
                            self.player_color = Color.WHITE
                            self.init_game()
                        elif ai_rect.collidepoint(event.pos):
                            self._show_ai_settings = True

                        # Handle color selection
                        if len(result) > 2 and result[2] is not None:
                            white_rect, black_rect = result[2], result[3]
                            if white_rect.collidepoint(event.pos):
                                self.menu_selected_color = Color.WHITE
                            elif black_rect.collidepoint(event.pos):
                                self.menu_selected_color = Color.BLACK

                            # Start game if settings visible and AI rect clicked again
                            if ai_rect.collidepoint(event.pos) and hasattr(
                                self, "_show_ai_settings"
                            ):
                                self.mode = GameMode.AI
                                self.player_color = self.menu_selected_color
                                if self.depth_slider:
                                    self.ai_depth = self.depth_slider.value
                                delattr(self, "_show_ai_settings")
                                self.init_game()

                    elif self.game_state != GameState.PLAYING and new_game_rect:
                        if new_game_rect.collidepoint(event.pos):
                            self.mode = GameMode.MENU
                            self.depth_slider = None

                    else:
                        flip_btn, undo_btn, menu_btn = self.draw_panel()

                        if menu_btn.collidepoint(event.pos):
                            self.mode = GameMode.MENU
                            self.depth_slider = None
                        elif flip_btn.collidepoint(event.pos):
                            self.board_flipped = not self.board_flipped
                        elif undo_btn.collidepoint(event.pos):
                            if (
                                self.game_state == GameState.PLAYING
                                and not self.thinking
                            ):
                                self.undo_move()
                                if self.mode == GameMode.AI:
                                    self.undo_move()
                        else:
                            self.handle_click(event.pos)

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_u and self.mode != GameMode.MENU:
                        if not self.thinking and self.game_state == GameState.PLAYING:
                            self.undo_move()
                            if self.mode == GameMode.AI:
                                self.undo_move()
                    elif event.key == pygame.K_f and self.mode != GameMode.MENU:
                        self.board_flipped = not self.board_flipped

            # Trigger AI move if needed
            if (
                self.mode == GameMode.AI
                and self.game_state == GameState.PLAYING
                and not self.thinking
                and self.board
            ):
                # AI plays when it's not the player's turn
                if self.board.side != self.player_color:
                    self.start_ai_turn()

            # Render
            if self.mode == GameMode.MENU:
                self.draw_menu()
            else:
                self.display.fill(self.COLORS["bg"])
                self.draw_board()
                self.draw_pieces()
                self.draw_panel()
                new_game_rect = self.draw_game_over_overlay()

            pygame.display.flip()
            self.clock.tick(60)

        profiler.disable()
        profiler.dump_stats("gui_loop.prof")

        pygame.quit()
        sys.exit()




if __name__ == "__main__":
    if not os.path.exists("images"):
        os.makedirs("images")
        print("Tip: Add piece images to 'images/' folder for better visuals.")

    gui = ChessGUI(width=1600, height=1000)
    gui.run()
