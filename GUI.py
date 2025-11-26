import sys
import os
import threading
import queue
import time
import pygame
from enum import Enum
from typing import Optional, Callable, Dict, Tuple, List
from random import choice


from Board import Board, parse_fen, is_square_attacked
from Board_Move_gen import (
    Move_generator,
    Move,
    unmove,
    get_start_square,
    get_target_square,
    move_to_uci,
    get_flag,
)
from Constants import Pieces, Color, Flag
from pregen.Utilities import get_lsb1_index


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
        self.cache: Dict[str, pygame.Surface] = {}
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


class ChessGUI:
    COLORS = {
        "light_sq": (235, 236, 208),
        "dark_sq": (119, 149, 86),
        "bg": (49, 46, 43),
        "panel": (38, 36, 33),
        "highlight": (255, 255, 0, 100),
        "selected": (186, 202, 68, 180),
        "last_move": (245, 246, 130, 120),
        "text": (240, 240, 240),
        "text_muted": (160, 160, 160),
        "btn_default": (60, 60, 60),
        "btn_hover": (80, 80, 80),
        "accent": (129, 182, 76),
        "overlay": (0, 0, 0, 180),
    }

    def __init__(
        self, width: int = 1200, height: int = 800, ai_func: Optional[Callable] = None
    ):
        pygame.init()
        self.display = pygame.display.set_mode(
            (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Chess Engine - JIT Enabled")

        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()

        self.fonts = {
            "header": pygame.font.SysFont("Segoe UI", 60, bold=True),
            "subheader": pygame.font.SysFont("Segoe UI", 40, bold=True),
            "title": pygame.font.SysFont("Segoe UI", 28, bold=True),
            "normal": pygame.font.SysFont("Segoe UI", 20),
            "small": pygame.font.SysFont("Consolas", 14),
            "coords": pygame.font.SysFont("Segoe UI", 12, bold=True),
        }

        self.board_size = min(height - 40, width - 350)
        self.square_size = self.board_size // 8
        self.offset_x = 30
        self.offset_y = (height - self.board_size) // 2

        self.piece_cache = PieceImageCache(self.square_size)
        self.ai_func = ai_func

        self.ai_queue = queue.Queue()
        self.thinking = False

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

        self.warmup_engine()

    def warmup_engine(self):
        """Forces Numba compilation so the game doesn't freeze on the first move."""
        self.display.fill(self.COLORS["bg"])
        text = self.fonts["header"].render(
            "Loading Engine...", True, self.COLORS["text"]
        )
        self.display.blit(
            text, (self.width // 2 - text.get_width() // 2, self.height // 2)
        )
        pygame.display.flip()

        fen = b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        pieces, occ, side, castle, ep = parse_fen(fen)
        dummy_board = Board(pieces, occ, side, castle, ep)
        Move_generator(dummy_board)
        is_square_attacked(dummy_board, 0, 0)
        print("Warmup complete.")

    def pixel_to_square(self, x: int, y: int) -> Optional[int]:
        x -= self.offset_x
        y -= self.offset_y
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return (y // self.square_size) * 8 + (x // self.square_size)
        return None

    def square_to_pixel(self, square: int) -> Tuple[int, int]:
        rank, file = divmod(square, 8)
        return (
            self.offset_x + file * self.square_size + self.square_size // 2,
            self.offset_y + rank * self.square_size + self.square_size // 2,
        )

    def get_legal_moves_for_square(self, square: int) -> list:
        moves = Move_generator(self.board)
        prev_c, prev_ep = self.board.castle, self.board.enpassant
        legal_moves = []
        for i in range(moves.counter):
            m = moves.moves[i]
            if get_start_square(m) == square:
                if Move(self.board, m):
                    unmove(self.board, m, prev_c, prev_ep)
                    legal_moves.append(m)
        return legal_moves

    def check_game_over(self):
        """
        Check if the current player (self.board.side) has any legal moves.
        If not -> Checkmate or Stalemate.
        """
        moves = Move_generator(self.board)
        prev_c, prev_ep = self.board.castle, self.board.enpassant
        has_legal_move = False

        for i in range(moves.counter):
            m = moves.moves[i]
            if Move(self.board, m):
                has_legal_move = True
                unmove(self.board, m, prev_c, prev_ep)
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
        threading.Thread(target=self._run_ai_logic, daemon=True).start()

    def _run_ai_logic(self):
        try:
            board_copy = self.board.copy()
            move = self.ai_func(board_copy)
            self.ai_queue.put(move)
        except Exception as e:
            print(f"AI Error: {e}")
            self.ai_queue.put(None)

    def draw_board(self):
        for r in range(8):
            for f in range(8):
                color = (
                    self.COLORS["light_sq"]
                    if (r + f) % 2 == 0
                    else self.COLORS["dark_sq"]
                )
                x = self.offset_x + f * self.square_size
                y = self.offset_y + r * self.square_size
                pygame.draw.rect(
                    self.display, color, (x, y, self.square_size, self.square_size)
                )

                if f == 0:
                    c = (
                        self.COLORS["dark_sq"]
                        if color == self.COLORS["light_sq"]
                        else self.COLORS["light_sq"]
                    )
                    t = self.fonts["coords"].render(str(8 - r), True, c)
                    self.display.blit(t, (x + 2, y + 2))
                if r == 7:
                    c = (
                        self.COLORS["dark_sq"]
                        if color == self.COLORS["light_sq"]
                        else self.COLORS["light_sq"]
                    )
                    t = self.fonts["coords"].render(chr(97 + f), True, c)
                    self.display.blit(
                        t, (x + self.square_size - 10, y + self.square_size - 16)
                    )

        if self.last_move:
            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            s.fill(self.COLORS["last_move"])
            for sq in self.last_move:
                r, f = divmod(sq, 8)
                self.display.blit(
                    s,
                    (
                        self.offset_x + f * self.square_size,
                        self.offset_y + r * self.square_size,
                    ),
                )

        if self.selected is not None:
            r, f = divmod(self.selected, 8)
            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            s.fill(self.COLORS["selected"])
            self.display.blit(
                s,
                (
                    self.offset_x + f * self.square_size,
                    self.offset_y + r * self.square_size,
                ),
            )

        for move in self.valid_moves:
            target = get_target_square(move)
            r, f = divmod(target, 8)
            cx, cy = self.square_to_pixel(target)

            is_capture = False
            for i in range(12):
                if (self.board.bitboard[i] >> target) & 1:
                    is_capture = True
                    break

            s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
            if is_capture:
                pygame.draw.circle(
                    s,
                    (0, 0, 0, 50),
                    (self.square_size // 2, self.square_size // 2),
                    self.square_size // 2,
                    4,
                )
            else:
                pygame.draw.circle(
                    s,
                    (0, 0, 0, 50),
                    (self.square_size // 2, self.square_size // 2),
                    self.square_size // 7,
                )

            self.display.blit(
                s,
                (
                    self.offset_x + f * self.square_size,
                    self.offset_y + r * self.square_size,
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
        panel_x = self.offset_x + self.board_size + 20
        panel_w = self.width - panel_x - 20

        pygame.draw.rect(
            self.display,
            self.COLORS["panel"],
            (panel_x, self.offset_y, panel_w, self.board_size),
            border_radius=8,
        )

        if self.game_state == GameState.PLAYING:
            turn_str = (
                "White to Move" if self.board.side == Color.WHITE else "Black to Move"
            )
            if self.thinking:
                turn_str = "AI Thinking..."
            col = self.COLORS["accent"] if self.thinking else self.COLORS["text"]
        else:
            turn_str = "Game Over"
            col = (200, 100, 100)

        txt = self.fonts["title"].render(turn_str, True, col)
        self.display.blit(txt, (panel_x + 20, self.offset_y + 20))

        pygame.draw.line(
            self.display,
            (60, 60, 60),
            (panel_x + 15, self.offset_y + 60),
            (panel_x + panel_w - 15, self.offset_y + 60),
            2,
        )

        history_start = max(0, len(self.move_log) - 18)
        y_pos = self.offset_y + 75
        for i, m in enumerate(self.move_log[history_start:]):
            num = history_start + i + 1
            row_color = self.COLORS["text"] if i % 2 == 0 else self.COLORS["text_muted"]
            t = self.fonts["normal"].render(f"{num}. {m}", True, row_color)
            self.display.blit(t, (panel_x + 20, y_pos))
            y_pos += 26

        mx, my = pygame.mouse.get_pos()
        undo_rect = pygame.Rect(
            panel_x + 20, self.offset_y + self.board_size - 130, panel_w - 40, 50
        )
        menu_rect = pygame.Rect(
            panel_x + 20, self.offset_y + self.board_size - 70, panel_w - 40, 50
        )

        self._draw_btn(undo_rect, "Undo Move", mx, my)
        self._draw_btn(menu_rect, "Main Menu", mx, my)

        return undo_rect, menu_rect

    def draw_game_over_overlay(self):
        if self.game_state == GameState.PLAYING:
            return

        s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        s.fill(self.COLORS["overlay"])
        self.display.blit(s, (0, 0))

        box_w, box_h = 400, 250
        cx, cy = self.width // 2, self.height // 2
        box_rect = pygame.Rect(cx - box_w // 2, cy - box_h // 2, box_w, box_h)

        pygame.draw.rect(self.display, self.COLORS["panel"], box_rect, border_radius=12)
        pygame.draw.rect(self.display, (100, 100, 100), box_rect, 2, border_radius=12)

        win_col = (255, 215, 0) if "Win" in self.winner_text else self.COLORS["text"]
        t1 = self.fonts["subheader"].render(self.winner_text, True, win_col)
        t2 = self.fonts["title"].render(
            self.end_reason, True, self.COLORS["text_muted"]
        )

        self.display.blit(t1, (cx - t1.get_width() // 2, cy - 60))
        self.display.blit(t2, (cx - t2.get_width() // 2, cy - 10))

    def _draw_btn(self, rect, text, mx, my, active=False):
        hover = rect.collidepoint(mx, my)
        color = (
            self.COLORS["accent"]
            if active
            else (self.COLORS["btn_hover"] if hover else self.COLORS["btn_default"])
        )

        pygame.draw.rect(
            self.display,
            (20, 20, 20),
            (rect.x, rect.y + 4, rect.w, rect.h),
            border_radius=8,
        )

        pygame.draw.rect(self.display, color, rect, border_radius=8)

        t = self.fonts["normal"].render(text, True, self.COLORS["text"])
        self.display.blit(
            t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2)
        )

    def draw_menu(self):
        self.display.fill(self.COLORS["bg"])

        t = self.fonts["header"].render("CHESS ENGINE", True, self.COLORS["text"])
        self.display.blit(t, (self.width // 2 - t.get_width() // 2, 120))

        mx, my = pygame.mouse.get_pos()
        bw, bh = 300, 60
        cx = self.width // 2
        cy = self.height // 2

        pvp = pygame.Rect(cx - bw // 2, cy - 40, bw, bh)
        ai = pygame.Rect(cx - bw // 2, cy + 60, bw, bh)

        self._draw_btn(pvp, "Player vs Player", mx, my)
        self._draw_btn(ai, "Player vs AI", mx, my)

        return pvp, ai

    def init_game(self):
        fen = b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        pieces, occ, side, castle, ep = parse_fen(fen)
        self.board = Board(pieces, occ, side, castle, ep)
        self.selected = None
        self.valid_moves = []
        self.move_log = []
        self.history_states = []
        self.last_move = None
        self.thinking = False
        self.game_state = GameState.PLAYING
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
        running = True
        while running:
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

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.mode == GameMode.MENU:
                        pvp, ai = self.draw_menu()
                        if pvp.collidepoint(event.pos):
                            self.mode = GameMode.PVP
                            self.init_game()
                        elif ai.collidepoint(event.pos):
                            self.mode = GameMode.AI
                            self.init_game()
                    else:
                        undo_btn, menu_btn = self.draw_panel()

                        if menu_btn.collidepoint(event.pos):
                            self.mode = GameMode.MENU
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

            if (
                self.mode == GameMode.AI
                and self.game_state == GameState.PLAYING
                and not self.thinking
                and self.board
            ):
                if self.board.side == Color.BLACK:
                    self.start_ai_turn()

            if self.mode == GameMode.MENU:
                self.draw_menu()
            else:
                self.display.fill(self.COLORS["bg"])
                self.draw_board()
                self.draw_pieces()
                self.draw_panel()
                self.draw_game_over_overlay()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


def simple_ai(board: Board):
    time.sleep(0.2)
    moves = Move_generator(board)
    prev_c, prev_ep = board.castle, board.enpassant
    valid = []
    for i in range(moves.counter):
        m = moves.moves[i]
        if Move(board, m):
            unmove(board, m, prev_c, prev_ep)
            valid.append(m)
    if not valid:
        return None
    return choice(valid)



if __name__ == "__main__":
    if not os.path.exists("images"):
        os.makedirs("images")
        print("Tip: Add piece images to 'images/' folder for better visuals.")

    gui = ChessGUI(width=1600, height=1000, ai_func=simple_ai)
    gui.run()
