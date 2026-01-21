import os
import queue
import random
import sys
import threading
import time
from enum import Enum

import msgpack
import pygame
from IPython import embed

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

# Load opening book
try:
    with open(r"Process Opening\opening_tree.msgpack", "rb") as f:
        OPENING_TREE = msgpack.unpack(f, raw=False)
    OPENING_BOOK_LOADED = True
except:
    OPENING_TREE = {}
    OPENING_BOOK_LOADED = False


class FENS:
    WKQ_VS_BK = b"6k1/8/8/8/8/8/6K1/6Q1 w - - 0 1"
    BKQ_VS_WK = b"6q1/6k1/8/8/8/8/8/6K1 b - - 0 1"

    WKR_VS_BK = b"6k1/8/8/8/8/8/6K1/6R1 w - - 0 1"
    BKR_VS_WK = b"6r1/6k1/8/8/8/8/8/6K1 b - - 0 1"

    WKRR_VS_BK = b"6k1/8/8/8/8/8/6K1/5RR1 w - - 0 1"
    BKRR_VS_WK = b"5rr1/6k1/8/8/8/8/8/6K1 b - - 0 1"

    WKQQ_VS_BK = b"6k1/8/8/8/8/8/6K1/5QQ1 w - - 0 1"
    BKQQ_VS_WK = b"5qq1/6k1/8/8/8/8/8/6K1 b - - 0 1"

    WKBB_VS_BK = b"6k1/8/8/8/8/8/6K1/5BB1 w - - 0 1"
    BKBB_VS_WK = b"5bb1/6k1/8/8/8/8/8/6K1 b - - 0 1"

    WKBN_VS_BK = b"6k1/8/8/8/8/8/6K1/5BN1 w - - 0 1"
    BKBN_VS_WK = b"5bn1/6k1/8/8/8/8/8/6K1 b - - 0 1"


STARTING_FEN = b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# STARTING_FEN = FENS.BKQ_VS_WK


class GameMode(Enum):
    MENU = 0
    PVP = 1
    AI = 2


class GameState(Enum):
    PLAYING = 0
    CHECKMATE = 1
    STALEMATE = 2
    DRAW = 3


class OpeningNavigator:
    """Navigates opening tree and provides book moves"""

    def __init__(self, tree):
        self.tree = tree
        self.reset()

    def reset(self):
        self.current = self.tree
        self.history = []
        self.active = True

    def apply(self, uci_move: str) -> bool:
        """Apply move, return True if still in book"""
        if not self.active:
            return False

        if uci_move in self.current:
            self.current = self.current[uci_move]
            self.history.append(uci_move)
            return True
        else:
            self.active = False
            return False

    def get_options(self) -> list:
        """Available book moves from current position"""
        if not self.active or not isinstance(self.current, dict):
            return []
        return list(self.current.keys())

    def select_random(self) -> str | None:
        """Random book move for AI"""
        opts = self.get_options()
        return random.choice(opts) if opts else None


class PieceCache:
    """Piece image caching with fallback rendering"""

    def __init__(self, size: int = 80):
        self.size = size
        self.cache = {}
        self.names = ["pawn", "knight", "bishop", "rook", "queen", "king"] * 2
        self.colors = ["white"] * 6 + ["black"] * 6

    def get(self, idx: int) -> pygame.Surface:
        key = f"{self.colors[idx]}_{self.names[idx]}"
        if key not in self.cache:
            path = f"images/{key}.png"
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path)
                    self.cache[key] = pygame.transform.smoothscale(
                        img, (self.size, self.size)
                    )
                except:
                    self.cache[key] = self._fallback(self.names[idx], self.colors[idx])
            else:
                self.cache[key] = self._fallback(self.names[idx], self.colors[idx])
        return self.cache[key]

    def _fallback(self, name, color):
        s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        font = pygame.font.SysFont("Arial", self.size // 2, bold=True)

        fg = (255, 255, 255) if color == "white" else (0, 0, 0)
        bg = (220, 220, 220) if color == "white" else (60, 60, 60)
        border = (0, 0, 0) if color == "white" else (255, 255, 255)

        pygame.draw.circle(s, bg, (self.size // 2, self.size // 2), self.size // 2 - 5)
        pygame.draw.circle(
            s, border, (self.size // 2, self.size // 2), self.size // 2 - 5, 2
        )

        txt = font.render(name[0].upper(), True, fg)
        s.blit(
            txt,
            (
                self.size // 2 - txt.get_width() // 2,
                self.size // 2 - txt.get_height() // 2,
            ),
        )
        return s


class Slider:
    """Depth selection slider"""

    def __init__(self, x, y, w, h, min_v, max_v, init, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min, self.max, self.val = min_v, max_v, init
        self.label = label
        self.drag = False
        self.radius = h // 2 + 2

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            hx = (
                self.rect.x
                + ((self.val - self.min) / (self.max - self.min)) * self.rect.width
            )
            if ((mx - hx) ** 2 + (my - self.rect.centery) ** 2) <= self.radius**2:
                self.drag = True
                return True
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.drag = False
        elif ev.type == pygame.MOUSEMOTION and self.drag:
            ratio = max(0, min(1, (ev.pos[0] - self.rect.x) / self.rect.width))
            self.val = int(self.min + ratio * (self.max - self.min))
            return True
        return False

    def draw(self, surf, fonts, cols):
        track = pygame.Rect(self.rect.x, self.rect.centery - 3, self.rect.width, 6)
        pygame.draw.rect(surf, (60, 60, 60), track, border_radius=3)

        fill_w = ((self.val - self.min) / (self.max - self.min)) * self.rect.width
        fill = pygame.Rect(self.rect.x, self.rect.centery - 3, fill_w, 6)
        pygame.draw.rect(surf, cols["accent"], fill, border_radius=3)

        hx = self.rect.x + fill_w
        pygame.draw.circle(surf, (40, 40, 40), (hx, self.rect.centery + 2), self.radius)
        pygame.draw.circle(
            surf,
            cols["accent"] if self.drag else (200, 200, 200),
            (hx, self.rect.centery),
            self.radius,
        )

        lbl = fonts["small"].render(f"{self.label}: {self.val}", True, cols["text"])
        surf.blit(lbl, (self.rect.x, self.rect.y - 25))


class ChessGUI:
    C = {
        "light_sq": (240, 217, 181),
        "dark_sq": (181, 136, 99),
        "bg": (18, 18, 22),
        "panel": (28, 28, 34),
        "panel_dark": (18, 18, 22),
        "selected": (130, 151, 105, 200),
        "last_move": (205, 210, 106, 140),
        "text": (240, 240, 245),
        "text_muted": (155, 155, 165),
        "btn": (40, 40, 48),
        "btn_hover": (55, 55, 63),
        "accent": (76, 158, 255),
        "accent_hover": (96, 178, 255),
        "overlay": (0, 0, 0, 200),
        "check": (220, 70, 70, 180),
        "book": (76, 175, 80),
        "engine": (255, 152, 0),
    }

    def __init__(self, w=1600, h=1000):
        pygame.init()
        self.disp = pygame.display.set_mode((w, h), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Chess Engine")

        self.w, self.h = w, h
        self.clock = pygame.time.Clock()

        self.fonts = {
            "header": pygame.font.SysFont("Segoe UI", 72, bold=True),
            "subhead": pygame.font.SysFont("Segoe UI", 48, bold=True),
            "title": pygame.font.SysFont("Segoe UI", 32, bold=True),
            "normal": pygame.font.SysFont("Segoe UI", 22),
            "small": pygame.font.SysFont("Segoe UI", 16),
            "tiny": pygame.font.SysFont("Segoe UI", 14),
            "coord": pygame.font.SysFont("Segoe UI", 13, bold=True),
        }

        self.bsize = min(h - 80, w - 450)
        self.sqsize = self.bsize // 8
        self.offx, self.offy = 50, (h - self.bsize) // 2

        self.pieces = PieceCache(self.sqsize)
        self.openings = OpeningNavigator(OPENING_TREE)

        self.aiqueue = queue.Queue()
        self.thinking, self.aistart = False, 0

        self.mode = GameMode.MENU
        self.board, self.sel, self.valids = None, None, []
        self.log, self.states, self.lastmv = [], [], None
        self.gstate, self.winner, self.reason = GameState.PLAYING, "", ""

        self.pcolor, self.depth, self.flipped = Color.WHITE, 5, False
        self.mcolor, self.slider = Color.WHITE, None

        self._warmup()

    def check_repetition(self):
        """Check for threefold repetition draw"""
        if len(self.log) < 8:  # Need at least 8 moves for repetition
            return False

        current_hash = self.board.hash
        repetitions = 1  # Count current position

        # Count how many times current position occurred in game history
        for state in self.states:
            if state["board"].hash == current_hash:
                repetitions += 1
                if repetitions >= 3:
                    return True

        return False

    def _warmup(self):
        """JIT compilation warmup"""
        self.disp.fill(self.C["bg"])
        for i in range(3):
            self.disp.fill(self.C["bg"])
            t = self.fonts["title"].render(
                f"Initializing{'.' * (i + 1)}", True, self.C["accent"]
            )
            self.disp.blit(t, (self.w // 2 - t.get_width() // 2, self.h // 2))
            pygame.display.flip()
            time.sleep(0.2)

        p, o, s, c, e, pl = parse_fen(STARTING_FEN)
        b = Board(p, o, s, c, e, pl)
        print(pl,b.halfmove)
        Move_generator(b)
        is_square_attacked(b, 0, 0)
        AI(b, s, 5)

        self.disp.fill(self.C["bg"])
        msg = "Opening Book Loaded!" if OPENING_BOOK_LOADED else "Ready (No Book)"
        col = self.C["accent"] if OPENING_BOOK_LOADED else self.C["text_muted"]
        t = self.fonts["title"].render(msg, True, col)
        self.disp.blit(t, (self.w // 2 - t.get_width() // 2, self.h // 2))
        pygame.display.flip()
        time.sleep(0.4)

    def px2sq(self, x, y):
        x, y = x - self.offx, y - self.offy
        if 0 <= x < self.bsize and 0 <= y < self.bsize:
            f, r = x // self.sqsize, y // self.sqsize
            if self.flipped:
                r, f = 7 - r, 7 - f
            return r * 8 + f
        return None

    def sq2px(self, sq):
        r, f = divmod(sq, 8)
        if self.flipped:
            r, f = 7 - r, 7 - f
        return (
            self.offx + f * self.sqsize + self.sqsize // 2,
            self.offy + r * self.sqsize + self.sqsize // 2,
        )

    def legal_for(self, sq):
        mvs = Move_generator(self.board)
        pc, pe, pp = self.board.castle, self.board.enpassant, self.board.halfmove
        legal = []
        for i in range(mvs.counter):
            m = mvs.moves[i]
            if get_start_square(m) == sq:
                if Move(self.board, m):
                    unmove(self.board, m, pc, pe, pp)
                    legal.append(m)
        return legal

    def check_end(self):
        mvs = Move_generator(self.board)
        pc, pe, pp = self.board.castle, self.board.enpassant, self.board.halfmove
        has_legal = False

        for i in range(mvs.counter):
            if Move(self.board, mvs.moves[i]):
                has_legal = True
                unmove(self.board, mvs.moves[i], pc, pe, pp)
                break

        if not has_legal:
            kp = Pieces.K.value if self.board.side == Color.WHITE else Pieces.k.value
            ksq = get_lsb1_index(self.board.bitboard[kp])
            att = Color.BLACK if self.board.side == Color.WHITE else Color.WHITE

            if is_square_attacked(self.board, ksq, att):
                self.gstate = GameState.CHECKMATE
                self.winner = (
                    "White Wins!" if self.board.side == Color.BLACK else "Black Wins!"
                )
                self.reason = "Checkmate"
            elif self.board.halfmove >= 50:
                self.gstate, self.winner, self.reason = (
                    GameState.DRAW,
                    "Draw",
                    "50-Move Rule",
                )
            else:
                self.gstate, self.winner, self.reason = (
                    GameState.STALEMATE,
                    "Draw",
                    "Stalemate",
                )

    def push(self, mv):
        self.states.append(
            {
                "board": self.board.copy(),
                "lastmv": self.lastmv,
                "log": self.log[:],
                "gstate": self.gstate,
                "opnav": self._copy_opnav(),
            }
        )

        if Move(self.board, mv):
            uci = move_to_uci(mv)
            self.lastmv = (get_start_square(mv), get_target_square(mv))
            self.log.append(uci)
            self.openings.apply(uci)
            self.sel, self.valids = None, []
            self.check_end()
            return True
        return False

    def _copy_opnav(self):
        nav = OpeningNavigator(OPENING_TREE)
        nav.current, nav.history, nav.active = (
            self.openings.current,
            self.openings.history[:],
            self.openings.active,
        )
        return nav

    def undo(self):
        if not self.states:
            return
        st = self.states.pop()
        self.board, self.lastmv, self.log = st["board"], st["lastmv"], st["log"]
        self.gstate, self.openings = st["gstate"], st["opnav"]
        self.sel, self.valids, self.thinking = None, [], False

    def start_ai(self):
        if self.thinking or self.gstate != GameState.PLAYING:
            return
        self.thinking, self.aistart = True, time.time()
        threading.Thread(target=self._ai_logic, daemon=True).start()

    def _ai_logic(self):
        try:
            book_uci = self.openings.select_random() if self.openings.active else None

            if book_uci:
                mvs = Move_generator(self.board)
                pc, pe, pp = (
                    self.board.castle,
                    self.board.enpassant,
                    self.board.halfmove,
                )
                for i in range(mvs.counter):
                    m = mvs.moves[i]
                    if move_to_uci(m) == book_uci:
                        if Move(self.board, m):
                            unmove(self.board, m, pc, pe, pp)
                            self.aiqueue.put(("book", m))
                            return
                        unmove(self.board, m, pc, pe, pp)

            bc = self.board.copy()
            mv = AI(bc, bc.side, self.depth)
            self.aiqueue.put(("engine", mv))
        except Exception as e:
            print(f"AI error: {e}")
            self.aiqueue.put(("error", None))


    def draw_board(self):
        for r in range(8):
            for f in range(8):
                ar, af = (7 - r, 7 - f) if self.flipped else (r, f)
                col = self.C["light_sq"] if (ar + af) % 2 == 0 else self.C["dark_sq"]
                x, y = self.offx + f * self.sqsize, self.offy + r * self.sqsize
                pygame.draw.rect(self.disp, col, (x, y, self.sqsize, self.sqsize))

                ccol = (
                    self.C["dark_sq"]
                    if col == self.C["light_sq"]
                    else self.C["light_sq"]
                )
                if f == 0:
                    t = self.fonts["coord"].render(str(8 - ar), True, ccol)
                    self.disp.blit(t, (x + 4, y + 4))
                if r == 7:
                    t = self.fonts["coord"].render(chr(97 + af), True, ccol)
                    self.disp.blit(t, (x + self.sqsize - 14, y + self.sqsize - 18))

        if self.lastmv:
            s = pygame.Surface((self.sqsize, self.sqsize), pygame.SRCALPHA)
            s.fill(self.C["last_move"])
            for sq in self.lastmv:
                r, f = divmod(sq, 8)
                if self.flipped:
                    r, f = 7 - r, 7 - f
                self.disp.blit(
                    s, (self.offx + f * self.sqsize, self.offy + r * self.sqsize)
                )

        if self.sel is not None:
            r, f = divmod(self.sel, 8)
            if self.flipped:
                r, f = 7 - r, 7 - f
            s = pygame.Surface((self.sqsize, self.sqsize), pygame.SRCALPHA)
            s.fill(self.C["selected"])
            self.disp.blit(
                s, (self.offx + f * self.sqsize, self.offy + r * self.sqsize)
            )

        for mv in self.valids:
            tgt = get_target_square(mv)
            r, f = divmod(tgt, 8)
            if self.flipped:
                r, f = 7 - r, 7 - f
            cap = any((self.board.bitboard[i] >> tgt) & 1 for i in range(12))
            s = pygame.Surface((self.sqsize, self.sqsize), pygame.SRCALPHA)
            if cap:
                pygame.draw.circle(
                    s,
                    (90, 90, 90, 120),
                    (self.sqsize // 2, self.sqsize // 2),
                    self.sqsize // 2 - 4,
                    5,
                )
            else:
                pygame.draw.circle(
                    s,
                    (70, 70, 70, 120),
                    (self.sqsize // 2, self.sqsize // 2),
                    self.sqsize // 6,
                )
            self.disp.blit(
                s, (self.offx + f * self.sqsize, self.offy + r * self.sqsize)
            )

    def draw_pieces(self):
        for i in range(12):
            bb, sq = self.board.bitboard[i], 0
            while bb:
                if bb & 1:
                    x, y = self.sq2px(sq)
                    img = self.pieces.get(i)
                    self.disp.blit(img, img.get_rect(center=(x, y)))
                bb, sq = bb >> 1, sq + 1

    def draw_panel(self):
        px = self.offx + self.bsize + 30
        pw = self.w - px - 40
        pygame.draw.rect(
            self.disp,
            self.C["panel"],
            (px, self.offy, pw, self.bsize),
            border_radius=16,
        )

        y = self.offy + 25

        if self.gstate == GameState.PLAYING:
            turn = (
                "White to Move" if self.board.side == Color.WHITE else "Black to Move"
            )
            if self.thinking:
                turn = f"AI Thinking... ({time.time() - self.aistart:.1f}s)"
                col = self.C["accent"]
            else:
                col = self.C["text"]

            if OPENING_BOOK_LOADED:
                badge = pygame.Surface((125, 30), pygame.SRCALPHA)
                if self.openings.active:
                    pygame.draw.rect(
                        badge, (*self.C["book"], 200), (0, 0, 125, 30), border_radius=15
                    )
                    bt = self.fonts["tiny"].render("IN BOOK", True, (255, 255, 255))
                else:
                    pygame.draw.rect(
                        badge,
                        (*self.C["engine"], 200),
                        (0, 0, 125, 30),
                        border_radius=15,
                    )
                    bt = self.fonts["tiny"].render("ENGINE", True, (255, 255, 255))
                badge.blit(bt, (12, 7))
                self.disp.blit(badge, (px + pw - 140, y + 5))
        else:
            turn, col = "Game Over", (220, 100, 100)

        t = self.fonts["title"].render(turn, True, col)
        self.disp.blit(t, (px + 25, y))
        y += 60

        pygame.draw.line(self.disp, (60, 60, 65), (px + 20, y), (px + pw - 20, y), 2)
        y += 20

        mode_txt = (
            f"Mode: {self._cname(self.pcolor)} vs AI"
            if self.mode == GameMode.AI
            else "Mode: PvP"
        )
        t = self.fonts["small"].render(mode_txt, True, self.C["text_muted"])
        self.disp.blit(t, (px + 25, y))
        y += 28

        if self.mode == GameMode.AI:
            t = self.fonts["small"].render(
                f"AI Depth: {self.depth}", True, self.C["text_muted"]
            )
            self.disp.blit(t, (px + 25, y))
            y += 28

        t = self.fonts["small"].render(
            f"Moves: {len(self.log)}", True, self.C["text_muted"]
        )
        self.disp.blit(t, (px + 25, y))
        y += 40

        pygame.draw.line(self.disp, (60, 60, 65), (px + 20, y), (px + pw - 20, y), 2)
        y += 20

        t = self.fonts["normal"].render("Move History", True, self.C["text"])
        self.disp.blit(t, (px + 25, y))
        y += 40

        start = max(0, len(self.log) - 26)
        for i, m in enumerate(self.log[start:]):
            col = self.C["text"] if i % 2 == 0 else self.C["text_muted"]
            xp = px + 25 if i % 2 == 0 else px + pw // 2 + 15
            t = self.fonts["tiny"].render(f"{start + i + 1}. {m}", True, col)
            self.disp.blit(t, (xp, y))
            if i % 2 == 1:
                y += 24

        mx, my = pygame.mouse.get_pos()
        by = self.offy + self.bsize - 200
        flip_r = pygame.Rect(px + 25, by, pw - 50, 48)
        undo_r = pygame.Rect(px + 25, by + 58, pw - 50, 48)
        menu_r = pygame.Rect(px + 25, by + 116, pw - 50, 48)

        self._btn(flip_r, "Flip Board", mx, my)
        self._btn(undo_r, "Undo", mx, my)
        self._btn(menu_r, "Menu", mx, my)

        return flip_r, undo_r, menu_r

    def draw_end_overlay(self):
        if self.gstate == GameState.PLAYING:
            return None

        s = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        s.fill(self.C["overlay"])
        self.disp.blit(s, (0, 0))

        bw, bh = 520, 340
        cx, cy = self.w // 2, self.h // 2
        br = pygame.Rect(cx - bw // 2, cy - bh // 2, bw, bh)

        pygame.draw.rect(self.disp, self.C["panel"], br, border_radius=18)
        pygame.draw.rect(self.disp, self.C["accent"], br, 3, border_radius=18)

        iy = cy - 100
        if "Win" in self.winner:
            pygame.draw.circle(self.disp, (255, 215, 0), (cx, iy), 38)
            pygame.draw.rect(
                self.disp, (255, 215, 0), (cx - 16, iy + 22, 32, 28), border_radius=4
            )
        else:
            pygame.draw.circle(self.disp, (150, 150, 150), (cx - 18, iy), 22)
            pygame.draw.circle(self.disp, (150, 150, 150), (cx + 18, iy), 22)

        wcol = (255, 215, 0) if "Win" in self.winner else self.C["text"]
        t1 = self.fonts["subhead"].render(self.winner, True, wcol)
        t2 = self.fonts["title"].render(self.reason, True, self.C["text_muted"])

        self.disp.blit(t1, (cx - t1.get_width() // 2, cy - 20))
        self.disp.blit(t2, (cx - t2.get_width() // 2, cy + 30))

        mx, my = pygame.mouse.get_pos()
        ngr = pygame.Rect(cx - 110, cy + 95, 220, 52)
        self._btn(ngr, "New Game", mx, my)

        return ngr

    def _btn(self, r, txt, mx, my, act=False):
        hov = r.collidepoint(mx, my)
        col = self.C["accent_hover"] if (act or hov) else self.C["btn"]

        sh = pygame.Rect(r.x, r.y + 3, r.w, r.h)
        pygame.draw.rect(self.disp, (15, 15, 15), sh, border_radius=8)
        pygame.draw.rect(self.disp, col, r, border_radius=8)

        if hov:
            pygame.draw.rect(self.disp, self.C["accent"], r, 2, border_radius=8)

        t = self.fonts["normal"].render(txt, True, self.C["text"])
        self.disp.blit(
            t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2)
        )

    def draw_menu(self):
        self.disp.fill(self.C["bg"])

        t1 = self.fonts["header"].render("CHESS", True, self.C["accent"])
        t2 = self.fonts["subhead"].render("ENGINE", True, self.C["text_muted"])
        self.disp.blit(t1, (self.w // 2 - t1.get_width() // 2, 80))
        self.disp.blit(t2, (self.w // 2 - t2.get_width() // 2, 160))

        mx, my = pygame.mouse.get_pos()
        bw, bh = 360, 72
        cx, cy = self.w // 2, self.h // 2 - 20

        pvp_r = pygame.Rect(cx - bw // 2, cy - 60, bw, bh)
        ai_r = pygame.Rect(cx - bw // 2, cy + 40, bw, bh)

        self._btn(pvp_r, "Player vs Player", mx, my)
        self._btn(ai_r, "Player vs AI", mx, my)

        if ai_r.collidepoint(mx, my) or hasattr(self, "_show_ai"):
            self._show_ai = True
            sy = cy + 140

            lbl = self.fonts["normal"].render("Play as:", True, self.C["text"])
            self.disp.blit(lbl, (cx - bw // 2, sy))

            wr = pygame.Rect(cx - bw // 2 + 105, sy - 5, 105, 42)
            br = pygame.Rect(cx - bw // 2 + 220, sy - 5, 105, 42)

            self._btn(wr, "White", mx, my, self.mcolor == Color.WHITE)
            self._btn(br, "Black", mx, my, self.mcolor == Color.BLACK)

            if self.slider is None:
                self.slider = Slider(cx - bw // 2, sy + 65, bw, 20, 1, 8, 5, "AI Depth")

            self.slider.draw(self.disp, self.fonts, self.C)

            return pvp_r, ai_r, wr, br

        return pvp_r, ai_r, None, None

    def _cname(self, c):
        return "White" if c == Color.WHITE else "Black"

    def init_game(self):
        p, o, s, c, e, pl = parse_fen(STARTING_FEN)
        self.board = Board(p, o, s, c, e, pl)
        print(pl,self.board.halfmove)
        self.openings.reset()
        self.sel, self.valids = None, []
        self.log, self.states, self.lastmv = [], [], None
        self.thinking, self.gstate = False, GameState.PLAYING
        self.flipped = self.mode == GameMode.AI and self.pcolor == Color.BLACK

        with self.aiqueue.mutex:
            self.aiqueue.queue.clear()

    def handle_click(self, pos):
        if self.thinking or self.gstate != GameState.PLAYING:
            return

        sq = self.px2sq(pos[0], pos[1])
        if sq is None:
            return

        if self.sel is None:
            is_w = self.board.side == Color.WHITE
            start, end = (0, 6) if is_w else (6, 12)
            has = any((self.board.bitboard[i] >> sq) & 1 for i in range(start, end))

            if has:
                self.sel = sq
                self.valids = self.legal_for(sq)
        else:
            if sq == self.sel:
                self.sel, self.valids = None, []
            else:
                chosen = None
                possible = [m for m in self.valids if get_target_square(m) == sq]

                if possible:
                    chosen = possible[0]
                    for pm in possible:
                        if get_flag(pm) in [
                            Flag.QUEEN_PROMOTION,
                            Flag.CAPTURE_PROMOTION_QUEEN,
                        ]:
                            chosen = pm
                            break
                    self.push(chosen)
                else:
                    self.sel, self.valids = None, []
                    self.handle_click(pos)

    def run(self):
        running = True
        ng_rect = None

        while running:
            # Handle AI completion
            if self.mode == GameMode.AI and self.thinking:
                try:
                    src, m = self.aiqueue.get_nowait()
                    self.thinking = False

                    if m is None:
                        if self.gstate == GameState.PLAYING:
                            self.check_end()
                    else:
                        self.push(m)
                except queue.Empty:
                    pass

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False

                # Slider handling
                if self.mode == GameMode.MENU and self.slider:
                    if self.slider.handle(ev):
                        self.depth = self.slider.val

                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self.mode == GameMode.MENU:
                        res = self.draw_menu()
                        pvp_r, ai_r = res[0], res[1]

                        if pvp_r.collidepoint(ev.pos):
                            self.mode = GameMode.PVP
                            self.pcolor = Color.WHITE
                            self.init_game()
                        elif ai_r.collidepoint(ev.pos):
                            self._show_ai = True

                        if len(res) > 2 and res[2] is not None:
                            wr, br = res[2], res[3]
                            if wr.collidepoint(ev.pos):
                                self.mcolor = Color.WHITE
                            elif br.collidepoint(ev.pos):
                                self.mcolor = Color.BLACK

                            if ai_r.collidepoint(ev.pos) and hasattr(self, "_show_ai"):
                                self.mode = GameMode.AI
                                self.pcolor = self.mcolor
                                if self.slider:
                                    self.depth = self.slider.val
                                delattr(self, "_show_ai")
                                self.init_game()

                    elif self.gstate != GameState.PLAYING and ng_rect:
                        if ng_rect.collidepoint(ev.pos):
                            self.mode = GameMode.MENU
                            self.slider = None

                    else:
                        flip_b, undo_b, menu_b = self.draw_panel()

                        if menu_b.collidepoint(ev.pos):
                            self.mode = GameMode.MENU
                            self.slider = None
                        elif flip_b.collidepoint(ev.pos):
                            self.flipped = not self.flipped
                        elif undo_b.collidepoint(ev.pos):
                            if self.gstate == GameState.PLAYING and not self.thinking:
                                self.undo()
                                if self.mode == GameMode.AI:
                                    self.undo()
                        else:
                            self.handle_click(ev.pos)

                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_u and self.mode != GameMode.MENU:
                        if not self.thinking and self.gstate == GameState.PLAYING:
                            self.undo()
                            if self.mode == GameMode.AI:
                                self.undo()
                    elif ev.key == pygame.K_f and self.mode != GameMode.MENU:
                        self.flipped = not self.flipped

            # Trigger AI
            if (
                self.mode == GameMode.AI
                and self.gstate == GameState.PLAYING
                and not self.thinking
                and self.board
            ):
                if self.board.side != self.pcolor:
                    self.start_ai()

            # Render
            if self.mode == GameMode.MENU:
                self.draw_menu()
            else:
                self.disp.fill(self.C["bg"])
                self.draw_board()
                self.draw_pieces()
                self.draw_panel()
                ng_rect = self.draw_end_overlay()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


def interactive_shell():
    embed()


if __name__ == "__main__":
    if not os.path.exists("images"):
        os.makedirs("images")
        print("Tip: Add piece images to 'images/' folder for better visuals.")

    gui = ChessGUI(w=1600, h=1000)

    gui.run()
