import os
import queue
import random
import sys
import threading
import time
import multiprocessing as mp
from enum import Enum, auto
import datetime
from tkinter import filedialog, Tk

import msgpack
import pygame
import numpy as np
import chess
import chess.pgn

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
from Minimax import AI, SingleSearch
from pregen.Utilities import get_lsb1_index


try:
    with open(r"Process Opening\opening_tree.msgpack", "rb") as f:
        OPENING_TREE = msgpack.unpack(f, raw=False)
    OPENING_BOOK_LOADED = True
except:
    OPENING_TREE = {}
    OPENING_BOOK_LOADED = False


def board_to_state(board):
    return (
        board.bitboard.copy(),
        board.occupancy.copy(),
        np.uint8(board.side),
        np.uint8(board.castle),
        np.uint8(board.enpassant),
        np.uint8(board.halfmove),
    )


def state_to_board(state):
    p, o, s, c, e, pl = state
    return Board(p, o, s, c, e, pl)


def engine_loop(cmd_q, info_q, result_q):
    p, o, s, c, e, pl = parse_fen(STARTING_FEN)
    warm_board = Board(p, o, s, c, e, pl)
    Move_generator(warm_board)
    is_square_attacked(warm_board, 0, 0)
    SingleSearch(warm_board, depth=1).search()
    while True:
        cmd = cmd_q.get()
        if not cmd:
            continue
        if cmd[0] == "stop":
            break
        if cmd[0] == "search":
            _, state, depth, delay, sid = cmd
            if delay:
                time.sleep(delay)
            board = state_to_board(state)
            def info_cb(msg):
                info_q.put(msg)
            searcher = SingleSearch(board, depth=depth, info_callback=info_cb)
            best_move = searcher.search()
            result_q.put(("engine", int(best_move), sid))


def analysis_worker(state, info_q):
    board = state_to_board(state)
    def info_cb(msg):
        info_q.put(msg)
    searcher = SingleSearch(board, depth=24, info_callback=info_cb)
    searcher.search()


class FENS:
    START = b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


STARTING_FEN = FENS.START


class GameMode(Enum):
    MENU = auto()
    PVP = auto()
    P_VS_AI = auto()
    AI_VS_AI = auto()
    ANALYSIS = auto()


class GameState(Enum):
    PLAYING = auto()
    CHECKMATE = auto()
    STALEMATE = auto()
    DRAW = auto()


class MoveClass(Enum):
    BEST = ("Best", (39, 174, 96))      
    EXCELLENT = ("Excellent", (46, 204, 113)) 
    GOOD = ("Good", (52, 152, 219))     
    INACCURACY = ("Inaccuracy", (241, 196, 15)) 
    MISTAKE = ("Mistake", (230, 126, 34)) 
    BLUNDER = ("Blunder", (231, 76, 60))  
    BOOK = ("Book", (155, 89, 182))       


class OpeningNavigator:
    def __init__(self, tree):
        self.tree = tree
        self.reset()

    def reset(self):
        self.current = self.tree
        self.history = []
        self.active = True

    def apply(self, uci_move: str) -> bool:
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
        if not self.active or not isinstance(self.current, dict):
            return []
        return list(self.current.keys())

    def select_random(self) -> str | None:
        opts = self.get_options()
        return random.choice(opts) if opts else None


class PieceCache:
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
        pygame.draw.circle(s, border, (self.size // 2, self.size // 2), self.size // 2 - 5, 2)
        txt = font.render(name[0].upper(), True, fg)
        s.blit(txt, (self.size // 2 - txt.get_width() // 2, self.size // 2 - txt.get_height() // 2))
        return s


class UIComponent:
    def __init__(self, rect):
        self.rect = rect
    def draw(self, surf): pass
    def handle(self, event): return False


class Button(UIComponent):
    def __init__(self, rect, text, callback, color_theme, active=False, font_key="normal"):
        super().__init__(rect)
        self.text = text
        self.callback = callback
        self.theme = color_theme
        self.active = active
        self.hover = False
        self.font_key = font_key
        self.hover_t = 0.0

    def draw(self, surf, fonts):
        target = 1.0 if (self.active or self.hover) else 0.0
        self.hover_t += (target - self.hover_t) * 0.2
        col = tuple(int(self.theme["btn"][i] + (self.theme["accent_hover"][i] - self.theme["btn"][i]) * self.hover_t) for i in range(3))
        shadow = pygame.Rect(self.rect.x, self.rect.y + int(3 - 2 * self.hover_t), self.rect.w, self.rect.h)
        pygame.draw.rect(surf, (12, 14, 18), shadow, border_radius=8)
        pygame.draw.rect(surf, col, self.rect, border_radius=8)
        if self.hover or self.active:
            pygame.draw.rect(surf, self.theme["accent"], self.rect, 2, border_radius=8)
        t = fonts[self.font_key].render(self.text, True, self.theme["text"])
        surf.blit(t, (self.rect.centerx - t.get_width() // 2, self.rect.centery - t.get_height() // 2))

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.callback: self.callback()
                return True
        return False


class Slider(UIComponent):
    def __init__(self, x, y, w, h, min_v, max_v, init, label, theme):
        super().__init__(pygame.Rect(x, y, w, h))
        self.min, self.max, self.val = min_v, max_v, init
        self.label = label
        self.theme = theme
        self.drag = False
        self.radius = h // 2 + 4

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            hx = self.rect.x + ((self.val - self.min) / (self.max - self.min)) * self.rect.width
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

    def draw(self, surf, fonts):
        track = pygame.Rect(self.rect.x, self.rect.centery - 3, self.rect.width, 6)
        pygame.draw.rect(surf, (60, 60, 60), track, border_radius=3)
        fill_w = ((self.val - self.min) / (self.max - self.min)) * self.rect.width
        fill = pygame.Rect(self.rect.x, self.rect.centery - 3, fill_w, 6)
        pygame.draw.rect(surf, self.theme["accent"], fill, border_radius=3)
        hx = self.rect.x + fill_w
        pygame.draw.circle(surf, (40, 40, 40), (hx, self.rect.centery + 2), self.radius)
        pygame.draw.circle(surf, self.theme["accent"] if self.drag else (220, 220, 220), (hx, self.rect.centery), self.radius)
        lbl = fonts["small"].render(f"{self.label}: {self.val}", True, self.theme["text"])
        surf.blit(lbl, (self.rect.x, self.rect.y - 25))


class GameAnalysis:
    """Performs full game analysis"""
    def __init__(self, pgn_game, engine_depth, callback):
        self.game = pgn_game
        self.depth = engine_depth
        self.callback = callback
        self.stop_flag = False
        
    def run(self):
        board = self.game.board()
        analysis = []
        
        
        self._analyze_pos(board, 0, analysis)
        
        node = self.game
        ply = 0
        while node.next() and not self.stop_flag:
            node = node.next()
            ply += 1
            move = node.move
            
            
            board.push(move)
            self._analyze_pos(board, ply, analysis)
            
        self.callback(analysis)

    def _analyze_pos(self, chess_board, ply, analysis_list):
        
        fen = chess_board.fen()
        
        p, o, s, c, e, pl = parse_fen(bytes(fen, "utf-8"))
        b = Board(p, o, s, c, e, pl)
        
        searcher = SingleSearch(b, depth=self.depth)
        best_move = searcher.search()
        
        
        
        
        
        
        
        
        pass

class ChessGUI:
    DARK_THEME = {
        "light_sq": (228, 230, 236),
        "dark_sq": (118, 140, 176),
        "bg": (16, 18, 23),
        "panel": (26, 30, 38),
        "selected": (186, 202, 68, 160),
        "last_move": (255, 226, 130, 120),
        "hover": (120, 176, 255, 90),
        "mark": (255, 200, 80, 110),
        "arrow": (120, 168, 255, 200),
        "text": (240, 244, 250),
        "text_muted": (170, 178, 190),
        "btn": (36, 40, 50),
        "btn_hover": (52, 58, 72),
        "accent": (94, 132, 255),
        "accent_hover": (122, 160, 255),
        "overlay": (0, 0, 0, 180),
        "check": (200, 60, 60, 180),
        "book": (76, 175, 80),
        "engine": (255, 170, 64),
        "eval_border": (74, 86, 102),
        "eval_white": (245, 247, 250),
        "eval_black": (18, 20, 24),
    }
    LIGHT_THEME = {
        "light_sq": (248, 249, 252),
        "dark_sq": (176, 196, 220),
        "bg": (236, 238, 242),
        "panel": (250, 252, 255),
        "selected": (180, 205, 90, 150),
        "last_move": (255, 226, 150, 140),
        "hover": (100, 160, 240, 80),
        "mark": (255, 188, 80, 110),
        "arrow": (86, 130, 210, 190),
        "text": (26, 30, 36),
        "text_muted": (90, 96, 108),
        "btn": (228, 232, 238),
        "btn_hover": (210, 216, 226),
        "accent": (72, 110, 200),
        "accent_hover": (92, 130, 220),
        "overlay": (255, 255, 255, 180),
        "check": (220, 70, 70, 160),
        "book": (56, 145, 80),
        "engine": (210, 120, 0),
        "eval_border": (150, 160, 170),
        "eval_white": (250, 250, 250),
        "eval_black": (50, 54, 60),
    }
    THEMES = {"dark": DARK_THEME, "light": LIGHT_THEME}

    def __init__(self, w=1600, h=1000):
        pygame.init()
        self.disp = pygame.display.set_mode((w, h), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Chess Engine - Pro Interface")
        self.w, self.h = w, h
        self.clock = pygame.time.Clock()
        self.theme_name = "dark"
        self.C = self.THEMES[self.theme_name]

        self.fonts = {
            "header": pygame.font.SysFont("Segoe UI", 80, bold=True),
            "subhead": pygame.font.SysFont("Segoe UI", 50, bold=True),
            "title": pygame.font.SysFont("Segoe UI", 32, bold=True),
            "normal": pygame.font.SysFont("Segoe UI", 20),
            "small": pygame.font.SysFont("Segoe UI", 16),
            "tiny": pygame.font.SysFont("Segoe UI", 14),
            "coord": pygame.font.SysFont("Segoe UI", 14, bold=True),
            "mono": pygame.font.SysFont("Consolas", 14),
        }

        self._compute_layout()

        self.pieces = PieceCache(self.sqsize)
        self.openings = OpeningNavigator(OPENING_TREE)

        
        self.mode = GameMode.MENU
        self.board = None
        self.sel = None
        self.valids = []
        self.log = [] 
        self.states = []
        self.lastmv = None
        self.gstate = GameState.PLAYING
        self.winner = ""
        self.reason = ""
        self.flipped = False
        
        
        self.aiqueue = mp.Queue()
        self.infoqueue = mp.Queue()
        self.engine_cmd_q = mp.Queue()
        self.thinking = False
        self.ai_thread = None
        self.ai_process = None
        self.analysis_process = None
        self.analysis_thread = None
        self.analysis_stop = None
        self.analysis_info = {"depth": 0, "score": "0.00", "pv": "-", "pv_uci": "", "nps": 0}
        self.search_id = 0
        self.pending_search_id = None
        
        
        self.move_classifications = [] 
        self.analysis_scores = [] 
        self.pgn_headers = {}

        
        self.depth = 6
        self.pcolor = Color.WHITE
        self.ai_auto = False 
        self.ai_manual_trigger = False
        self.ai_delay = 0.5
        self.arrows = []
        self.arrow_start = None
        self.marked = {}
        self.hover_sq = None
        self.show_eval_bar = True
        self.show_debug = False
        self.color_picker_active = False
        self.color_buttons = []
        self.pending_pcolor = None
        self.panel_reserved_h = 140
        
        
        self.menu_buttons = []
        self.menu_slider = None
        self.panel_buttons = []
        self.analysis_buttons = []
        
        self._warmup()
        self._start_engine_process()
        self._init_menu()

    def _compute_layout(self):
        left = 48
        right = 40
        top = 60
        bottom = 60
        eval_w = 34
        gap_be = 16
        gap_ep = 18
        panel_min = 360
        max_board_w = self.w - left - right - eval_w - gap_be - gap_ep - panel_min
        max_board_h = self.h - top - bottom
        bsize = min(max_board_w, max_board_h)
        if bsize < 320:
            bsize = max(260, min(max_board_w, max_board_h))
        panel_w = self.w - left - right - eval_w - gap_be - gap_ep - bsize
        if panel_w < panel_min:
            bsize = max(260, self.w - left - right - eval_w - gap_be - gap_ep - panel_min)
            panel_w = self.w - left - right - eval_w - gap_be - gap_ep - bsize
        self.bsize = bsize
        self.sqsize = self.bsize // 8
        self.offx = left
        self.offy = (self.h - self.bsize) // 2
        self.eval_w = eval_w
        self.eval_x = self.offx + self.bsize + gap_be
        self.eval_y = self.offy + 12
        self.panel_x = self.eval_x + self.eval_w + gap_ep
        self.panel_w = panel_w
        self.panel_y = self.offy
        self.panel_h = self.bsize
    def _warmup(self):
        self.disp.fill(self.C["bg"])
        t = self.fonts["title"].render("Initializing Engine...", True, self.C["accent"])
        self.disp.blit(t, (self.w//2 - t.get_width()//2, self.h//2))
        pygame.display.flip()
        p, o, s, c, e, pl = parse_fen(STARTING_FEN)
        b = Board(p, o, s, c, e, pl)
        Move_generator(b)
        is_square_attacked(b, 0, 0)
        AI(b, s, 1)

    def _stop_ai_process(self):
        if self.ai_process is not None:
            try:
                self.engine_cmd_q.put(("stop",))
            except:
                pass
            self.ai_process.join(timeout=0.2)
            if self.ai_process.is_alive():
                self.ai_process.terminate()
                self.ai_process.join(timeout=0.2)
            self.ai_process = None
            self.pending_search_id = None

    def _start_engine_process(self):
        if self.ai_process is not None and self.ai_process.is_alive():
            return
        ctx = mp.get_context("spawn")
        self.ai_process = ctx.Process(
            target=engine_loop,
            args=(self.engine_cmd_q, self.infoqueue, self.aiqueue),
            daemon=True,
        )
        self.ai_process.start()

    def _stop_analysis_process(self):
        if self.analysis_process is not None:
            if self.analysis_process.is_alive():
                self.analysis_process.terminate()
            self.analysis_process.join(timeout=0.2)
            self.analysis_process = None

    def _init_menu(self):
        cx, cy = self.w // 2, self.h // 2
        bw, bh = 420, 76
        gap = 22
        slider_h = 30
        total_h = 4 * bh + 3 * gap + slider_h + 40
        start_y = max(140, (self.h - total_h) // 2 + 40)
        self.menu_buttons = [
            Button(pygame.Rect(cx - bw//2, start_y, bw, bh), "Player vs Player", lambda: self.set_mode(GameMode.PVP), self.C),
            Button(pygame.Rect(cx - bw//2, start_y + (bh + gap), bw, bh), "Player vs AI", self.open_color_picker, self.C),
            Button(pygame.Rect(cx - bw//2, start_y + 2*(bh + gap), bw, bh), "AI vs AI", lambda: self.set_mode(GameMode.AI_VS_AI), self.C),
            Button(pygame.Rect(cx - bw//2, start_y + 3*(bh + gap), bw, bh), "Analysis Board", lambda: self.set_mode(GameMode.ANALYSIS), self.C),
        ]
        slider_y = start_y + 4 * (bh + gap) + 20
        self.menu_slider = Slider(cx - bw//2, slider_y, bw, slider_h, 1, 12, self.depth, "Engine Depth", self.C)
        self.color_buttons = []
        self.color_picker_active = False

    def set_mode(self, mode):
        self.mode = mode
        self.depth = self.menu_slider.val
        if mode == GameMode.P_VS_AI and self.pending_pcolor is not None:
            self.pcolor = self.pending_pcolor
            self.pending_pcolor = None
        else:
            self.pcolor = Color.WHITE
        self.flipped = False
        self.init_game()

    def init_game(self):
        p, o, s, c, e, pl = parse_fen(STARTING_FEN)
        self.board = Board(p, o, s, c, e, pl)
        self.openings.reset()
        self.sel, self.valids = None, []
        self.log, self.states, self.lastmv = [], [], None
        self.thinking = False
        self.gstate = GameState.PLAYING
        self.analysis_info = {"depth": 0, "score": "0.00", "pv": "-", "pv_uci": "", "nps": 0}
        self.move_classifications = []
        self.pgn_headers = {"Event": "Engine Game", "Site": "Local", "Date": datetime.date.today().strftime("%Y.%m.%d")}
        self.ai_auto = False
        self.ai_manual_trigger = False
        self.arrows = []
        self.arrow_start = None
        self.marked = {}
        self.hover_sq = None
        self._stop_analysis_process()
        if self.analysis_stop: self.analysis_stop[0] = 1
        try:
            while True:
                self.aiqueue.get_nowait()
        except:
            pass
        try:
            while True:
                self.infoqueue.get_nowait()
        except:
            pass
        self.pending_search_id = None

        self._build_panel_buttons()

        if self.mode == GameMode.ANALYSIS:
            self.start_analysis()
        elif self.mode == GameMode.AI_VS_AI:
            self.start_ai()

    def toggle_auto(self):
        self.ai_auto = not self.ai_auto
        for btn in self.panel_buttons:
            if "Auto:" in btn.text:
                btn.text = f"Auto: {'On' if self.ai_auto else 'Off'}"
                btn.active = self.ai_auto
        if self.ai_auto:
            self.start_ai()

    def trigger_manual(self):
        self.ai_manual_trigger = True
        self.start_ai()
    
    def toggle_theme(self):
        self._apply_theme("light" if self.theme_name == "dark" else "dark")
    
    def toggle_eval_bar(self):
        self.show_eval_bar = not self.show_eval_bar
        for btn in self.panel_buttons:
            if btn.text.startswith("Eval:"):
                btn.text = f"Eval: {'On' if self.show_eval_bar else 'Off'}"
                btn.active = self.show_eval_bar
    
    def toggle_debug(self):
        self.show_debug = not self.show_debug
        for btn in self.panel_buttons:
            if btn.text.startswith("Debug:"):
                btn.text = f"Debug: {'On' if self.show_debug else 'Off'}"
                btn.active = self.show_debug
    
    def clear_marks(self):
        self.arrows = []
        self.marked = {}

    def open_color_picker(self):
        self.color_picker_active = True
        bw, bh = 320, 68
        gap = 20
        cx, cy = self.w // 2, self.h // 2
        total_h = 4 * bh + 3 * gap
        start_y = cy - total_h // 2
        self.color_buttons = [
            Button(pygame.Rect(cx - bw//2, start_y, bw, bh), "Play White", lambda: self._choose_color(Color.WHITE), self.C),
            Button(pygame.Rect(cx - bw//2, start_y + (bh + gap), bw, bh), "Play Black", lambda: self._choose_color(Color.BLACK), self.C),
            Button(pygame.Rect(cx - bw//2, start_y + 2*(bh + gap), bw, bh), "Random Color", self._choose_random_color, self.C),
            Button(pygame.Rect(cx - bw//2, start_y + 3*(bh + gap), bw, bh), "Back", self._close_color_picker, self.C),
        ]

    def _choose_random_color(self):
        self._choose_color(Color.WHITE if random.randint(0, 1) == 0 else Color.BLACK)

    def _choose_color(self, color):
        self.pending_pcolor = color
        self.color_picker_active = False
        self.color_buttons = []
        self.set_mode(GameMode.P_VS_AI)

    def _close_color_picker(self):
        self.color_picker_active = False
        self.color_buttons = []

    def load_pgn(self):
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("PGN Files", "*.pgn")])
        root.destroy()
        if not file_path: return
        
        try:
            with open(file_path) as f:
                game = chess.pgn.read_game(f)
            
            if not game: return
            
            
            self.init_game()
            self.pgn_headers = dict(game.headers)
            
            
            board = game.board()
            for move in game.mainline_moves():
                uci = move.uci()
                
                mvs = Move_generator(self.board)
                found = False
                for i in range(mvs.counter):
                    m = mvs.moves[i]
                    if move_to_uci(m) == uci:
                        self.push(m)
                        found = True
                        break
                if not found:
                    print(f"Error parsing PGN move: {uci}")
                    break
        except Exception as e:
            print(f"PGN Load Error: {e}")

    def save_pgn(self):
        game = chess.pgn.Game()
        game.headers.update(self.pgn_headers)
        game.headers["White"] = "Engine" if self.pcolor == Color.BLACK else "Player"
        game.headers["Black"] = "Engine" if self.pcolor == Color.WHITE else "Player"
        
        node = game
        for uci in self.log:
            move = chess.Move.from_uci(uci)
            node = node.add_variation(move)
            
        filename = f"game_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pgn"
        with open(filename, "w") as f:
            print(game, file=f, end="\n\n")
        print(f"Saved to {filename}")

    def run_full_analysis(self):
        
        
        
        threading.Thread(target=self._full_analysis_worker, daemon=True).start()

    def _full_analysis_worker(self):
        
        
        
        pass

    def info_callback(self, msg):
        self.infoqueue.put(msg)

    def start_ai(self):
        if self.thinking or self.gstate != GameState.PLAYING:
            return
        
        
        if self.mode == GameMode.AI_VS_AI and not self.ai_auto and not self.ai_manual_trigger:
            return
            
        self.thinking = True
        self.ai_manual_trigger = False
        book_move = None
        if self.openings.active:
            book_uci = self.openings.select_random()
            if book_uci:
                mvs = Move_generator(self.board)
                for i in range(mvs.counter):
                    m = mvs.moves[i]
                    if move_to_uci(m) == book_uci:
                        book_move = m
                        break
        self.search_id += 1
        self.pending_search_id = self.search_id
        if book_move:
            self.aiqueue.put(("book", book_move, self.pending_search_id))
            return
        state = board_to_state(self.board)
        delay = self.ai_delay if self.mode == GameMode.AI_VS_AI else 0.0
        self._start_engine_process()
        self.engine_cmd_q.put(("search", state, int(self.depth), delay, self.pending_search_id))

    def start_analysis(self):
        self._stop_analysis_process()
        state = board_to_state(self.board)
        ctx = mp.get_context("spawn")
        self.analysis_process = ctx.Process(
            target=analysis_worker,
            args=(state, self.infoqueue),
            daemon=True,
        )
        self.analysis_process.start()

    def parse_info(self):
        while True:
            try:
                msg = self.infoqueue.get_nowait()
            except:
                break
            if msg.startswith("info depth"):
                parts = msg.split()
                try:
                    d_idx = parts.index("depth")
                    s_idx = parts.index("score")
                    pv_idx = parts.index("pv")
                    self.analysis_info["depth"] = parts[d_idx + 1]
                    
                    score_type = parts[s_idx + 1]
                    val = int(parts[s_idx + 2])
                    if score_type == "cp":
                        
                        self.analysis_info["score"] = f"{val/100:.2f}"
                    else:
                        self.analysis_info["score"] = f"M{val}"
                        
                    pv_moves = parts[pv_idx + 1:pv_idx + 6]
                    self.analysis_info["pv"] = " ".join(pv_moves)
                    self.analysis_info["pv_uci"] = pv_moves[0] if pv_moves else ""
                except: pass
            elif msg.startswith("info stats"):
                parts = msg.split()
                try:
                    nps_idx = parts.index("nps")
                    self.analysis_info["nps"] = parts[nps_idx + 1]
                except: pass

    def push(self, mv):
        self.states.append({
            "board": self.board.copy(),
            "lastmv": self.lastmv,
            "log": self.log[:],
            "gstate": self.gstate,
            "opnav": self._copy_opnav(),
            "classifications": self.move_classifications[:]
        })
        
        
        
        if self.openings.active:
            self.move_classifications.append(MoveClass.BOOK)
        else:
            
            
            
            self.move_classifications.append(MoveClass.GOOD) 

        if Move(self.board, mv):
            uci = move_to_uci(mv)
            self.lastmv = (get_start_square(mv), get_target_square(mv))
            self.log.append(uci)
            self.openings.apply(uci)
            self.sel, self.valids = None, []
            self.check_end()
            
            if self.mode == GameMode.ANALYSIS:
                self.start_analysis()
            return True
        return False

    def undo(self):
        if not self.states: return
        st = self.states.pop()
        self.board, self.lastmv, self.log = st["board"], st["lastmv"], st["log"]
        self.gstate, self.openings = st["gstate"], st["opnav"]
        self.move_classifications = st["classifications"]
        self.sel, self.valids, self.thinking = None, [], False
        
        if self.mode == GameMode.ANALYSIS:
            self.start_analysis()
        elif self.mode == GameMode.P_VS_AI and self.board.side != self.pcolor:
             self.undo()

    def _copy_opnav(self):
        nav = OpeningNavigator(OPENING_TREE)
        nav.current, nav.history, nav.active = self.openings.current, self.openings.history[:], self.openings.active
        return nav

    def check_end(self):
        mvs = Move_generator(self.board)
        has_legal = False
        pc, pe, pp = self.board.castle, self.board.enpassant, self.board.halfmove
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
                self.winner = "Black Wins" if self.board.side == Color.WHITE else "White Wins"
                self.reason = "Checkmate"
            else:
                self.gstate = GameState.STALEMATE
                self.winner, self.reason = "Draw", "Stalemate"
        elif self.board.halfmove >= 100:
             self.gstate = GameState.DRAW
             self.winner, self.reason = "Draw", "50-Move Rule"
        
        if self.gstate != GameState.PLAYING and self.analysis_stop:
            self.analysis_stop[0] = 1

    
    def draw_board(self):
        frame = pygame.Rect(self.offx - 15, self.offy - 15, self.bsize + 30, self.bsize + 30)
        shadow = pygame.Surface((frame.w + 12, frame.h + 12), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 180), shadow.get_rect(), border_radius=16)
        self.disp.blit(shadow, (frame.x + 8, frame.y + 10))
        pygame.draw.rect(self.disp, self.C["panel"], frame, border_radius=16)
        pygame.draw.rect(self.disp, self.C["accent"], frame, 3, border_radius=16)
        for r in range(8):
            for f in range(8):
                ar, af = (7 - r, 7 - f) if self.flipped else (r, f)
                col = self.C["light_sq"] if (ar + af) % 2 == 0 else self.C["dark_sq"]
                x, y = self.offx + f * self.sqsize, self.offy + r * self.sqsize
                pygame.draw.rect(self.disp, col, (x, y, self.sqsize, self.sqsize))
                
                if f == 0:
                    ccol = self.C["dark_sq"] if col == self.C["light_sq"] else self.C["light_sq"]
                    t = self.fonts["coord"].render(str(8 - ar), True, ccol)
                    self.disp.blit(t, (x + 3, y + 3))
                if r == 7:
                    ccol = self.C["dark_sq"] if col == self.C["light_sq"] else self.C["light_sq"]
                    t = self.fonts["coord"].render(chr(97 + af), True, ccol)
                    self.disp.blit(t, (x + self.sqsize - 12, y + self.sqsize - 18))

        if self.lastmv:
             s, e = self.lastmv
             self._highlight_sq(s, self.C["last_move"])
             self._highlight_sq(e, self.C["last_move"])
        if self.sel is not None:
             self._highlight_sq(self.sel, self.C["selected"])
        if self.marked:
            for sq in self.marked:
                self._highlight_sq(sq, self.C["mark"])
        if self.hover_sq is not None:
            self._highlight_sq(self.hover_sq, self.C["hover"])
        
        kp = Pieces.K.value if self.board.side == Color.WHITE else Pieces.k.value
        ksq = get_lsb1_index(self.board.bitboard[kp])
        att = Color.BLACK if self.board.side == Color.WHITE else Color.WHITE
        if is_square_attacked(self.board, ksq, att):
             self._highlight_sq(ksq, self.C["check"])

    def _highlight_sq(self, sq, col):
        r, f = divmod(sq, 8)
        if self.flipped: r, f = 7 - r, 7 - f
        x, y = self.offx + f * self.sqsize, self.offy + r * self.sqsize
        s = pygame.Surface((self.sqsize, self.sqsize), pygame.SRCALPHA)
        s.fill(col)
        self.disp.blit(s, (x, y))

    def draw_pieces(self):
        for i in range(12):
            bb = self.board.bitboard[i]
            while bb:
                sq = get_lsb1_index(bb)
                r, f = divmod(sq, 8)
                if self.flipped: r, f = 7 - r, 7 - f
                x = self.offx + f * self.sqsize + self.sqsize // 2
                y = self.offy + r * self.sqsize + self.sqsize // 2
                img = self.pieces.get(i)
                self.disp.blit(img, img.get_rect(center=(x, y)))
                bb &= bb - 1
    
    def draw_arrows(self):
        for s, e, col in self.arrows:
            self._draw_arrow(s, e, col)
        pv = self.analysis_info.get("pv_uci", "")
        if pv and self.show_eval_bar:
            start = self._coord_to_sq(pv[:2])
            end = self._coord_to_sq(pv[2:4])
            if start is not None and end is not None:
                self._draw_arrow(start, end, self.C["arrow"])
    
    def _draw_arrow(self, s_sq, e_sq, col):
        if s_sq == e_sq:
            return
        sx, sy = self._sq_center(s_sq)
        ex, ey = self._sq_center(e_sq)
        dx, dy = ex - sx, ey - sy
        dist = max(1, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / dist, dy / dist
        head_len = self.sqsize * 0.35
        head_w = self.sqsize * 0.18
        bx, by = ex - ux * head_len, ey - uy * head_len
        px, py = -uy * head_w, ux * head_w
        arrow_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pygame.draw.line(arrow_surf, col, (sx, sy), (bx, by), 6)
        pygame.draw.polygon(arrow_surf, col, [(ex, ey), (bx + px, by + py), (bx - px, by - py)])
        self.disp.blit(arrow_surf, (0, 0))
    
    def _sq_center(self, sq):
        r, f = divmod(sq, 8)
        if self.flipped: r, f = 7 - r, 7 - f
        x = self.offx + f * self.sqsize + self.sqsize // 2
        y = self.offy + r * self.sqsize + self.sqsize // 2
        return x, y
    
    def _coord_to_sq(self, coord):
        if len(coord) < 2:
            return None
        f = ord(coord[0]) - 97
        r = int(coord[1]) - 1
        if f < 0 or f > 7 or r < 0 or r > 7:
            return None
        return (7 - r) * 8 + f

    def draw_moves(self):
        for mv in self.valids:
            tgt = get_target_square(mv)
            r, f = divmod(tgt, 8)
            if self.flipped: r, f = 7 - r, 7 - f
            is_cap = any((self.board.bitboard[i] >> tgt) & 1 for i in range(12))
            s = pygame.Surface((self.sqsize, self.sqsize), pygame.SRCALPHA)
            if is_cap:
                pygame.draw.circle(s, (0, 0, 0, 60), (self.sqsize//2, self.sqsize//2), self.sqsize//2 - 2, 6)
            else:
                pygame.draw.circle(s, (0, 0, 0, 40), (self.sqsize//2, self.sqsize//2), self.sqsize//6)
            self.disp.blit(s, (self.offx + f * self.sqsize, self.offy + r * self.sqsize))

    def draw_panel(self):
        px = self.panel_x
        w = self.panel_w
        y = self.panel_y
        panel_rect = pygame.Rect(px - 16, self.offy - 16, w + 32, self.bsize + 32)
        panel_shadow = pygame.Surface((panel_rect.w + 10, panel_rect.h + 10), pygame.SRCALPHA)
        pygame.draw.rect(panel_shadow, (0, 0, 0, 120), panel_shadow.get_rect(), border_radius=18)
        self.disp.blit(panel_shadow, (panel_rect.x + 5, panel_rect.y + 7))
        pygame.draw.rect(self.disp, self.C["panel"], panel_rect, border_radius=18)
        pygame.draw.rect(self.disp, self.C["accent"], panel_rect, 2, border_radius=18)

        status = "White to Move" if self.board.side == Color.WHITE else "Black to Move"
        if self.gstate != GameState.PLAYING:
            status = self.winner
            col = self.C["accent"]
        elif self.thinking:
            status = "Engine Thinking..."
            col = self.C["engine"]
        else:
            col = self.C["text"]
        status_bg = pygame.Rect(px - 8, y - 6, w + 16, 38)
        status_bg_col = (min(255, self.C["panel"][0] + 10), min(255, self.C["panel"][1] + 10), min(255, self.C["panel"][2] + 10))
        pygame.draw.rect(self.disp, status_bg_col, status_bg, border_radius=8)
        t = self.fonts["title"].render(status, True, col)
        self.disp.blit(t, (px, y))
        y += 50
        
        mode_str = str(self.mode).replace("GameMode.", "").replace("_", " ")
        t = self.fonts["normal"].render(f"{mode_str} | Depth: {self.depth}", True, self.C["text_muted"])
        self.disp.blit(t, (px, y))
        y += 42
        
        if self.mode == GameMode.ANALYSIS or self.mode == GameMode.AI_VS_AI or self.thinking:
            self._draw_analysis_box(px, y, w, 130)
            y += 150
        if self.show_debug:
            self._draw_debug_box(px, y, w, 110)
            y += 130
        panel_bottom = self.offy + self.bsize
        history_h = panel_bottom - y - self.panel_reserved_h
        if history_h < 90:
            history_h = max(0, panel_bottom - y - 8)
        if history_h > 0:
            self._draw_history(px, y, w, history_h)
        
        for btn in self.panel_buttons:
            btn.draw(self.disp, self.fonts)

    def _build_panel_buttons(self):
        self.panel_buttons = []
        px = self.panel_x
        panel_w = self.panel_w
        gap, bh = 16, 50
        items = [
            ("Menu", lambda: self.set_mode(GameMode.MENU), False),
            ("Flip", lambda: setattr(self, "flipped", not self.flipped), False),
            ("Undo", self.undo, False),
            ("Save PGN", self.save_pgn, False),
            (f"Eval: {'On' if self.show_eval_bar else 'Off'}", self.toggle_eval_bar, self.show_eval_bar),
            (f"Debug: {'On' if self.show_debug else 'Off'}", self.toggle_debug, self.show_debug),
        ]
        min_bw = 140
        max_cols = min(3, len(items))
        cols = max(2, min(max_cols, int((panel_w + gap) // (min_bw + gap))))
        bw = int((panel_w - gap * (cols - 1)) / cols)
        if bw < min_bw:
            cols = max(1, min(max_cols, int(panel_w // min_bw)))
            bw = int((panel_w - gap * (cols - 1)) / cols)
        rows = (len(items) + cols - 1) // cols
        extra_rows = 1 if self.mode in (GameMode.ANALYSIS, GameMode.AI_VS_AI) else 0
        buttons_h = rows * bh + (rows - 1) * gap + (bh + gap if extra_rows else 0)
        panel_bottom = self.offy + self.bsize
        base_y = panel_bottom - buttons_h
        base_y = max(self.offy + 20, base_y)
        self.panel_reserved_h = panel_bottom - base_y + 18
        for i, (text, cb, active) in enumerate(items):
            r = i // cols
            c = i % cols
            x = px + c * (bw + gap)
            y = base_y + r * (bh + gap)
            self.panel_buttons.append(Button(pygame.Rect(x, y, bw, bh), text, cb, self.C, active=active))
        if self.mode == GameMode.ANALYSIS:
            y = max(self.offy + 20, base_y - (bh + gap))
            self.panel_buttons.append(Button(pygame.Rect(px, y, bw, bh), "Load PGN", self.load_pgn, self.C))
            self.panel_buttons.append(Button(pygame.Rect(px + bw + gap, y, bw, bh), "Analyze Game", self.run_full_analysis, self.C))
        if self.mode == GameMode.AI_VS_AI:
            y = max(self.offy + 20, base_y - (bh + gap))
            self.panel_buttons.append(Button(pygame.Rect(px, y, bw, bh), f"Auto: {'On' if self.ai_auto else 'Off'}", self.toggle_auto, self.C, font_key="small"))
            self.panel_buttons.append(Button(pygame.Rect(px + bw + gap, y, bw, bh), "Next Move", self.trigger_manual, self.C, font_key="small"))

    def _draw_analysis_box(self, x, y, w, h):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.disp, self.C["panel"], rect, border_radius=16)
        pygame.draw.rect(self.disp, (70, 74, 86), rect, 2, border_radius=16)
        info = self.analysis_info
        lines = [
            f"Evaluation: {info['score']}",
            f"Search Depth: {info['depth']}  |  Nodes/sec: {info['nps']}",
            f"Principal Variation: {info['pv']}..."
        ]
        iy = y + 18
        for line in lines:
            t = self.fonts["mono"].render(line, True, self.C["accent"])
            self.disp.blit(t, (x + 18, iy))
            iy += 26

    def _draw_color_picker(self):
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill(self.C["overlay"])
        self.disp.blit(overlay, (0, 0))
        box_w, box_h = 420, 360
        rect = pygame.Rect(self.w//2 - box_w//2, self.h//2 - box_h//2, box_w, box_h)
        shadow = pygame.Rect(rect.x + 6, rect.y + 8, rect.w, rect.h)
        pygame.draw.rect(self.disp, (0, 0, 0, 120), shadow, border_radius=20)
        pygame.draw.rect(self.disp, self.C["panel"], rect, border_radius=20)
        pygame.draw.rect(self.disp, self.C["accent"], rect, 3, border_radius=20)
        glow = pygame.Rect(rect.x + 2, rect.y + 2, rect.w - 4, rect.h - 4)
        pygame.draw.rect(self.disp, (255, 255, 255, 20), glow, 1, border_radius=18)
        t = self.fonts["title"].render("Choose Your Color", True, self.C["text"])
        self.disp.blit(t, (rect.centerx - t.get_width()//2, rect.y + 30))
        subtitle = self.fonts["small"].render("Select your preferred side to play", True, self.C["text_muted"])
        self.disp.blit(subtitle, (rect.centerx - subtitle.get_width()//2, rect.y + 60))
        
        for btn in self.color_buttons:
            btn.draw(self.disp, self.fonts)
    
    def _draw_debug_box(self, x, y, w, h):
        rect = pygame.Rect(x, y, w, h)
        shadow = pygame.Rect(rect.x + 4, rect.y + 4, rect.w, rect.h)
        pygame.draw.rect(self.disp, (0, 0, 0, 100), shadow, border_radius=16)
        pygame.draw.rect(self.disp, self.C["panel"], rect, border_radius=16)
        pygame.draw.rect(self.disp, self.C["accent"], rect, 2, border_radius=16)
        side = "W" if self.board.side == Color.WHITE else "B"
        last = self.log[-1] if self.log else "-"
        info = [
            f"Side: {side} | Halfmove: {self.board.halfmove} | EP: {int(self.board.enpassant)}",
            f"Castle: {int(self.board.castle)} | Hash: {int(self.board.hash)}",
            f"Last: {last} | Sel: {self.sel if self.sel is not None else '-'} | Hover: {self.hover_sq if self.hover_sq is not None else '-'}",
            f"Marks: {len(self.marked)} | Arrows: {len(self.arrows)}",
        ]
        iy = y + 18
        for line in info:
            t = self.fonts["small"].render(line, True, self.C["text"])
            self.disp.blit(t, (x + 18, iy))
            iy += 24
    
    def _eval_to_float(self):
        s = self.analysis_info.get("score", "0.00")
        if isinstance(s, str) and s.startswith("M"):
            try:
                val = int(s[1:])
                return 100.0 if val > 0 else -100.0
            except:
                return 0.0
        try:
            return float(s)
        except:
            return 0.0
    
    def draw_eval_bar(self):
        if not self.show_eval_bar:
            return
        x = self.eval_x
        y = self.eval_y
        w = self.eval_w
        h = self.bsize - 24
        score = self._eval_to_float()
        max_eval = 10.0
        if score > max_eval:
            score = max_eval
        if score < -max_eval:
            score = -max_eval
        ratio = (score + max_eval) / (2 * max_eval)
        white_h = int(h * ratio)
        bar_rect = pygame.Rect(x, y, w, h)
        shadow = pygame.Rect(x + 2, y + 2, w, h)
        pygame.draw.rect(self.disp, (0, 0, 0, 80), shadow, border_radius=10)
        black_rect = pygame.Rect(x, y + white_h, w, h - white_h)
        pygame.draw.rect(self.disp, self.C["eval_black"], black_rect, border_radius=10)
        white_rect = pygame.Rect(x, y, w, white_h)
        pygame.draw.rect(self.disp, self.C["eval_white"], white_rect, border_radius=10)
        pygame.draw.rect(self.disp, self.C["eval_border"], bar_rect, 3, border_radius=10)
        inner_glow = pygame.Rect(x + 1, y + 1, w - 2, h - 2)
        pygame.draw.rect(self.disp, (255, 255, 255, 25), inner_glow, 1, border_radius=9)
        score_text = f"{score:+.2f}"
        t = self.fonts["small"].render(score_text, True, self.C["text"])
        score_y = y - 25 if score > 0 else y + h + 5
        self.disp.blit(t, (x + w//2 - t.get_width()//2, score_y))
        label_w = self.fonts["tiny"].render("+10", True, self.C["text_muted"])
        label_b = self.fonts["tiny"].render("-10", True, self.C["text_muted"])
        self.disp.blit(label_w, (x + w + 5, y - 2))
        self.disp.blit(label_b, (x + w + 5, y + h - label_b.get_height() + 2))

    def _draw_history(self, x, y, w, h):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.disp, self.C["panel"], rect, border_radius=16)
        pygame.draw.rect(self.disp, (60, 66, 78), rect, 2, border_radius=16)
        header_text = self.fonts["normal"].render("Move History", True, self.C["text"])
        self.disp.blit(header_text, (x + 20, y + 15))
        separator_y = y + 50
        pygame.draw.line(self.disp, self.C["accent"], (x + 15, separator_y), (x + w - 15, separator_y), 2)
        line_height = 28
        header_height = 60
        available_height = h - header_height - 15
        max_lines = available_height // line_height
        rows = []
        for i in range(0, len(self.log), 2):
            move_num = i // 2 + 1
            w_move = self.log[i]
            b_move = self.log[i+1] if i+1 < len(self.log) else None
            
            w_cls = self.move_classifications[i] if i < len(self.move_classifications) else None
            b_cls = self.move_classifications[i+1] if (i+1) < len(self.move_classifications) else None
            
            rows.append((move_num, w_move, b_move, w_cls, b_cls))
        start_row = max(0, len(rows) - max_lines)
        visible_rows = rows[start_row:]

        hy = y + header_height + 5
        for (num, wm, bm, wc, bc) in visible_rows:
            t_num = self.fonts["small"].render(f"{num}.", True, self.C["text_muted"])
            self.disp.blit(t_num, (x + 20, hy))
            col_w = self.C["text"]
            t_w = self.fonts["small"].render(wm, True, col_w)
            self.disp.blit(t_w, (x + 60, hy))
            if wc:
                pygame.draw.circle(self.disp, wc.value[1], (x + 110, hy + 10), 5)
            if bm:
                col_b = self.C["text"]
                t_b = self.fonts["small"].render(bm, True, col_b)
                self.disp.blit(t_b, (x + 140, hy))
                if bc:
                    pygame.draw.circle(self.disp, bc.value[1], (x + 190, hy + 10), 5)
            
            hy += line_height

    def draw_menu(self):
        self.disp.fill(self.C["bg"])
        for i in range(0, self.w, 100):
            for j in range(0, self.h, 100):
                pygame.draw.rect(self.disp, (self.C["bg"][0] + 3, self.C["bg"][1] + 3, self.C["bg"][2] + 3), 
                               pygame.Rect(i, j, 2, 2), border_radius=1)
        title_y = 60
        if self.menu_buttons:
            title_y = max(32, self.menu_buttons[0].rect.y - 130)
        title_text = self.fonts["header"].render("CHESS ENGINE", True, self.C["accent"])
        self.disp.blit(title_text, (self.w//2 - title_text.get_width()//2, title_y))
        
        if self.menu_buttons and self.menu_slider:
            min_x = min(btn.rect.x for btn in self.menu_buttons)
            max_x = max(btn.rect.right for btn in self.menu_buttons)
            top_y = self.menu_buttons[0].rect.y - 40
            bottom_y = self.menu_slider.rect.bottom + 50
            card = pygame.Rect(min_x - 45, top_y, (max_x - min_x) + 90, bottom_y - top_y)
            card_shadow = pygame.Surface((card.w + 16, card.h + 16), pygame.SRCALPHA)
            pygame.draw.rect(card_shadow, (0, 0, 0, 140), card_shadow.get_rect(), border_radius=22)
            self.disp.blit(card_shadow, (card.x + 8, card.y + 10))
            pygame.draw.rect(self.disp, self.C["panel"], card, border_radius=22)
            pygame.draw.rect(self.disp, self.C["accent"], card, 2, border_radius=22)
            inner_glow = pygame.Surface((card.w - 8, card.h - 8), pygame.SRCALPHA)
            pygame.draw.rect(inner_glow, (255, 255, 255, 15), inner_glow.get_rect(), border_radius=18)
            self.disp.blit(inner_glow, (card.x + 4, card.y + 4))
        
        for btn in self.menu_buttons: btn.draw(self.disp, self.fonts)
        self.menu_slider.draw(self.disp, self.fonts)
        if self.color_picker_active:
            self._draw_color_picker()

    def handle_click(self, pos):
        if self.thinking and self.mode != GameMode.ANALYSIS: return
        sq = self._px2sq(pos)
        if sq is None: return
        
        if self.sel is None:
            is_white = self.board.side == Color.WHITE
            start, end = (0, 6) if self.board.side == Color.WHITE else (6, 12)
            has_piece = any((self.board.bitboard[i] >> sq) & 1 for i in range(start, end))
            if has_piece:
                self.sel = sq
                self.valids = self._get_legal_moves(sq)
        else:
            if sq == self.sel:
                self.sel, self.valids = None, []
            else:
                chosen = None
                for m in self.valids:
                    if get_target_square(m) == sq:
                        chosen = m
                        if get_flag(m) in [Flag.QUEEN_PROMOTION, Flag.CAPTURE_PROMOTION_QUEEN]: break
                if chosen: self.push(chosen)
                else:
                    self.sel = None
                    self.handle_click(pos)

    def _px2sq(self, pos):
        x, y = pos[0] - self.offx, pos[1] - self.offy
        if 0 <= x < self.bsize and 0 <= y < self.bsize:
            c, r = x // self.sqsize, y // self.sqsize
            if self.flipped: r, c = 7 - r, 7 - c
            return r * 8 + c
        return None

    def _get_legal_moves(self, sq):
        mvs = Move_generator(self.board)
        pc, pe, pp = self.board.castle, self.board.enpassant, self.board.halfmove
        legals = []
        for i in range(mvs.counter):
            m = mvs.moves[i]
            if get_start_square(m) == sq:
                if Move(self.board, m):
                    unmove(self.board, m, pc, pe, pp)
                    legals.append(m)
        return legals
    
    def _apply_theme(self, name):
        self.theme_name = name
        self.C = self.THEMES[name]
        for btn in self.menu_buttons:
            btn.theme = self.C
        for btn in self.panel_buttons:
            btn.theme = self.C
        for btn in self.analysis_buttons:
            btn.theme = self.C
        for btn in self.color_buttons:
            btn.theme = self.C
        if self.menu_slider:
            self.menu_slider.theme = self.C
    
    def _toggle_mark(self, sq):
        if sq in self.marked:
            del self.marked[sq]
        else:
            self.marked[sq] = True
    
    def _toggle_arrow(self, s, e):
        for i, (a, b, _) in enumerate(self.arrows):
            if a == s and b == e:
                self.arrows.pop(i)
                return
        self.arrows.append((s, e, self.C["arrow"]))
    
    def handle_shortcuts(self, ev):
        if ev.key == pygame.K_m:
            self.set_mode(GameMode.MENU)
        elif ev.key == pygame.K_1:
            self.set_mode(GameMode.PVP)
        elif ev.key == pygame.K_2:
            if self.mode == GameMode.MENU:
                self.open_color_picker()
            else:
                self.set_mode(GameMode.P_VS_AI)
        elif ev.key == pygame.K_3:
            self.set_mode(GameMode.AI_VS_AI)
        elif ev.key == pygame.K_4:
            self.set_mode(GameMode.ANALYSIS)
        elif ev.key == pygame.K_f:
            self.flipped = not self.flipped
        elif ev.key == pygame.K_u:
            self.undo()
        elif ev.key == pygame.K_s:
            if self.mode != GameMode.MENU:
                self.save_pgn()
        elif ev.key == pygame.K_l:
            if self.mode == GameMode.ANALYSIS:
                self.load_pgn()
        elif ev.key == pygame.K_a:
            if self.mode == GameMode.AI_VS_AI:
                self.toggle_auto()
        elif ev.key == pygame.K_n:
            if self.mode == GameMode.AI_VS_AI:
                self.trigger_manual()
        elif ev.key == pygame.K_t:
            self.toggle_theme()
        elif ev.key == pygame.K_e:
            self.toggle_eval_bar()
        elif ev.key == pygame.K_d:
            self.toggle_debug()
        elif ev.key == pygame.K_c:
            self.clear_marks()

    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: running = False
                if ev.type == pygame.KEYDOWN:
                    self.handle_shortcuts(ev)
                
                if self.mode == GameMode.MENU:
                    handled = False
                    if self.color_picker_active:
                        for btn in self.color_buttons:
                            if btn.handle(ev): handled = True
                    else:
                        for btn in self.menu_buttons:
                            if btn.handle(ev): handled = True
                        if not handled: self.menu_slider.handle(ev)
                else:
                    handled = False
                    for btn in self.panel_buttons:
                        if btn.handle(ev): handled = True
                    if ev.type == pygame.MOUSEMOTION:
                        self.hover_sq = self._px2sq(ev.pos)
                    if not handled and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        self.handle_click(ev.pos)
                    if not handled and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
                        self.arrow_start = self._px2sq(ev.pos)
                    if not handled and ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                        end = self._px2sq(ev.pos)
                        if self.arrow_start is not None and end is not None:
                            if end == self.arrow_start:
                                self._toggle_mark(end)
                            else:
                                self._toggle_arrow(self.arrow_start, end)
                        self.arrow_start = None
            
            self.parse_info()
            if self.mode != GameMode.MENU:
                try:
                    kind, move, sid = self.aiqueue.get_nowait()
                    if sid == self.pending_search_id:
                        self.thinking = False
                        self.pending_search_id = None
                        if move is not None: self.push(move)
                except queue.Empty: pass
                except Exception: pass

                if self.gstate == GameState.PLAYING and not self.thinking:
                    if self.mode == GameMode.AI_VS_AI and (self.ai_auto or self.ai_manual_trigger):
                        self.start_ai()
                    elif self.mode == GameMode.P_VS_AI:
                        if self.board.side != self.pcolor: self.start_ai()

            if self.mode == GameMode.MENU: self.draw_menu()
            else:
                self.disp.fill(self.C["bg"])
                self.draw_board()
                self.draw_arrows()
                self.draw_eval_bar()
                self.draw_moves()
                self.draw_pieces()
                self.draw_panel()
            pygame.display.flip()
            self.clock.tick(60)

        self._stop_ai_process()
        self._stop_analysis_process()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    if not os.path.exists("images"): os.makedirs("images")
    gui = ChessGUI()
    gui.run()
