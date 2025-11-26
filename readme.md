# Chess Engine

A high-performance chess engine based on bitboards, Numba JIT compilation, and a Pygame graphical interface. The standard play mode includes player vs player and player vs AI. It also remembers the history of moves, highlights possible legal moves, and provides smooth interactions.

## Project Overview

- Utilizes bitboards for fast move representation.
- Numba JIT compilation significantly speeds up move generation.
- GUI is built with Pygame with visual indicators for normal moves, captures, and last move highlighting.
- Uses magic bitboards for sliding piece attacks.
- Supports legal move checks, checkmate, stalemate, and undo.
- Includes threading for the AI to avoid GUI freezing.
- max NPS reached is 7 million (till now didnt check far higher Depths)

## File Structure Explanation

### Board.py
Handles bitboard representation, FEN parsing, occupancy arrays, and attack detection.

### Board_Move_gen.py
Generates pseudo-legal moves, executes moves, undoes moves, and converts moves to UCI format.

### Move_gen_pieces.py
Contains pre-generated attack tables for pawns, knights, bishops, rooks, queens, and kings.

### Constants.py
Defines enums for squares, pieces, colors, flags, and castling rights.

### GUI.py
Main Pygame interface implementing rendering, event handling, and AI threading.

### pregen/
Scripts for generating magic bitboards and attack lookup tables.

## Key Features

- Bitboard-based piece storage (64-bit integers).
- Magic bitboard sliding attacks for bishops and rooks.
- Numba @njit acceleration for move generation.
- 24-bit move encoding format.
- Pre-computed attack tables for speed.
- FEN parsing support.
- GUI with click-to-select, move highlighting, undo, and main menu.

## Running Requirements

Install required packages:
```bash
pip install pygame numpy numba
```

Requires pre-generated tables such as:
- PAWN_ATTACKS.npy
- KNIGHT_ATTACKS.npy
- KING_ATTACKS.npy
- Magic bitboard tables for bishops and rooks

## Running the Game

Run:
```bash
python GUI.py
```

## UI/UX Behavior

*Main Menu:*
- Choose Player vs Player or Player vs AI.

*In Game:*
- Clicking a piece shows all legal moves.
- Small dots indicate normal moves.
- Circles indicate capture moves.
- Last move is highlighted.
- Undo available using the U key or on-screen button.
- Main menu button resets the game.

*Game Over:*
- Detects checkmate, stalemate, and draw conditions.

## AI Information

The current AI picks a random legal move.  
A stronger AI can be integrated using minimax, alpha-beta pruning, heuristics, and move ordering.

## Performance Details

- Numba accelerates move generation.
- Rendering uses caching for better performance.
- Threaded AI prevents GUI freezing.
- Typical performance: fast move generation, 60 FPS rendering.

## Troubleshooting

*First Move Freeze:*
- Happens due to Numba JIT warmup during the first compilation.

*Missing Images:*
- Fallback piece rendering is used if images are missing.

*NumPy/Numba Errors:*
- Use compatible versions.

## Future Improvements (Planned)

- Opening book
- Endgame tablebases
- ELO rating
- PGN loading and saving
- Online multiplayer
- Mobile version
- Search depth/time controls
- Move annotations and analysis board

## License

MIT License with full permissions for modification, distribution, and commercial use. No warranty is provided.
