# SAT Chess Engine

A high-performance chess engine featuring bitboard representation, Numba JIT compilation, advanced evaluation, and a Pygame graphical interface. Combines sophisticated search algorithms with an intuitive user experience.

## Project Overview

SAT is a competitive chess engine built from the ground up with modern optimization techniques:

- **Bitboard Architecture**: Ultra-fast 64-bit integer representation for all pieces and board states
- **Numba JIT Compilation**: Near-C performance for critical move generation and evaluation paths
- **Advanced Search**: Negamax with alpha-beta pruning, aspiration windows, and iterative deepening
- **Sophisticated Evaluation**: Multi-phase evaluation with piece-square tables, king safety, pawn structure, mobility, and tactical awareness
- **Magic Bitboards**: Blazing-fast sliding piece attack generation using pre-computed magic numbers
- **Transposition Tables**: 64M entry hash table with Zobrist hashing for position caching
- **GUI Excellence**: Smooth Pygame interface with visual move indicators and game history
- **Peak Performance**: 1+ million nodes per second achieved on standard hardware

## File Structure

### Core Engine Files

**Board.py**
- Bitboard representation and manipulation
- FEN parsing with optimized starting position handling
- Occupancy arrays for efficient move validation
- Square attack detection for check and legal move verification
- Zobrist hash computation for transposition tables

**Board_Move_gen.py**
- Pseudo-legal move generation for all piece types
- Legal move validation with check detection
- Move execution and undo with full state preservation
- UCI move notation support
- Perft testing framework for move generation verification
- Capture-only move generation for quiescence search

**Move_gen_pieces.py**
- Pre-computed attack tables loaded from .npy files
- Magic bitboard attack generation for bishops and rooks
- Pawn, knight, and king attack lookups
- Queen attacks via bishop + rook combination

**Constants.py**
- Square, piece, color, and flag enumerations
- Castling rights bit masks
- Move encoding constants
- Coordinate mappings

**Evaluation.py**
- **Multi-phase evaluation**: Smooth interpolation between midgame and endgame
- **Piece-Square Tables**: Position-dependent piece values for all game phases
- **Mobility**: Safe and unsafe square counting with enemy attack consideration
- **King Safety**: Pawn shield evaluation, attacker counting, and king zone pressure
- **Pawn Structure**: Passed pawns, candidate pawns, and pawn advancement bonuses
- **Piece Safety**: Attack/defense analysis with tactical vulnerability detection
- **Endgame Expertise**: Specialized evaluation for basic endgames with forced win detection
- **Space Evaluation**: Territory control and piece cramping assessment
- **Tempo Bonus**: Side-to-move advantage

**Minimax.py** (Search Engine)
- **Negamax Framework**: Efficient minimax implementation with alpha-beta pruning
- **Iterative Deepening**: Progressive depth search with best move refinement
- **Aspiration Windows**: Narrow search windows for faster deep searches
- **Transposition Table**: 64M entry table with depth-preferred replacement
- **Quiescence Search**: Tactical move continuation to avoid horizon effect
- **Move Ordering**: 
  - TT move first
  - Promotions
  - MVV-LVA captures with SEE verification
  - Killer moves (2 per ply)
- **Pruning Techniques**:
  - Null move pruning with verification
  - Futility pruning
  - Late move reductions (LMR)
  - Late move pruning (LMP)
  - Delta pruning in quiescence
- **Special Endgame Handling**: Enhanced king pressure heuristics for basic endgames
- **Repetition Detection**: Draw by repetition with game history tracking
- **Mate Distance Pruning**: Efficient mate detection
- **Time Management**: Adaptive time allocation with increment support

**PST.py**
- Separate midgame and endgame piece-square tables
- All piece types with unique positional valuations
- Mirrored tables for Black pieces
- Fast numba-compiled table lookup

**Zobrist.py**
- Zobrist key generation for hashing
- Separate keys for pieces, castling rights, en passant, and side to move
- Reproducible random number generation with fixed seed

### Utility Files

**pregen/**
- Magic bitboard generation scripts
- Attack table pre-computation
- Utility functions for bit manipulation

**GUI.py**
- Pygame-based graphical interface
- Click-to-select piece interaction
- Move highlighting (normal moves, captures, last move)
- Threaded AI to prevent interface freezing
- Game state management
- Undo functionality
- Main menu with mode selection

## Key Technical Features

### Bitboard Representation
- Each piece type stored as a 64-bit unsigned integer
- Efficient set operations (AND, OR, XOR) for move generation
- Parallel attack map computation

### Move Encoding (32-bit format)
```
Bits 0-5:   Starting square (0-63)
Bits 6-11:  Target square (0-63)
Bits 12-15: Moving piece (0-11)
Bits 16-19: Captured piece (0-11, 12=none)
Bits 20-23: Move flag (promotion, castle, en passant, etc.)
```

### Magic Bitboards
- Pre-computed magic numbers for bishops and rooks
- O(1) sliding piece attack lookup
- Minimal perfect hashing of occupancy patterns

### Search Statistics
- Nodes searched
- Transposition table hits and cutoffs
- Evaluation calls
- Nodes per second (NPS)
- Depth-selective statistics

## Installation

### Requirements
```bash
pip install pygame numpy numba msgpack
```

### Setup
```bash
python pawn_att.py
python knight_att.py
python bishop_att.py
python rook_att.py
python king_att.py
python Random_number_gen.py
python slider_piece_generation.py
```

### Required Data Files
Ensure these pre-generated (see [Setup](#setup) for details) .npy files are in the project root:
- PAWN_ATTACKS.npy
- KNIGHT_ATTACKS.npy
- KING_ATTACKS.npy
- BISHOP_ATTACKS.npy
- BISHOP_MASK.npy
- BISHOP_SHIFTS.npy
- BISHOP_MAGICS.npy
- ROOK_ATTACKS.npy
- ROOK_MASK.npy
- ROOK_SHIFTS.npy
- ROOK_MAGICS.npy

## Running the Engine

### Graphical Interface
```bash
python GUI.py
```

## User Interface

### Main Menu
- **Player vs Player**: Two human players take turns
- **Player vs AI**: Play against the engine (configurable depth)

### In-Game Controls
- **Mouse Click**: Select piece and move destination
- **Visual Indicators**:
  - Small dots: Normal legal moves
  - Circles: Capture moves
  - Highlighted squares: Last move made
- **Undo**: Press 'U' key or click undo button
- **Flip Board**: Press 'F' key or click flip board button
- **Main Menu**: Return to mode selection

### Game Over Detection
- Checkmate with winner announcement
- Stalemate detection

## Performance Characteristics

### Search Speed
- **700 thousand NPS** on modern CPUs (single-threaded)
- Numba JIT compilation provides near-C performance
- First move may be slow due to JIT warmup (one-time cost)

### Search Depth
- Tactical positions: 8-12 ply in reasonable time
- Quiet positions: 12-16 ply with good move ordering
- Endgames: 16+ ply with specialized evaluation

### Memory Usage
- Transposition table: ~4GB (64M entries × 64 bytes)
- Adjustable via TT_SIZE constant

## Engine Strength

### Current Features
- **Tactical**: Excellent tactical vision with quiescence search
- **Positional**: Strong positional understanding via evaluation
- **Endgames**: Specialized endgame knowledge for basic endings
- **Opening**: Relies on search (opening book not yet implemented)

### Estimated Strength
- Approximate rating: 2000-2200 Elo (at depth 5)
- Capable of defeating intermediate players
- Tactical errors minimal due to deep search

## Advanced Configuration

### Search Parameters (in Minimax.py)
```python
MAX_SEARCH_DEPTH = 50  # Maximum ply depth
TT_SIZE = 1 << 26      # Transposition table size (64M entries)
ASPIRATION_WINDOW = 50 # Initial aspiration window
LMP_DEPTH = 6          # Late move pruning depth threshold
```

### Evaluation Weights (in Evaluation.py)
- Piece values
- Mobility bonuses
- King safety parameters
- Passed pawn bonuses
- Space evaluation weights

### Time Management
```python
searcher = SingleSearch(
    board, 
    depth=64,           # Max depth (use iterative deepening)
    time_limit=5000,    # Milliseconds
    increment=100,      # Fischer increment
    movestogo=30        # Moves to time control
)
```

## Troubleshooting

### First Move Delay
**Cause**: Numba JIT compilation occurs on first function call  
**Solution**: Expected behavior; subsequent moves will be fast

### Missing Piece Images
**Cause**: Image files not found in assets directory  
**Solution**: Engine uses fallback text rendering; add PNG files for pieces

### NumPy/Numba Version Conflicts
**Cause**: Incompatible package versions  
**Solution**: Use Python 3.8-3.12 with latest compatible Numba

### High Memory Usage
**Cause**: Large transposition table  
**Solution**: Reduce TT_SIZE constant in Minimax.py

### Performance Issues
**Cause**: Running in Python interpreter mode  
**Solution**: Ensure Numba JIT is enabled (check console for compilation messages)

## Future Roadmap

### Planned Features
- [ ] **Better Opening Book**: Better opening theory for faster early game
- [ ] **Endgame Tablebases**: Perfect play in simple endings (Syzygy format)
- [ ] **UCI Protocol**: Standard chess interface support
- [ ] **NNUE Evaluation**: Neural network-based position assessment
- [ ] **Multi-Threading**: Parallel search with Lazy SMP
- [ ] **Better Analysis Mode**: Position analysis with multiple variations
- [ ] **ELO Rating System**: Track player strength over time
- [ ] **Online Play**: Network multiplayer support
- [ ] **Mobile Version**: Touch-friendly interface for tablets
- [ ] **Configurable Difficulty**: Adjustable AI strength for casual play

### Optimization Targets
- [ ] Bitboard attack generation improvements
- [ ] Singular extension search
- [ ] Multi-cut pruning
- [ ] Razoring
- [ ] Probcut with dynamic margins

## Development

### Testing
```bash
# Perft testing (verify move generation correctness)
python Board_Move_gen.py

# Performance benchmarking
# (Uncomment test code in Board_Move_gen.py)
```

### Code Style
- Numba-compatible code in performance-critical paths
- Type hints where possible
- Clear variable naming for bitboard operations
- Comprehensive inline comments for complex algorithms

### Special Thanks
-For inspiration and guidance
- **Ethereal** 
- **Tantabus**
- **Boychesser**

### Acknowledgments
- Chess Programming Wiki for algorithm references
- Magic bitboard generation based on Tord Romstad's work

## License

MIT License

Copyright (c) 2025 SATYENDHRAN L

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

**Happy Chess!**