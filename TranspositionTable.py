
import numpy as np
from numba import njit, int8, int32, int64, uint8, uint32, uint64
from numba.experimental import jitclass





@njit(uint32(uint32))
def xorshift32(state):
    """Fast PRNG for hash initialization"""
    state ^= state << np.uint32(13)
    state ^= state >> np.uint32(17)
    state ^= state << np.uint32(5)
    return state


@njit(uint64(uint32))
def generate_64bit_hash(state):
    """Generate 64-bit hash from 32-bit seed"""
    state = xorshift32(state)
    n1 = uint64(state)
    state = xorshift32(state)
    n2 = uint64(state)
    state = xorshift32(state)
    n3 = uint64(state)
    state = xorshift32(state)
    n4 = uint64(state)

    result = (n1 << uint64(48)) | (n2 << uint64(32)) | (n3 << uint64(16)) | n4
    return result


def initialize_zobrist_keys(seed=0x2C3E4F5A):

    state = uint32(seed)

    
    piece_keys = np.zeros((12, 64), dtype=np.uint64)
    for piece in range(12):
        for square in range(64):
            state = xorshift32(state)
            piece_keys[piece, square] = generate_64bit_hash(state)

    
    castling_keys = np.zeros(16, dtype=np.uint64)
    for i in range(16):
        state = xorshift32(state)
        castling_keys[i] = generate_64bit_hash(state)

    
    enpassant_keys = np.zeros(64, dtype=np.uint64)
    for i in range(64):
        state = xorshift32(state)
        enpassant_keys[i] = generate_64bit_hash(state)

    
    state = xorshift32(state)
    side_key = generate_64bit_hash(state)

    return piece_keys, castling_keys, enpassant_keys, side_key



ZOBRIST_PIECE_KEYS, ZOBRIST_CASTLE_KEYS, ZOBRIST_EP_KEYS, ZOBRIST_SIDE_KEY = initialize_zobrist_keys()







FLAG_EXACT = uint8(0)   
FLAG_ALPHA = uint8(1)   
FLAG_BETA = uint8(2)    


@njit(uint64(uint64[:], uint8, uint8, uint8), inline='always')
def compute_zobrist_hash(bitboards, side, castle, enpassant):

    h = uint64(0)

    
    for piece in range(12):
        bb = bitboards[piece]
        square = 0
        while bb:
            if bb & uint64(1):
                h ^= ZOBRIST_PIECE_KEYS[piece, square]
            bb >>= uint64(1)
            square += 1

    
    h ^= ZOBRIST_CASTLE_KEYS[castle]

    
    if enpassant < 64:
        h ^= ZOBRIST_EP_KEYS[enpassant]

    
    if side == 1:
        h ^= ZOBRIST_SIDE_KEY

    return h






tt_spec = [
    ('keys', uint64[:]),         
    ('values', uint64[:]),       
    ('size', uint64),            
    ('mask', uint64),            
    ('hits', uint64),            
    ('collisions', uint64),      
]


@jitclass(tt_spec)
class TranspositionTable:
    def __init__(self, size_mb=256):
        
        bytes_total = size_mb * 1024 * 1024
        num_entries = bytes_total // 16

        
        power = int(np.log2(num_entries))
        self.size = uint64(1) << uint64(power)
        self.mask = self.size - uint64(1)

        
        self.keys = np.zeros(self.size, dtype=np.uint64)
        self.values = np.zeros(self.size, dtype=np.uint64)

        
        self.hits = uint64(0)
        self.collisions = uint64(0)

    def clear(self):
        """Clear all entries"""
        self.keys.fill(0)
        self.values.fill(0)
        self.hits = uint64(0)
        self.collisions = uint64(0)

    def probe(self, hash_key, depth, alpha, beta, ply):

        index = hash_key & self.mask
        stored_key = self.keys[index]

        
        if stored_key != hash_key:
            return (False, int32(0), uint32(0), uint8(0))

        self.hits += uint64(1)

        
        packed = self.values[index]
        best_move = uint32(packed & uint64(0xFFFF))
        score = int32((packed >> uint64(16)) & uint64(0xFFFF))
        
        if score & int32(0x8000):
            score |= int32(0xFFFF0000)

        entry_depth = uint8((packed >> uint64(32)) & uint64(0xFF))
        flag = uint8((packed >> uint64(48)) & uint64(0x3))

        
        if entry_depth < depth:
            
            return (False, score, best_move, flag)

        
        if score > int32(90000):  
            score -= int32(ply)
        elif score < int32(-90000):
            score += int32(ply)

        
        if flag == FLAG_EXACT:
            return (True, score, best_move, flag)
        elif flag == FLAG_ALPHA and score <= alpha:
            return (True, alpha, best_move, flag)
        elif flag == FLAG_BETA and score >= beta:
            return (True, beta, best_move, flag)

        
        return (False, score, best_move, flag)

    def store(self, hash_key, depth, score, best_move, flag, ply, age):

        index = hash_key & self.mask
        stored_key = self.keys[index]

        
        if stored_key != uint64(0) and stored_key != hash_key:
            self.collisions += uint64(1)

            
            old_packed = self.values[index]
            old_depth = uint8((old_packed >> uint64(32)) & uint64(0xFF))
            old_age = uint8((old_packed >> uint64(40)) & uint64(0xFF))

            
            if old_depth > depth and old_age == age:
                return

        
        adjusted_score = score
        if score > int32(90000):
            adjusted_score = score + int32(ply)
        elif score < int32(-90000):
            adjusted_score = score - int32(ply)

        
        if adjusted_score > int32(32767):
            adjusted_score = int32(32767)
        elif adjusted_score < int32(-32768):
            adjusted_score = int32(-32768)

        
        packed = uint64(0)
        packed |= uint64(best_move & uint32(0xFFFF))
        packed |= (uint64(adjusted_score & int32(0xFFFF)) << uint64(16))
        packed |= (uint64(depth) << uint64(32))
        packed |= (uint64(age & uint8(0xFF)) << uint64(40))
        packed |= (uint64(flag & uint8(0x3)) << uint64(48))

        
        self.keys[index] = hash_key
        self.values[index] = packed

    def prefetch(self, hash_key):

        pass  

    def get_pv_move(self, hash_key):

        index = hash_key & self.mask
        if self.keys[index] != hash_key:
            return uint32(0)

        packed = self.values[index]
        return uint32(packed & uint64(0xFFFF))

    def get_hashfull(self):

        sample_size = min(uint64(1000), self.size)
        used = uint64(0)

        for i in range(sample_size):
            if self.keys[i] != uint64(0):
                used += uint64(1)

        return uint32((used * uint64(1000)) // sample_size)






rep_spec = [
    ('hashes', uint64[:]),       
    ('head', uint32),            
    ('size', uint32),            
]


@jitclass(rep_spec)
class RepetitionTable:


    def __init__(self, max_positions=512):

        self.size = uint32(max_positions)
        self.hashes = np.zeros(max_positions, dtype=np.uint64)
        self.head = uint32(0)

    def clear(self):

        self.hashes.fill(0)
        self.head = uint32(0)

    def push(self, hash_key):
        
        self.hashes[self.head] = hash_key
        self.head = (self.head + uint32(1)) % self.size

    def pop(self):
        """Remove most recent position"""
        if self.head == uint32(0):
            self.head = self.size - uint32(1)
        else:
            self.head -= uint32(1)
        self.hashes[self.head] = uint64(0)

    def is_repetition(self, hash_key, halfmove_clock):
       
        count = uint32(0)

        
        positions_to_check = min(uint32(halfmove_clock), self.head)

        
        i = self.head
        checked = uint32(0)

        while checked < positions_to_check:
            
            if i < uint32(2):
                break
            i -= uint32(2)
            checked += uint32(2)

            if self.hashes[i] == hash_key:
                count += uint32(1)
                if count >= uint32(2):  
                    return count

        return count

    def get_history_size(self):
        """Get number of positions in history"""
        return self.head






@njit
def tt_cutoff(tt, hash_key, depth, alpha, beta, ply):
    
    found, score, best_move, flag = tt.probe(hash_key, depth, alpha, beta, ply)

    if found:
        return (True, score, best_move)

    return (False, int32(0), best_move)


@njit
def update_pv(pv_table, pv_length, ply, move):
   
    pv_table[ply, 0] = move
    for i in range(pv_length[ply + 1]):
        pv_table[ply, i + 1] = pv_table[ply + 1, i]
    pv_length[ply] = pv_length[ply + 1] + uint32(1)






@njit
def search_with_tt(board, depth, alpha, beta, ply, tt, rep_table, age):
    
    
    hash_key = compute_zobrist_hash(
        board.bitboard,
        board.side,
        board.castle,
        board.enpassant
    )

    
    if ply > 0:
        rep_count = rep_table.is_repetition(hash_key, board.halfmove)
        if rep_count >= uint32(1):  
            return int32(0)  

    
    tt_cutoff_found, tt_score, tt_move = tt_cutoff(tt, hash_key, depth, alpha, beta, ply)
    if tt_cutoff_found:
        return tt_score

    
    

    
    return int32(0)


def print_tt_stats(tt):
    """Print transposition table statistics"""
    print(f"\n=== Transposition Table Statistics ===")
    print(f"Size: {tt.size:,} entries ({(tt.size * 16) / (1024*1024):.1f} MB)")
    print(f"Hits: {tt.hits:,}")
    print(f"Collisions: {tt.collisions:,}")
    print(f"Occupancy: {tt.get_hashfull() / 10:.1f}%")

    if tt.hits > 0:
        hit_rate = (tt.hits * 100.0) / (tt.hits + tt.collisions + 1)
        print(f"Hit rate: {hit_rate:.2f}%")






if __name__ == "__main__":
    
    tt = TranspositionTable(size_mb=128)
    rep_table = RepetitionTable(max_positions=512)

    print(f"Transposition table initialized: {tt.size:,} entries")
    print(f"Memory usage: {(tt.size * 16) / (1024*1024):.1f} MB")

    
    test_hash = uint64(0x123456789ABCDEF0)
    tt.store(
        test_hash,     
        5,             
        123,           
        0,             
        FLAG_EXACT,    
        0,             
        uint8(0)       
    )


    found, score, move, flag = tt.probe(
        test_hash,
        uint8(8),
        int32(-1000),
        int32(1000),
        uint8(5)
    )

    print(f"\nTest probe: found={found}, score={score}, move={move}")

    
    test_positions = [uint64(0x1111), uint64(0x1111), uint64(0x1111)]
    for h in test_positions:
        rep_table.push(h)

    rep_count = rep_table.is_repetition(uint64(0x1111), uint32(10))
    print(f"Repetition count: {rep_count}")

    print("\nTransposition table ready for integration!")