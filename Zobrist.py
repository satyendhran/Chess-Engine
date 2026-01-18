import numpy as np

_rng = np.random.default_rng(20250107)

piece_keys = _rng.integers(
    low=0,
    high=np.uint64(2**64 - 1),
    size=(12, 64),
    dtype=np.uint64,
)

enpassant_keys = _rng.integers(
    low=0,
    high=np.uint64(2**64 - 1),
    size=64,
    dtype=np.uint64,
)

castle_keys = _rng.integers(
    low=0,
    high=np.uint64(2**64 - 1),
    size=16,
    dtype=np.uint64,
)

side_key = np.uint64(_rng.integers(0, np.uint64(2**64 - 1), dtype=np.uint64))
