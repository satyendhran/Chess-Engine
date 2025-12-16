import pstats
p = pstats.Stats("gui_loop.prof")
p.sort_stats("cumtime").print_stats()
