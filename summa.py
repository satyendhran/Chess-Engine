# import os
# import ast
# import sys

# # --- Fix Windows Unicode output ---
# try:
#     sys.stdout.reconfigure(encoding="utf-8")
#     TREE = {
#         "branch": "├── ",
#         "last": "└── ",
#         "pipe": "│   ",
#         "space": "    ",
#     }
# except Exception:
#     # ASCII fallback (never crashes)
#     TREE = {
#         "branch": "+-- ",
#         "last": "`-- ",
#         "pipe": "|   ",
#         "space": "    ",
#     }

# CURRENT_FILE = os.path.abspath(__file__)

# def get_classes_and_functions(file_path):
#     result = {None: []}

#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             tree = ast.parse(f.read(), filename=file_path)

#         for node in tree.body:
#             if isinstance(node, ast.FunctionDef):
#                 result[None].append(node.name)
#             elif isinstance(node, ast.ClassDef):
#                 methods = [
#                     n.name for n in node.body if isinstance(n, ast.FunctionDef)
#                 ]
#                 result[node.name] = methods

#     except Exception as e:
#         result[None].append(f"<parse error: {e}>")

#     return result


# def print_directory_structure(root_dir, prefix=""):
#     try:
#         entries = sorted(os.listdir(root_dir))
#     except PermissionError:
#         return

#     visible = [
#         e for e in entries
#         if not e.startswith(".") and e != "__pycache__"
#     ]

#     for i, entry in enumerate(visible):
#         path = os.path.join(root_dir, entry)

#         # Skip current script
#         if os.path.abspath(path) == CURRENT_FILE:
#             continue

#         is_last = i == len(visible) - 1
#         connector = TREE["last"] if is_last else TREE["branch"]
#         next_prefix = TREE["space"] if is_last else TREE["pipe"]

#         if os.path.isdir(path):
#             print(f"{prefix}{connector}{entry}/")
#             print_directory_structure(path, prefix + next_prefix)

#         elif entry.endswith(".py"):
#             print(f"{prefix}{connector}{entry}")
#             content = get_classes_and_functions(path)

#             # Top-level functions
#             for func in content.get(None, []):
#                 print(f"{prefix}{next_prefix}{TREE['last']}{func}()")

#             # Classes
#             for cls, methods in content.items():
#                 if cls is not None:
#                     print(f"{prefix}{next_prefix}{TREE['last']}class {cls}")
#                     for m in methods:
#                         print(f"{prefix}{next_prefix}{TREE['space']}{TREE['last']}{m}()")


# if __name__ == "__main__":
#     print_directory_structure(".")


from statistics import mean, median, stdev
from time import perf_counter_ns as pf

from Board import Board, parse_fen
from Evaluation import Evaluation as E1
from Evaluation2 import Evaluation as E2



def test(Eval, val):
    N = 5600086
    times = []
    b = Board(*parse_fen(b"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
    for _ in range(3):
        Eval(b).evaluate()

    with open("fens.txt", "rb") as f:
        for _ in range(N):
            fen = f.readline().rstrip(b"\n")
            b = Board(*parse_fen(fen))
            t1 = pf()
            Eval(b).evaluate()
            t2 = pf()
            times.append(t2 - t1)

    mn = min(times)
    mx = max(times)
    avg = mean(times)
    med = median(times)
    sd = stdev(times) if N > 1 else 0

    print(f"Evaluation Timing Statistics for Verison {val}")
    print("---------------------------------")
    print(f"Runs            : {N}")
    print(f"Min             : {mn}")
    print(f"Max             : {mx}")
    print(f"Mean            : {avg:.2f}")
    print(f"Median          : {med}")
    print(f"Std Deviation   : {sd:.2f}")
    print(f"Ops / second    : {1e9 / avg:.2f}")
    print(f"Avg (microsec)  : {avg / 1e3:.2f}")
    print(f"Avg (millisec)  : {avg / 1e6:.4f}")


test(E1, 1)
test(E2, 2)



# Evaluation Timing Statistics
# ------------------------------
# Runs            : 100
# Min             : 8600
# Max             : 66500
# Mean            : 9464.00
# Median          : 9100.0
# Std Deviation   : 1002.07
# Ops / second    : 105663.57
# Avg (microsec)  : 9.46
# Avg (millisec)  : 0.0095
