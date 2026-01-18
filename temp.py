import csv

CSV_FILE = r"C:\Users\Satyendhran\Downloads\lichess_db_puzzle.csv"  # replace with your CSV path
FEN_COLUMN = "FEN"  # name of the column containing FENs
OUTPUT_FILE = "fens.txt"

with (
    open(CSV_FILE, newline="", encoding="utf-8") as csvfile,
    open(OUTPUT_FILE, "w", encoding="utf-8") as fout,
):
    reader = csv.DictReader(csvfile)

    for row in reader:
        fen = row[FEN_COLUMN].strip()
        fout.write(fen + "\n")
