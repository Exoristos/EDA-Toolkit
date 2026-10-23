"""Generate synthetic CSVs for advanced EDA tests."""

import csv
import os
import random
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(42)


def generate_numeric_clusters(n: int = 200) -> list[dict]:
    """Generate numeric_heavy_with_clusters.csv — known clusters + outliers + correlated columns."""
    rows = []
    # 3 clusters
    centers = [
        (20, 1000, 300),   # cluster 0: young, low income, low score
        (45, 5000, 600),   # cluster 1: middle, mid income, mid score
        (65, 9000, 800),   # cluster 2: old, high income, high score
    ]
    # yas2 is highly correlated with yas (linear + noise)
    # gelir2 is highly correlated with gelir
    # skor2 moderately correlated with skor

    for i in range(n):
        cl = random.randint(0, 2)
        cx, cy, cz = centers[cl]
        yas = max(18, min(80, round(random.gauss(cx, 5))))
        gelir = max(100, round(random.gauss(cy, 500), 2))
        skor = max(200, min(950, round(random.gauss(cz, 40))))

        # Correlated columns
        yas2 = max(18, min(80, yas + round(random.gauss(0, 2))))
        gelir2 = max(100, round(gelir + random.gauss(0, 300), 2))
        skor2 = max(200, min(950, round(skor + random.gauss(0, 20))))
        # Anti-correlated (negatively): yas_neg ~ -0.7 * gelir + noise
        yas_neg = max(18, min(80, round(70 - (gelir / 8000) * 60 + random.gauss(0, 8), 2)))

        # Inject outliers on skor (manual outliers)
        skor_val = skor
        if i < 10:  # first 10 rows have outlier skor values
            skor_val = random.choice([100, 150, 20000, 9500, 50])

        # ID column
        cat_col = random.choice(["A", "B", "C", "D"])

        # Missing some values
        if random.random() < 0.05:
            yas = None
        if random.random() < 0.04:
            gelir = None

        rows.append({
            "id": f"AC{i:04d}",
            "yas": "" if yas is None else str(yas),
            "gelir": "" if gelir is None else str(gelir),
            "skor": str(skor_val),
            "yas2": str(yas2),
            "gelir2": str(gelir2),
            "skor2": str(skor2),
            "yas_neg": str(yas_neg),
            "cat_col": cat_col,
        })
    return rows


def generate_degenerate(n: int = 30) -> list[dict]:
    """Generate degenerate.csv — edge cases: single numeric col, constant col, tiny rows."""
    rows = []
    for i in range(n):
        # single_numeric: only one numeric column (but we hide poly_numeric for degenerate case)
        # Actually for the degenerate test, we'll have: constant_col, single_num (varying), tiny size
        rows.append({
            "id": str(i + 1),
            "constant_col": "100",       # all same
            "single_num": str(random.randint(1, 10)),
            "text_col": random.choice(["x", "y", "z"]),
        })
    return rows


def write_csv(filename: str, rows: list[dict], fieldnames: list[str]):
    path = os.path.join(SCRIPT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    print("Generating advanced EDA test CSVs...")

    rows = generate_numeric_clusters(200)
    write_csv("numeric_clusters.csv", rows, ["id", "yas", "gelir", "skor", "yas2", "gelir2", "skor2", "yas_neg", "cat_col"])

    rows = generate_degenerate(30)
    write_csv("degenerate.csv", rows, ["id", "constant_col", "single_num", "text_col"])

    print("Done.")


if __name__ == "__main__":
    main()
