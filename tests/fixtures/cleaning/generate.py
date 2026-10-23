"""Generate synthetic CSVs for cleaning module tests."""
import csv
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
random.seed(42)

# ─── 1. dirty_strings.csv — whitespace, missing markers, type string issues ───
def generate_dirty_strings(n: int = 30):
    rows = []
    for i in range(n):
        yas = random.randint(18, 75)
        if random.random() < 0.1:
            yas_value = " " + str(yas) + "  "  # whitespace padded
        elif random.random() < 0.15:
            yas_value = str(yas)
        else:
            yas_value = random.choice(["", "NA", "N/A", "na", "n/a", "-", "null", "NULL", "yok", "-"])

        cinsiyet = random.choice([" evet", "hayır ",  " Evet ", "Hayır", "e", "h", "1", "0", "true", "false", "", "NA"])
        sehir = random.choice([" Istanbul ", " Ankara", "İzmir", "  Bursa  ", "Antalya", "", "NA"])

        rows.append({
            "id": i + 1,
            "yas": yas_value,
            "cinsiyet": cinsiyet,
            "sehir": sehir,
            "notlar": random.choice(["  ", "", " test ", "yok"]),
        })
    return rows

# ─── 2. clean_data.csv — already clean, no issues ───
def generate_clean_data(n: int = 20):
    rows = []
    for i in range(n):
        rows.append({
            "id": i + 1,
            "yas": random.randint(20, 60),
            "gelir": round(random.lognormvariate(9.5, 0.5), 2),
            "sehir": random.choice(["Istanbul", "Ankara", "Izmir", "Bursa"]),
        })
    return rows

# ─── 3. missing_values.csv — high missing rate columns ───
def generate_missing_values(n: int = 50):
    rows = []
    for i in range(n):
        A = random.random() < 0.5 and str(random.uniform(1, 100)) or ""
        B = random.random() < 0.3 and random.randint(1, 100) or None
        # column hepsi_bos is empty for most rows
        C = None
        if random.random() < 0.05:
            C = random.randint(1, 10)
        rows.append({
            "id": i + 1,
            "yarim_bos": A,
            "az_bos": B,
            "hepsi_bos": C,
            "dolu": random.randint(50, 200),
        })
    return rows

# ─── 4. outliers.csv — numeric columns with clear outliers ───
def generate_outliers(n: int = 50):
    rows = []
    for i in range(n):
        normal = round(random.normalvariate(100, 15), 2)
        if i < 5:
            outlier_val = round(random.uniform(250, 500), 2)
        else:
            outlier_val = normal
        rows.append({
            "id": i + 1,
            "normal_col": round(normal, 2),
            "outlier_col": outlier_val,
            "skoru": random.randint(300, 850),
        })
    return rows

# ─── 5. bool_coercion.csv — Turkish boolean values ───
def generate_bool_coercion(n: int = 20):
    rows = []
    for i in range(n):
        rows.append({
            "id": i + 1,
            "onay": random.choice(["evet", "hayır", "1", "0", "true", "false", "e", "h"]),
            "aktif": random.choice(["evet", "hayır", "", "NA", "1"]),
            "tutar": round(random.uniform(10, 500), 2),
        })
    return rows

# ─── 6. all_types.csv — mixed everything for comprehensive tests ───
def generate_all_types(n: int = 40):
    rows = []
    for i in range(n):
        # string with whitespace and missing markers
        ad = random.choice([" Ali ", "  Veli", "Mehmet", "", "NA", "  ", "-", "Ayşe  "])
        # numeric-ish with missing
        yas = random.choice([str(random.randint(18, 70)), " ", "NA", "-", "yok"])
        # boolean-ish
        evli = random.choice([" evet", "hayır ", "e", "h", "true", "", "NA"])
        # numeric with outliers
        if i < 3:
            maas = round(random.uniform(80000, 150000), 2)
        elif random.random() < 0.15:
            maas = None
        else:
            maas = round(random.lognormvariate(9.0, 0.6), 2)
        # categorical with missing
        sehir = random.choice(["İstanbul", "Ankara", "", "NA", "Bursa"])
        # numeric with missing
        skor = random.choices([random.randint(300, 850), None], weights=[0.85, 0.15])[0]
        # all missing or mostly empty column
        bos_kolon = random.choices(["", "NA", None, "deger"], weights=[0.4, 0.3, 0.25, 0.05])[0]

        rows.append({
            "id": i + 1,
            "ad": ad,
            "yas": yas,
            "evli": evli,
            "maas": maas if maas is not None else "",
            "sehir": sehir,
            "skor": skor if skor is not None else "",
            "bos_kolon": bos_kolon if bos_kolon is not None else "",
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
    print("Generating cleaning test CSVs...")

    rows = generate_dirty_strings(30)
    write_csv("dirty_strings.csv", rows, ["id", "yas", "cinsiyet", "sehir", "notlar"])

    rows = generate_clean_data(20)
    write_csv("clean_data.csv", rows, ["id", "yas", "gelir", "sehir"])

    rows = generate_missing_values(50)
    write_csv("missing_values.csv", rows, ["id", "yarim_bos", "az_bos", "hepsi_bos", "dolu"])

    rows = generate_outliers(50)
    write_csv("outliers.csv", rows, ["id", "normal_col", "outlier_col", "skoru"])

    rows = generate_bool_coercion(20)
    write_csv("bool_coercion.csv", rows, ["id", "onay", "aktif", "tutar"])

    rows = generate_all_types(40)
    write_csv("all_types.csv", rows, ["id", "ad", "yas", "evli", "maas", "sehir", "skor", "bos_kolon"])

    print("Done.")


if __name__ == "__main__":
    main()
