"""Generate synthetic bank-ish data CSVs for testing and examples."""

import csv
import random
import os
import sys

# Ensure output goes to both tests/fixtures/data/ and examples/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR  # tests/fixtures/data/
# SCRIPT_DIR is .../tests/fixtures/data, go up 3 levels to project root, then into examples
EXAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))),
    "examples"
)

random.seed(42)


def generate_tckn_checksum(digits: list[int]) -> tuple[int, int]:
    """Compute TCKN checksum digits (positions 10 and 11)."""
    A = sum(digits[i] for i in range(0, 9, 2))  # odd positions: 0,2,4,6,8
    B = sum(digits[i] for i in range(1, 8, 2))  # even positions: 1,3,5,7
    d10 = (A * 7 - B) % 10
    d11 = (sum(digits) + d10) % 10
    return d10, d11


def generate_tckn() -> str:
    """Generate a valid synthetic TCKN."""
    digits = [random.randint(1, 9)]  # first digit 1-9
    digits.extend([random.randint(0, 9) for _ in range(8)])
    d10, d11 = generate_tckn_checksum(digits)
    digits.append(d10)
    digits.append(d11)
    return "".join(str(d) for d in digits)


def generate_iban() -> str:
    """Generate a synthetic valid TR IBAN (TR + 24 digits, MOD-97 valid).

    ISO 13616 check-digit computation: move the country code + placeholder
    check digits ("TR00") to the END before the MOD-97 step.
    """
    # Create BBAN: 5-digit bank code + 1 digit reserved + 16-digit account = 22 digits
    bank_code = f"{random.randint(0, 99999):05d}"
    reserved = "0"
    account = f"{random.randint(0, 9999999999999999):016d}"
    bban = bank_code + reserved + account

    # Rearranged form: BBAN + "TR00", then letters -> numbers (T=29, R=27)
    rearranged = bban + "TR00"
    numeric_str = ""
    for ch in rearranged:
        if ch.isalpha():
            numeric_str += str(ord(ch) - 55)
        else:
            numeric_str += ch
    remainder = int(numeric_str) % 97
    check = 98 - remainder
    check_str = f"{check:02d}"

    return f"TR{check_str}{bban}"


def generate_phone() -> str:
    """Generate a synthetic Turkish phone number with 0 prefix (matches spec regex)."""
    return f"05{random.randint(10, 99):02d}{random.randint(0, 9999999):07d}"


def generate_email(name: str, surname: str) -> str:
    """Generate a synthetic email."""
    domains = ["example.com", "test.com", "sirket.com", "banka.local"]
    return f"{name.lower()}.{surname.lower()}@{random.choice(domains)}"


def generate_numeric_heavy(n: int = 200):
    """numeric_heavy.csv — mostly numeric, some missing, outliers."""
    rows = []
    for i in range(n):
        yas = random.randint(18, 75)
        gelir = round(random.lognormvariate(9.5, 0.8), 2)
        kredi_skoru = random.randint(300, 900)
        aylik_odeme = round(random.lognormvariate(6.5, 0.6), 2)
        bakiye = round(random.normalvariate(5000, 25000), 2)
        kredi_tutari = round(random.lognormvariate(10, 0.9), 2) if random.random() > 0.1 else None

        # Inject outliers
        if random.random() < 0.03:
            bakiye = round(random.uniform(200000, 500000), 2)
        # Missing values
        if random.random() < 0.05:
            yas = None  # type: ignore
        if random.random() < 0.03:
            gelir = None
        if random.random() < 0.04:
            kredi_skoru = None

        rows.append({
            "musteri_id": f"N{i:04d}",
            "yas": yas,
            "gelir": gelir,
            "kredi_skoru": kredi_skoru,
            "aylik_odeme": aylik_odeme,
            "bakiye": bakiye,
            "kredi_tutari": kredi_tutari,
        })

    return rows


def generate_mixed_with_pii(n: int = 200):
    """mixed_with_pii.csv — PII columns + numeric cols."""
    names_m = ["Ali", "Mehmet", "Ahmet", "Mustafa", "Hasan", "Huseyin", "Ibrahim",
               "Osman", "Yusuf", "Murat", "Kemal", "Omer", "Fatih", "Emre", "Burak"]
    names_f = ["Ayse", "Fatma", "Emine", "Zeynep", "Hatice", "Elif", "Meryem",
               "Sibel", "Derya", "Gul", "Sevgi", "Nur", "Canan", "Pinar", "Deniz"]
    surnames = ["Yilmaz", "Kaya", "Demir", "Celik", "Sahin", "Yildiz", "Ozdemir",
                "Aydin", "Arslan", "Dogan", "Kilic", "Aslan", "Cetin", "Kara", "Koc"]

    rows = []
    for i in range(n):
        is_female = random.random() < 0.45
        ad = random.choice(names_f if is_female else names_m)
        soyad = random.choice(surnames)
        tckn = generate_tckn()
        iban = generate_iban()
        telefon = generate_phone()
        email = generate_email(ad, soyad)

        # Some missing PII
        if random.random() < 0.03:
            tckn = ""
        if random.random() < 0.02:
            telefon = ""

        rows.append({
            "musteri_id": f"P{i:04d}",
            "ad": ad,
            "soyad": soyad,
            "tc_kimlik": tckn,
            "iban": iban,
            "telefon": telefon,
            "email": email,
            "yas": random.randint(18, 80),
            "gelir": round(random.lognormvariate(9.5, 0.8), 2),
            "kredi_skoru": random.randint(300, 900),
            "sektor": random.choice(["Tarim", "Sanayi", "Hizmet", "Teknoloji", "Finans", "Egitim"]),
        })

    return rows


def generate_categorical(n: int = 200):
    """categorical.csv — mostly categorical + one numeric target."""
    iller = ["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya", "Adana", "Konya",
             "Gaziantep", "Mersin", "Kayseri", "Eskisehir", "Samsun", "Diyarbakir", "Trabzon"]
    sektorler = ["Tarim", "Sanayi", "Hizmet", "Teknoloji", "Finans", "Egitim", "Saglik", "Turizm"]
    urunler = ["Kredi", "Mevduat", "Fon", "Sigorta", "Kredi_Karti", "EFT", "Yatirim"]
    kanallar = ["Sube", "Internet", "Mobil", "ATM", "Cagri_Merkezi"]
    risk_gruplari = ["Dusuk", "Orta", "Yuksek", "Cok_Yuksek"]

    rows = []
    for i in range(n):
        il = random.choice(iller)
        sektor = random.choice(sektorler)
        urun_tipi = random.choice(urunler)
        kanal = random.choice(kanallar)
        risk_grubu = random.choice(risk_gruplari)

        # Target: risk skoru
        if risk_grubu == "Dusuk":
            risk_skoru = round(random.uniform(50, 200), 2)
        elif risk_grubu == "Orta":
            risk_skoru = round(random.uniform(200, 450), 2)
        elif risk_grubu == "Yuksek":
            risk_skoru = round(random.uniform(450, 700), 2)
        else:
            risk_skoru = round(random.uniform(700, 1000), 2)

        # Inject some missing values
        if random.random() < 0.04:
            sektor = ""
        if random.random() < 0.03:
            risk_grubu = ""

        rows.append({
            "musteri_id": f"C{i:04d}",
            "il": il,
            "sektor": sektor,
            "urun_tipi": urun_tipi,
            "kanal": kanal,
            "risk_grubu": risk_grubu,
            "risk_skoru": risk_skoru,
        })

    return rows


def write_csv(filename: str, rows: list[dict], fieldnames: list[str]):
    """Write a CSV to both tests/fixtures/data/ and examples/."""
    paths = []
    for base_dir in [OUTPUT_DIR, EXAMPLES_DIR]:
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
        print(f"  wrote {path} ({len(rows)} rows)")
    return paths[0]


def main():
    print("Generating synthetic bank-ish data...")

    # 1. numeric_heavy.csv
    rows = generate_numeric_heavy(200)
    fields = ["musteri_id", "yas", "gelir", "kredi_skoru", "aylik_odeme", "bakiye", "kredi_tutari"]
    write_csv("numeric_heavy.csv", rows, fields)

    # 2. mixed_with_pii.csv
    rows = generate_mixed_with_pii(200)
    fields = ["musteri_id", "ad", "soyad", "tc_kimlik", "iban", "telefon", "email",
              "yas", "gelir", "kredi_skoru", "sektor"]
    write_csv("mixed_with_pii.csv", rows, fields)

    # 3. categorical.csv
    rows = generate_categorical(200)
    fields = ["musteri_id", "il", "sektor", "urun_tipi", "kanal", "risk_grubu", "risk_skoru"]
    write_csv("categorical.csv", rows, fields)

    print("Done.")


if __name__ == "__main__":
    main()
