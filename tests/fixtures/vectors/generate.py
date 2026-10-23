"""Generate valid and invalid PII test vectors (TCKN, VKN, IBAN, PAN).

Outputs JSON files with {"valid": [...], "invalid": [...]} for each type.
All values are synthetic — no real personal data.
"""

import json
import os
import random

random.seed(12345)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── TCKN ────────────────────────────────────────────────────────────────────

def tckn_checksum(digits: list[int]) -> tuple[int, int]:
    """Compute TCKN d10 and d11."""
    A = sum(digits[i] for i in range(0, 9, 2))  # 1,3,5,7,9
    B = sum(digits[i] for i in range(1, 8, 2))  # 2,4,6,8
    d10 = (A * 7 - B) % 10
    d11 = (sum(digits) + d10) % 10
    return d10, d11


def generate_valid_tckn() -> str:
    """Generate a random valid TCKN."""
    digits = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
    d10, d11 = tckn_checksum(digits)
    return "".join(str(d) for d in digits + [d10, d11])


def validate_tckn(tckn: str) -> bool:
    """Validate a TCKN string."""
    if not tckn.isdigit() or len(tckn) != 11:
        return False
    if tckn[0] == "0":
        return False
    digits = [int(c) for c in tckn[:9]]
    expected_d10, expected_d11 = tckn_checksum(digits)
    return expected_d10 == int(tckn[9]) and expected_d11 == int(tckn[10])


def generate_tckn_vectors(n_valid: int = 20, n_invalid: int = 20) -> dict:
    valid = ["10000000146"]  # NVI canonical valid example
    valid += [generate_valid_tckn() for _ in range(n_valid - 1)]

    invalid = [
        "00000000000",   # starts with 0
        "12345678901",   # wrong checksum
        "1234567890",    # 10 digits
        "123456789012",  # 12 digits
        "ABC00000146",   # non-numeric
        "10000000147",   # valid checksum pattern but d11 wrong
        "10000000145",   # d10 wrong
        "99999999999",   # all 9s
        "11111111111",   # all 1s
        "1234",          # too short
    ]
    # Generate more random invalids (valid structure but wrong checksum)
    for _ in range(n_invalid - len(invalid)):
        digits = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
        # Corrupt one checksum digit
        d10, d11 = tckn_checksum(digits)
        d10 = (d10 + random.randint(1, 9)) % 10
        invalid.append("".join(str(d) for d in digits + [d10, d11]))

    return {"valid": valid, "invalid": invalid}


# ── VKN ─────────────────────────────────────────────────────────────────────

def vkn_checksum(digits: list[int]) -> int:
    """Compute VKN (10-digit) check digit using the authoritative GİB algorithm
    (matches python-stdnum stdnum.tr.vkn).

    Iterate the first 9 digits in reverse with i=1..9:
        c1 = (digit + i) % 10
        if c1: contribution = (c1 * 2**i) % 9 or 9
    check = (10 - sum) % 10
    """
    if len(digits) != 9:
        raise ValueError("VKN base must be 9 digits")
    s = 0
    for i, n in enumerate(reversed(digits), 1):
        c1 = (n + i) % 10
        if c1:
            c2 = (c1 * (2 ** i)) % 9 or 9
            s += c2
    return (10 - s) % 10


def generate_valid_vkn() -> str:
    """Generate a random valid VKN."""
    digits = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
    c10 = vkn_checksum(digits)
    return "".join(str(d) for d in digits + [c10])


def validate_vkn(vkn: str) -> bool:
    """Validate a VKN string."""
    if not vkn.isdigit() or len(vkn) != 10:
        return False
    digits = [int(c) for c in vkn[:9]]
    expected = vkn_checksum(digits)
    return expected == int(vkn[9])


def generate_vkn_vectors(n_valid: int = 15, n_invalid: int = 15) -> dict:
    # Authoritative valid examples (python-stdnum): 1111111114, 4540536920
    valid = ["1111111114", "4540536920"]
    valid += [generate_valid_vkn() for _ in range(n_valid - len(valid))]

    invalid = [
        "0000000000",  # stdnum rejects (check digit mismatch)
        "1000036101",  # NOT valid by the GİB algorithm (documented as a known bad example)
        "1111111111",
        "9999999999",
        "123456789",   # 9 digits
        "12345678901", # 11 digits
        "ABC0000000",
    ]
    for _ in range(n_invalid - len(invalid)):
        digits = [random.randint(1, 9)] + [random.randint(0, 9) for _ in range(8)]
        # Corrupt the check digit to guarantee invalidity
        c10 = (vkn_checksum(digits) + random.randint(1, 9)) % 10
        invalid.append("".join(str(d) for d in digits + [c10]))

    return {"valid": valid, "invalid": invalid}


# ── IBAN ─────────────────────────────────────────────────────────────────────

def _iban_to_numeric(iban: str) -> int:
    """Convert IBAN to numeric for MOD-97: move first 4 chars to end, A=10..Z=35."""
    rearranged = iban[4:] + iban[:4]
    numeric_str = ""
    for ch in rearranged:
        if ch.isalpha():
            numeric_str += str(ord(ch) - 55)
        else:
            numeric_str += ch
    return int(numeric_str)


def generate_valid_iban() -> str:
    """Generate a synthetic valid TR IBAN."""
    bank_code = f"{random.randint(0, 99999):05d}"
    reserved = "0"
    account = f"{random.randint(0, 9999999999999999):016d}"
    bban = bank_code + reserved + account

    # Compute check from TR00 + BBAN
    numeric_val = _iban_to_numeric("TR00" + bban)
    remainder = numeric_val % 97
    check = 98 - remainder
    check_str = f"{check:02d}"

    return f"TR{check_str}{bban}"


def validate_iban(iban: str) -> bool:
    """Validate TR IBAN via MOD-97."""
    cleaned = iban.replace(" ", "").upper()
    if not cleaned.startswith("TR") or len(cleaned) != 26:
        return False
    if not cleaned[2:].isdigit():
        return False
    return _iban_to_numeric(cleaned) % 97 == 1


def generate_iban_vectors(n_valid: int = 11, n_invalid: int = 11) -> dict:
    valid = ["TR470000100100000350930001"]  # canonical
    valid += [generate_valid_iban() for _ in range(n_valid - 1)]

    invalid = [
        "TR000000000000000000000000",
        "TR470000100100000350930000",  # off by 1
        "TR470000100100000350930002",
        "TR00000000000000000000000",   # 25 chars
        "TR0000000000000000000000000", # 27 chars
        "GB470000100100000350930001",  # wrong country
        "TRXX0000100100000350930001",  # non-numeric
        "TR0",
    ]
    # Generate more: valid structure but wrong check
    for _ in range(n_invalid - len(invalid)):
        bank_code = f"{random.randint(0, 99999):05d}"
        reserved = "0"
        account = f"{random.randint(0, 9999999999999999):016d}"
        bban = bank_code + reserved + account
        wrong_check = f"{(int(random.random() * 98) + 1):02d}"
        invalid.append(f"TR{wrong_check}{bban}")

    return {"valid": valid, "invalid": invalid}


# ── PAN (Luhn) ───────────────────────────────────────────────────────────────

def luhn_checksum(pan_digits: list[int]) -> int:
    """Compute Luhn checksum digit for 15-digit PAN base."""
    digits = list(pan_digits)
    for i in range(len(digits) - 1, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9
    total = sum(digits)
    return (10 - (total % 10)) % 10


def generate_valid_pan() -> str:
    """Generate a random valid 16-digit Luhn PAN."""
    # First digit 4 (Visa) or 5 (MasterCard)
    prefix = random.choice(["4", "5"])
    digits = [int(c) for c in prefix] + [random.randint(0, 9) for _ in range(14)]
    check = luhn_checksum(digits)
    digits.append(check)
    return "".join(str(d) for d in digits)


def validate_pan(pan: str) -> bool:
    """Validate PAN via Luhn algorithm."""
    if not pan.isdigit() or not (13 <= len(pan) <= 19):
        return False
    digits = [int(c) for c in pan[:-1]]
    expected = luhn_checksum(digits)
    return expected == int(pan[-1])


def generate_pan_vectors(n_valid: int = 10, n_invalid: int = 10) -> dict:
    valid = [generate_valid_pan() for _ in range(n_valid)]

    invalid = [
        "4532015112830360",  # non-Luhn
        "1234567890123456",
        "45320151128303",    # 14 digits, invalid
        "45320151128303600", # 17 digits, invalid
        "45320151128303AB",  # non-numeric
        "",
    ]
    for _ in range(n_invalid - len(invalid)):
        prefix = random.choice(["4", "5"])
        digits = [int(c) for c in prefix] + [random.randint(0, 9) for _ in range(14)]
        wrong_check = (luhn_checksum(digits) + random.randint(1, 9)) % 10
        digits.append(wrong_check)
        invalid.append("".join(str(d) for d in digits))

    return {"valid": valid, "invalid": invalid}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generators = [
        ("tckn_vectors.json", generate_tckn_vectors),
        ("vkn_vectors.json", generate_vkn_vectors),
        ("iban_vectors.json", generate_iban_vectors),
        ("pan_vectors.json", generate_pan_vectors),
    ]

    for filename, gen_func in generators:
        data = gen_func()
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  wrote {path} (valid={len(data['valid'])}, invalid={len(data['invalid'])})")

    print("Done.")


if __name__ == "__main__":
    main()
