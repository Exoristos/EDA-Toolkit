import random
import csv

# Veri havuzumuz
ads = ["Ali", "Veli", "Ayse", "Fatma", "Ahmet", "Mehmet", "Zeynep", "Elif", "Burak", "Omer", "Yusuf", "Canan", "Gizem", "Emre", "Kaan"]
soyads = ["Yilmaz", "Kaya", "Demir", "Celik", "Sahin", "Yildiz", "Arslan", "Cetin", "Ozturk", "Koc", "Kurt", "Ozdemir", "Polat"]
sektors = ["Finans", "Teknoloji", "Hizmet", "Saglik", "Egitim", "Uretim", "Perakende"]
domains = ["test.com", "banka.local", "sirket.com", "mail.com"]

# Burayı ne kadar data istiyorsan o sayıyla değiştir
uretilecek_satir_sayisi = 1000
dosya_adi = "sahte_veri_uretildi.csv"

with open(dosya_adi, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    # Başlıkları yazdırıyoruz
    writer.writerow(["musteri_id", "ad", "soyad", "tc_kimlik", "iban", "telefon", "email", "yas", "gelir", "kredi_skoru", "sektor"])
    
    # Satırları basıyoruz
    for i in range(uretilecek_satir_sayisi):
        ad = random.choice(ads)
        soyad = random.choice(soyads)
        tc = str(random.randint(10000000000, 99999999999))
        iban = "TR" + "".join([str(random.randint(0, 9)) for _ in range(24)])
        tel = "5" + "".join([str(random.randint(0, 9)) for _ in range(9)])
        email = f"{ad.lower()}.{soyad.lower()}@{random.choice(domains)}"
        yas = random.randint(22, 65)
        gelir = round(random.uniform(15000, 120000), 2) # Maaş skalası
        skor = random.randint(500, 1900)
        sektor = random.choice(sektors)
        
        # Dosyaya ekleme
        writer.writerow([f"P{i:04d}", ad, soyad, tc, iban, tel, email, yas, gelir, skor, sektor])

print(f"İşlem tamam kanka. {uretilecek_satir_sayisi} satır veri '{dosya_adi}' dosyasına yazıldı.")