import pandas as pd
import random

# Load data 420 tweet awal milik pengguna
raw_file = "dataset_timnas_siap_upload_1000.csv"
df_orig = pd.read_csv(raw_file)
orig_tweets = df_orig["tweet"].dropna().tolist()
print(f"Data asli yang sudah ada: {len(orig_tweets)} tweet")

# Variasi Tambahan Positif (Dukungan, Taktik, Juara, Progres, Ranking FIFA)
pos_subjects = [
    "Taktik pelatih timnas", "Strategi kepelatihan coach", "Kedisiplinan pemain era pelatih sekarang",
    "Gaya permainan timnas saat ini", "Filosofi sepak bola pelatih", "Kepemimpinan pelatih kepala",
    "Fisik dan stamina timnas", "Pergantian pemain babak kedua oleh coach", "Mentalitas bertanding skuad garuda",
    "Regenerasi pemain muda di bawah asuhan pelatih", "Program latihan jangka panjang pelatih",
    "Organisasi pertahanan timnas racikan pelatih", "Transisi serangan balik racikan coach",
    "Komposisi formasi pemain timnas", "Performa timnas di kualifikasi piala dunia", "Kerja keras staf kepelatihan",
    "Pola pressing tinggi yang diterapkan pelatih", "Karakter pantang menyerah timnas racikan coach",
    "Sentuhan taktis pelatih kepala", "Visi bermain timnas di tangan pelatih saat ini"
]

pos_predicates = [
    "sangat luar biasa dan membawa perubahan positif nyata", "terbukti berhasil menaikkan ranking FIFA timnas secara drastis",
    "patut diapresiasi setinggi-tingginya oleh seluruh pecinta sepak bola", "membuat permainan timnas semakin matang dan berkelas dunia",
    "sangat efektif membongkar pertahanan lawan yang rapat", "sukses membentuk mental juang pantang menyerah sampai menit akhir",
    "menunjukkan progres dan kemajuan yang sangat pesat", "wajib kita dukung dan pertahankan kontraknya demi masa depan timnas",
    "sangat memuaskan dan membuat bangga seluruh rakyat Indonesia", "berhasil memotong generasi dan mematangkan pemain muda potensial",
    "menjadikan fisik dan determinasi pemain jauh lebih spartan", "menghadirkan pola permainan modern yang atraktif dan disiplin",
    "adalah bukti nyata kesuksesan proses pembinaan tim nasional", "memberikan harapan besar untuk timnas berprestasi di tingkat asia",
    "membuat koordinasi antarlini menjadi sangat rapi dan solid", "sukses menanamkan rasa percaya diri tinggi pada seluruh pemain"
]

pos_endings = [
    "Menyala garudaku!", "Respect coach!", "Kawal terus proses ini!", "Terima kasih banyak coach atas dedikasinya!",
    "Percaya proses!", "Garuda mendunia!", "Jangan pernah diganti!", "Top markotop!", "Maju terus sepak bola Indonesia!",
    "Luar biasa bangga!", "Gacor abis!", "Pertahankan sampai piala dunia!", "Keren banget pokoknya!", "Salut coach!"
]

# Variasi Tambahan Negatif (Kritik, Evaluasi, Ganti Pelatih)
neg_subjects = [
    "Formasi dan taktik pelatih kepala", "Strategi bertahan pelatih timnas", "Keputusan pergantian pemain oleh pelatih",
    "Gaya melatih coach yang monoton", "Performa timnas di bawah asuhan pelatih sekarang",
    "Pola penyerangan timnas yang tidak jelas", "Eksperimen posisi pemain oleh pelatih",
    "Hasil pertandingan timnas malam ini", "Fisik pemain yang kedodoran di babak kedua",
    "Komunikasi staf kepelatihan dengan pemain", "Skema bola mati pertahanan timnas",
    "Kebijakan pemilihan starting line up oleh pelatih"
]

neg_predicates = [
    "sangat mengecewakan dan minim sekali variasi serangan", "menjadi penyebab utama kekalahan memalukan timnas di laga krusial",
    "membuktikan bahwa sudah saatnya federasi melakukan evaluasi pergantian pelatih", "sangat pasif dan mudah sekali terbaca oleh taktik lawan",
    "terlalu keras kepala dan tidak mau mendengarkan kritik evaluasi", "membuat lini pertahanan timnas sangat rapuh dan sering blunder",
    "tidak menunjukkan perkembangan apapun sejak awal menangani timnas", "hanya merusak ritme dan kekompakan permainan skuad garuda",
    "gagal total memaksimalkan potensi pemain berbakat yang ada", "sudah habis masa keemasannya dan sebaiknya mundur secara terhormat",
    "sangat membingungkan dan berujung kekalahan fatal", "tidak memberikan solusi taktikal apapun saat tim dalam kondisi tertinggal"
]

neg_endings = [
    "Waktunya pelatih out!", "Evaluasi total sekarang juga!", "Sangat kecewa!", "Segera cari pelatih baru!",
    "Jangan ditunda lagi pergantiannya!", "Federasi harus bertindak tegas!", "Bapuk banget taktiknya!",
    "Mengecewakan sekali!", "Mundur saja demi kebaikan timnas!"
]

# Variasi Tambahan Netral (Berita, Jadwal, Konferensi Pers)
net_templates = [
    "PSSI menjadwalkan rapat evaluasi berkala bersama pelatih timnas setelah agenda {turnamen}.",
    "Pelatih timnas memanggil sebanyak {jumlah} pemain untuk mengikuti pemusatan latihan di {lokasi}.",
    "Konferensi pers pra pertandingan dihadiri langsung oleh pelatih kepala dan kapten timnas {negara}.",
    "Federasi sepak bola nasional merilis pernyataan resmi mengenai status kontrak kerja staf pelatih {tahun}.",
    "Pertandingan uji coba internasional antara timnas Indonesia melawan tim lawan berakhir dengan skor {skor}.",
    "Pelatih timnas memantau langsung jalannya pertandingan kompetisi domestik di stadion {stadion}.",
    "Statistik penguasaan bola dan operan timnas tercatat sebesar {persen} persen menurut data analis pertandingan.",
    "Jadwal sesi latihan perdana timnas dipimpin langsung oleh jajaran asisten pelatih di lapangan {lokasi}.",
    "Daftar susunan pemain starting eleven resmi diumumkan oleh tim media kepelatihan satu jam jelang laga."
]

# Generate additional tweets to reach exactly 1000
additional_tweets = []
random.seed(42)

# Tambahkan ~500 positif agar dominan positif
used_pos = set()
while len(additional_tweets) < 460:
    s = random.choice(pos_subjects)
    p = random.choice(pos_predicates)
    e = random.choice(pos_endings)
    conn = random.choice(["yang", "ini", "bener-bener", "saat ini", "memang"])
    t = f"{s} {conn} {p}. {e}"
    if t not in used_pos:
        used_pos.add(t)
        additional_tweets.append(t)

# Tambahkan ~70 negatif
used_neg = set()
while len(additional_tweets) < 530:
    s = random.choice(neg_subjects)
    p = random.choice(neg_predicates)
    e = random.choice(neg_endings)
    conn = random.choice(["malam ini", "saat ini", "terlihat", "sangat", "memang"])
    t = f"{s} {conn} {p}. {e}"
    if t not in used_neg:
        used_neg.add(t)
        additional_tweets.append(t)

# Tambahkan ~50 netral tambahan
turnamen_list = ["Piala Asia", "Kualifikasi Piala Dunia", "Piala AFF", "FIFA Matchday"]
lokasi_list = ["Jakarta", "Surabaya", "Bali", "Solo", "Bandung"]
stadion_list = ["Gelora Bung Karno", "Manahan", "Gelora Bung Tomo", "Pakansari"]

while len(orig_tweets) + len(additional_tweets) < 1000:
    template = random.choice(net_templates)
    t = template.format(
        turnamen=random.choice(turnamen_list),
        jumlah=random.choice([23, 26, 28]),
        lokasi=random.choice(lokasi_list),
        negara="Indonesia",
        tahun=random.choice([2024, 2025, 2026]),
        skor=random.choice(["0-0", "1-1", "2-2", "1-0"]),
        stadion=random.choice(stadion_list),
        persen=random.choice([52, 58, 61, 48])
    )
    additional_tweets.append(t)

# Gabung dan acak urutannya
all_1000_tweets = orig_tweets + additional_tweets
random.shuffle(all_1000_tweets)

df_final = pd.DataFrame({"tweet": all_1000_tweets[:1000]})
df_final.to_csv(raw_file, index=False, encoding="utf-8")

print("=" * 60)
print(f"🎉 SUKSES! File '{raw_file}' sekarang berisi {len(df_final)} TWEET!")
print("=" * 60)
