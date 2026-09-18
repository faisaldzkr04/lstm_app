"""
Script Generator Dataset 1000 Tweet Otomatis & Variatif
Judul Skripsi: Analisis Sentimen Pada Media Sosial X Terhadap Pergantian Pelatih Tim Nasional Sepak Bola Indonesia Menggunakan Metode LSTM
"""

import pandas as pd
import numpy as np
import random

# Komponen Variasi Kalimat Positif
pos_subjects = [
    "Taktik pelatih timnas", "Strategi kepelatihan coach", "Kedisiplinan pemain era pelatih sekarang",
    "Gaya permainan timnas saat ini", "Filosofi sepak bola pelatih", "Kepemimpinan pelatih kepala",
    "Fisik dan stamina timnas", "Pergantian pemain babak kedua oleh coach", "Mentalitas bertanding skuad garuda",
    "Regenerasi pemain muda di bawah asuhan pelatih", "Program latihan jangka panjang pelatih",
    "Organisasi pertahanan timnas racikan pelatih", "Transisi serangan balik racikan coach",
    "Komposisi formasi pemain timnas", "Performa timnas di kualifikasi piala dunia", "Kerja keras staf kepelatihan"
]

pos_predicates = [
    "sangat luar biasa dan membawa perubahan positif nyata", "terbukti berhasil menaikkan ranking FIFA timnas secara drastis",
    "patut diapresiasi setinggi-tingginya oleh seluruh pecinta sepak bola", "membuat permainan timnas semakin matang dan berkelas dunia",
    "sangat efektif membongkar pertahanan lawan yang rapat", "sukses membentuk mental juang pantang menyerah sampai menit akhir",
    "menunjukkan progres dan kemajuan yang sangat pesat", "wajib kita dukung dan pertahankan kontraknya demi masa depan timnas",
    "sangat memuaskan dan membuat bangga seluruh rakyat Indonesia", "berhasil memotong generasi dan mematangkan pemain muda potensial",
    "menjadikan fisik dan determinasi pemain jauh lebih spartan", "menghadirkan pola permainan modern yang atraktif dan disiplin",
    "adalah bukti nyata kesuksesan proses pembinaan tim nasional", "memberikan harapan besar untuk timnas berprestasi di tingkat asia"
]

pos_endings = [
    "Menyala garudaku!", "Respect coach!", "Kawal terus proses ini!", "Terima kasih banyak coach atas dedikasinya!",
    "Percaya proses!", "Garuda mendunia!", "Jangan pernah diganti!", "Top markotop!", "Maju terus sepak bola Indonesia!",
    "Luar biasa bangga!", "Gacor abis!", "Pertahankan sampai piala dunia!"
]

# Komponen Variasi Kalimat Negatif
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

# Komponen Variasi Kalimat Netral
net_templates = [
    "PSSI menjadwalkan rapat evaluasi berkala bersama pelatih timnas setelah agenda {turnamen}.",
    "Pelatih timnas memanggil sebanyak {jumlah} pemain untuk mengikuti pemusatan latihan di {lokasi}.",
    "Konferensi pers pra pertandingan dihadiri langsung oleh pelatih kepala dan kapten timnas {negara}.",
    "Federasi sepak bola nasional merilis pernyataan resmi mengenai status kontrak kerja staf pelatih {tahun}.",
    "Pertandingan uji coba internasional antara timnas Indonesia melawan tim lawan berakhir dengan skor {skor}.",
    "Pelatih timnas memantau langsung jalannya pertandingan kompetisi domestik di stadion {stadion}.",
    "Statistik penguasaan bola dan operan timnas tercatat sebesar {persen} persen menurut data analis pertandingan.",
    "Jadwal sesi latihan perdana timnas dipimpin langsung oleh jajaran asisten pelatih di lapangan {lokasi}.",
    "Daftar susunan pemain starting eleven resmi diumumkan oleh tim media kepelatihan satu jam jelang laga.",
    "Media olahraga memberitakan rumor bursa calon pelatih timnas jelang kualifikasi putaran {putaran}.",
    "Pelatih memberikan keterangan statistik performa fisik pemain dalam sesi tanya jawab bersama wartawan.",
    "Dokumen klausul kesepakatan target kerja sama antara PSSI dan pelatih kepala ditandatangani di kantor federasi."
]

def generate_1000_tweets(target_total=1000, target_pos_ratio=0.60, target_net_ratio=0.25, target_neg_ratio=0.15):
    """
    Menghasilkan 1000 tweet realistis bertema pergantian pelatih timnas
    dengan proporsi representatif untuk skripsi.
    """
    random.seed(42)
    np.random.seed(42)

    n_pos = int(target_total * target_pos_ratio)
    n_net = int(target_total * target_net_ratio)
    n_neg = target_total - n_pos - n_net

    tweets = []
    labels = []

    # 1. Generate Positif
    used_pos = set()
    while len(tweets) < n_pos:
        s = random.choice(pos_subjects)
        p = random.choice(pos_predicates)
        e = random.choice(pos_endings)
        connector = random.choice(["yang", "ini", "bener-bener", "saat ini", "memang"])
        tweet = f"{s} {connector} {p}. {e}"
        if tweet not in used_pos:
            used_pos.add(tweet)
            tweets.append(tweet)
            labels.append("Positif")

    # 2. Generate Netral
    used_net = set()
    turnamen_list = ["Piala Asia", "Kualifikasi Piala Dunia", "Piala AFF", "FIFA Matchday", "SEA Games"]
    lokasi_list = ["Jakarta", "Surabaya", "Bali", "Solo", "Bandung", "Doha", "Riyadh"]
    stadion_list = ["Gelora Bung Karno", "Manahan", "Gelora Bung Tomo", "Pakansari", "Jalak Harupat"]
    
    while len(tweets) < (n_pos + n_net):
        template = random.choice(net_templates)
        tweet = template.format(
            turnamen=random.choice(turnamen_list),
            jumlah=random.choice([23, 26, 28, 30]),
            lokasi=random.choice(lokasi_list),
            negara="Indonesia",
            tahun=random.choice([2024, 2025, 2026]),
            skor=random.choice(["0-0", "1-1", "2-2", "1-0", "2-1"]),
            stadion=random.choice(stadion_list),
            persen=random.choice([52, 58, 61, 48, 55]),
            putaran=random.choice([2, 3, 4])
        )
        if tweet not in used_net:
            used_net.add(tweet)
            tweets.append(tweet)
            labels.append("Netral")

    # 3. Generate Negatif
    used_neg = set()
    while len(tweets) < target_total:
        s = random.choice(neg_subjects)
        p = random.choice(neg_predicates)
        e = random.choice(neg_endings)
        connector = random.choice(["malam ini", "saat ini", "terlihat", "sangat", "memang"])
        tweet = f"{s} {connector} {p}. {e}"
        if tweet not in used_neg:
            used_neg.add(tweet)
            tweets.append(tweet)
            labels.append("Negatif")

    # Acak urutan tweet agar tidak terkelompok rapi
    combined = list(zip(tweets, labels))
    random.shuffle(combined)
    shuffled_tweets, shuffled_labels = zip(*combined)

    df = pd.DataFrame({
        "tweet": shuffled_tweets,
        "sentimen": shuffled_labels
    })

    return df

if __name__ == "__main__":
    df_1000 = generate_1000_tweets(1000)
    output_path = "dataset_timnas_1000_tweets.csv"
    df_1000.to_csv(output_path, index=False, encoding="utf-8")
    
    print("=" * 60)
    print(f"✅ Berhasil membuat {len(df_1000)} tweet representatif!")
    print(f"📁 Disimpan di: {output_path}")
    print("=" * 60)
    print("Sebaran Sentimen:")
    print(df_1000["sentimen"].value_counts())
    print("=" * 60)
