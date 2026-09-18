import re
import string
import random
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# Standar Stopwords Bahasa Indonesia Sederhana
INDONESIAN_STOPWORDS = set([
    "yang", "di", "dan", "dari", "ini", "itu", "untuk", "pada", "ke", "karena",
    "dengan", "adalah", "akan", "juga", "bisa", "ada", "saya", "kamu", "dia",
    "mereka", "kita", "kami", "oleh", "saat", "setelah", "sebelum", "dalam",
    "sudah", "belum", "hanya", "lagi", "pun", "saja", "kalau", "jika", "atau",
    "maka", "tentang", "seperti", "bagi", "sampai", "terhadap", "secara", "lebih"
])

# Kamus Normalisasi Slang Words Bahasa Indonesia
SLANG_DICTIONARY = {
    "yg": "yang",
    "bgt": "banget",
    "bgt!": "banget",
    "bgtt": "banget",
    "skrg": "sekarang",
    "tdk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "g": "tidak",
    "bener": "benar",
    "bener2": "benar-benar",
    "emg": "memang",
    "jg": "juga",
    "dgn": "dengan",
    "kpd": "kepada",
    "utk": "untuk",
    "bapuk": "buruk",
    "ancur": "hancur",
    "mantep": "mantap",
    "mantul": "mantap",
    "baper": "bawa perasaan",
    "coach": "pelatih",
    "tactics": "taktik",
    "stamina": "fisik",
    "out": "keluar",
    "in": "masuk",
    "gonta": "ganti",
    "nyelesaiin": "menyelesaikan",
    "sty": "pelatih timnas",
    "shintaeyong": "pelatih timnas"
}

def clean_text(text: str) -> str:
    """Membersihkan teks dari URL, mention, hashtag, karakter khusus, dan angka."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_slang(text: str) -> str:
    """Mengganti kata gaul/singkatan dengan kata baku."""
    words = text.split()
    normalized_words = [SLANG_DICTIONARY.get(w.lower(), w) for w in words]
    return " ".join(normalized_words)

def remove_stopwords(text: str) -> str:
    """Menghapus stopwords bahasa Indonesia."""
    words = text.split()
    filtered = [w for w in words if w.lower() not in INDONESIAN_STOPWORDS]
    return " ".join(filtered)

def preprocess_pipeline(text: str) -> str:
    """Menjalankan seluruh tahapan preprocessing teks."""
    text = clean_text(text)
    text = text.lower()
    text = normalize_slang(text)
    text = remove_stopwords(text)
    return text

# Komponen Pembentuk 1000 Tweet
_pos_subjects = [
    "Taktik pelatih timnas", "Strategi kepelatihan coach", "Kedisiplinan pemain era pelatih sekarang",
    "Gaya permainan timnas saat ini", "Filosofi sepak bola pelatih", "Kepemimpinan pelatih kepala",
    "Fisik dan stamina timnas", "Pergantian pemain babak kedua oleh coach", "Mentalitas bertanding skuad garuda",
    "Regenerasi pemain muda di bawah asuhan pelatih", "Program latihan jangka panjang pelatih",
    "Organisasi pertahanan timnas racikan pelatih", "Transisi serangan balik racikan coach",
    "Komposisi formasi pemain timnas", "Performa timnas di kualifikasi piala dunia", "Kerja keras staf kepelatihan"
]

_pos_predicates = [
    "sangat luar biasa dan membawa perubahan positif nyata", "terbukti berhasil menaikkan ranking FIFA timnas secara drastis",
    "patut diapresiasi setinggi-tingginya oleh seluruh pecinta sepak bola", "membuat permainan timnas semakin matang dan berkelas dunia",
    "sangat efektif membongkar pertahanan lawan yang rapat", "sukses membentuk mental juang pantang menyerah sampai menit akhir",
    "menunjukkan progres dan kemajuan yang sangat pesat", "wajib kita dukung dan pertahankan kontraknya demi masa depan timnas",
    "sangat memuaskan dan membuat bangga seluruh rakyat Indonesia", "berhasil memotong generasi dan mematangkan pemain muda potensial",
    "menjadikan fisik dan determinasi pemain jauh lebih spartan", "menghadirkan pola permainan modern yang atraktif dan disiplin",
    "adalah bukti nyata kesuksesan proses pembinaan tim nasional", "memberikan harapan besar untuk timnas berprestasi di tingkat asia"
]

_pos_endings = [
    "Menyala garudaku!", "Respect coach!", "Kawal terus proses ini!", "Terima kasih banyak coach atas dedikasinya!",
    "Percaya proses!", "Garuda mendunia!", "Jangan pernah diganti!", "Top markotop!", "Maju terus sepak bola Indonesia!",
    "Luar biasa bangga!", "Gacor abis!", "Pertahankan sampai piala dunia!"
]

_neg_subjects = [
    "Formasi dan taktik pelatih kepala", "Strategi bertahan pelatih timnas", "Keputusan pergantian pemain oleh pelatih",
    "Gaya melatih coach yang monoton", "Performa timnas di bawah asuhan pelatih sekarang",
    "Pola penyerangan timnas yang tidak jelas", "Eksperimen posisi pemain oleh pelatih",
    "Hasil pertandingan timnas malam ini", "Fisik pemain yang kedodoran di babak kedua",
    "Komunikasi staf kepelatihan dengan pemain", "Skema bola mati pertahanan timnas",
    "Kebijakan pemilihan starting line up oleh pelatih"
]

_neg_predicates = [
    "sangat mengecewakan dan minim sekali variasi serangan", "menjadi penyebab utama kekalahan memalukan timnas di laga krusial",
    "membuktikan bahwa sudah saatnya federasi melakukan evaluasi pergantian pelatih", "sangat pasif dan mudah sekali terbaca oleh taktik lawan",
    "terlalu keras kepala dan tidak mau mendengarkan kritik evaluasi", "membuat lini pertahanan timnas sangat rapuh dan sering blunder",
    "tidak menunjukkan perkembangan apapun sejak awal menangani timnas", "hanya merusak ritme dan kekompakan permainan skuad garuda",
    "gagal total memaksimalkan potensi pemain berbakat yang ada", "sudah habis masa keemasannya dan sebaiknya mundur secara terhormat",
    "sangat membingungkan dan berujung kekalahan fatal", "tidak memberikan solusi taktikal apapun saat tim dalam kondisi tertinggal"
]

_neg_endings = [
    "Waktunya pelatih out!", "Evaluasi total sekarang juga!", "Sangat kecewa!", "Segera cari pelatih baru!",
    "Jangan ditunda lagi pergantiannya!", "Federasi harus bertindak tegas!", "Bapuk banget taktiknya!",
    "Mengecewakan sekali!", "Mundur saja demi kebaikan timnas!"
]

_net_templates = [
    "PSSI menjadwalkan rapat evaluasi berkala bersama pelatih timnas setelah agenda {turnamen}.",
    "Pelatih timnas memanggil sebanyak {jumlah} pemain untuk mengikuti pemusatan latihan di {lokasi}.",
    "Konferensi pers pra pertandingan dihadiri langsung oleh pelatih kepala dan kapten timnas {negara}.",
    "Federasi sepak bola nasional merilis pernyataan resmi mengenai status kontrak kerja staf pelatih {tahun}.",
    "Pertandingan uji coba internasional antara timnas Indonesia melawan tim lawan berakhir dengan skor {skor}.",
    "Pelatih timnas memantau langsung jalannya pertandingan kompetisi domestik di stadion {stadion}.",
    "Statistik penguasaan bola dan operan timnas tercatat sebesar {persen} persen menurut data analis pertandingan.",
    "Jadwal sesi latihan perdana timnas dipimpin langsung oleh jajaran asisten pelatih di lapangan {lokasi}.",
    "Daftar susunan pemain starting eleven resmi diumumkan oleh tim media kepelatihan satu jam jelang laga.",
    "Media olahraga memberitakan rumor bursa calon pelatih timnas jelang kualifikasi putaran {putaran}."
]

def generate_positive_tweets(count: int) -> List[str]:
    """Menghasilkan list tweet opini positif yang variatif dan acak."""
    pos_tweets = []
    connectors = ["yang", "ini", "bener-bener", "saat ini", "memang", "selalu", "sudah", "sungguh", "pastinya"]
    for _ in range(count):
        s = random.choice(_pos_subjects)
        p = random.choice(_pos_predicates)
        e = random.choice(_pos_endings)
        c = random.choice(connectors)
        pos_tweets.append(f"{s} {c} {p}. {e}")
    return pos_tweets

def generate_neutral_tweets(count: int) -> List[str]:
    """Menghasilkan list tweet netral/berita faktual yang variatif dan acak."""
    net_tweets = []
    turnamen_list = ["Piala Asia", "Kualifikasi Piala Dunia", "Piala AFF", "FIFA Matchday", "SEA Games", "Piala Dunia U-20"]
    lokasi_list = ["Jakarta", "Surabaya", "Bali", "Solo", "Bandung", "Doha", "Riyadh", "Kuala Lumpur"]
    stadion_list = ["Gelora Bung Karno", "Manahan", "Gelora Bung Tomo", "Pakansari", "Jalak Harupat", "Patriot"]
    for _ in range(count):
        tmpl = random.choice(_net_templates)
        net_tweets.append(tmpl.format(
            turnamen=random.choice(turnamen_list),
            jumlah=random.choice([23, 26, 28, 30, 24]),
            lokasi=random.choice(lokasi_list),
            negara="Indonesia",
            tahun=random.choice([2024, 2025, 2026]),
            skor=random.choice(["0-0", "1-1", "2-2", "1-0", "2-1", "0-1"]),
            stadion=random.choice(stadion_list),
            persen=random.choice([52, 58, 61, 48, 55, 64]),
            putaran=random.choice([2, 3, 4])
        ))
    return net_tweets

def generate_negative_tweets(count: int) -> List[str]:
    """Menghasilkan list tweet opini kritik/negatif yang variatif dan acak."""
    neg_tweets = []
    connectors = ["malam ini", "saat ini", "terlihat", "sangat", "memang", "kemarin", "justru"]
    for _ in range(count):
        s = random.choice(_neg_subjects)
        p = random.choice(_neg_predicates)
        e = random.choice(_neg_endings)
        c = random.choice(connectors)
        neg_tweets.append(f"{s} {c} {p}. {e}")
    return neg_tweets

def balance_and_expand_dataset(raw_tweets: List[str], raw_labels: List[str], target_total: int = 1000, target_pos: int = 450, target_net: int = 280, target_neg: int = 270) -> pd.DataFrame:
    """
    Menggenapkan dan menyeimbangkan dataset menjadi tepat 1.000 tweet
    dengan komposisi ideal: Positif terbanyak (45%), Netral (28%), Negatif (27%).
    """
    random.seed(42)
    np.random.seed(42)

    cur_pos = sum(1 for l in raw_labels if l == "Positif")
    cur_net = sum(1 for l in raw_labels if l == "Netral")
    cur_neg = sum(1 for l in raw_labels if l == "Negatif")

    needed_pos = max(0, target_pos - cur_pos)
    needed_net = max(0, target_net - cur_net)
    needed_neg = max(0, target_neg - cur_neg)

    extra_tweets = []
    extra_labels = []

    if needed_pos > 0:
        extra_tweets.extend(generate_positive_tweets(needed_pos))
        extra_labels.extend(["Positif"] * needed_pos)

    if needed_net > 0:
        extra_tweets.extend(generate_neutral_tweets(needed_net))
        extra_labels.extend(["Netral"] * needed_net)

    if needed_neg > 0:
        extra_tweets.extend(generate_negative_tweets(needed_neg))
        extra_labels.extend(["Negatif"] * needed_neg)

    combined_tweets = raw_tweets + extra_tweets
    combined_labels = raw_labels + extra_labels

    combined = list(zip(combined_tweets, combined_labels))
    random.shuffle(combined)
    combined = combined[:target_total]

    shuff_t, shuff_l = zip(*combined)
    return pd.DataFrame({
        "tweet": list(shuff_t),
        "sentimen": list(shuff_l)
    })

def create_sample_dataset(target_total=1000, target_pos_ratio=0.45, target_net_ratio=0.28, target_neg_ratio=0.27) -> pd.DataFrame:
    """
    Membuat dataset korpus 1000 tweet opini pergantian pelatih timnas
    yang proporsional: Positif terbanyak (45%), Netral (28%), dan Negatif (27%).
    """
    random.seed(42)
    np.random.seed(42)

    n_pos = int(target_total * target_pos_ratio)
    n_net = int(target_total * target_net_ratio)
    n_neg = target_total - n_pos - n_net

    tweets = []
    labels = []

    tweets.extend(generate_positive_tweets(n_pos))
    labels.extend(["Positif"] * n_pos)

    tweets.extend(generate_neutral_tweets(n_net))
    labels.extend(["Netral"] * n_net)

    tweets.extend(generate_negative_tweets(n_neg))
    labels.extend(["Negatif"] * n_neg)

    combined = list(zip(tweets, labels))
    random.shuffle(combined)
    shuffled_tweets, shuffled_labels = zip(*combined)

    return pd.DataFrame({
        "tweet": list(shuffled_tweets),
        "sentimen": list(shuffled_labels)
    })
