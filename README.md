# ⚽ Sistem Analisis Sentimen Pergantian Pelatih Timnas Sepak Bola Indonesia (LSTM)

Aplikasi Web Interaktif berbasis **Streamlit** untuk analisis sentimen opini publik di media sosial X (Twitter) terkait dinamika kepemimpinan dan isu pergantian pelatih Tim Nasional Sepak Bola Indonesia menggunakan metode **Long Short-Term Memory (LSTM)**.

Repository GitHub: [https://github.com/faisaldzkr04/lstm_app](https://github.com/faisaldzkr04/lstm_app)

---

## 🎯 Fitur Utama Aplikasi

1. **📊 Dashboard & Eksplorasi Data:**
   - Visualisasi distribusi kelas sentimen (Positif, Netral, Negatif) menggunakan *Pie Chart* dan *Bar Chart*.
   - Analisis frekuensi kata kunci teratas (*Top Keywords*) untuk setiap kategori sentimen.
   - Fitur upload dataset mandiri (`.csv`) atau menggunakan dataset bawaan (1000 tweet).
   - Tabel korpus data interaktif dengan fitur filter kategori sentimen.

2. **⚙️ Pipeline Preprocessing & Training LSTM:**
   - Preprocessing 5 tahap: *Cleaning (URL, Mention, Simbol, Angka)* $\rightarrow$ *Case Folding* $\rightarrow$ *Slang Words Normalization* $\rightarrow$ *Stopwords Removal* $\rightarrow$ *Tokenization & Padding*.
   - Hyperparameter Tuning interaktif: Konfigurasi *Epochs, Batch Size, LSTM Units, Embedding Dimension, Max Sequence Length*, dan *Vocab Size*.

3. **🔮 Prediksi Sentimen Teks (Single & Batch):**
   - Prediksi teks instan (*Real-time Single Tweet Prediction*) dengan probabilitas *Softmax*.
   - Prediksi massal (*Batch CSV Upload*) dilengkapi fitur unduh/ekspor hasil prediksi CSV.

4. **📈 Evaluasi Kinerja Model:**
   - Visualisasi *Confusion Matrix* 3x3.
   - Metrik evaluasi komprehensif: Akurasi, Presisi, Recall, dan F1-Score (Tabel *Classification Report*).
   - Grafik riwayat pelatihan (*Training & Validation Accuracy/Loss Curve*).

5. **📚 Dokumentasi Arsitektur & Teori:**
   - Rincian layer arsitektur: *Embedding $\rightarrow$ SpatialDropout $\rightarrow$ LSTM $\rightarrow$ Dense ReLU $\rightarrow$ Dense Softmax*.
   - Formulasi matematis gerbang sel LSTM (*Forget Gate, Input Gate, Candidate State, Cell State Update, Output Gate*).

---

## 📁 Struktur Berkas

```
lstm_app/
├── app.py                              # Aplikasi utama Streamlit
├── lstm_model.py                       # Arsitektur & Logika Klasifikasi LSTM / Lexicon Fallback
├── utils.py                            # Fungsi Preprocessing, Slang Normalization & Stopwords
├── generate_1000_dataset.py            # Generator dataset sintetis 1000 tweet
├── expand_user_file.py                 # Helper ekspansi dataset user
├── merge_and_expand.py                 # Helper gabung & perluas dataset
├── format_data.py                      # Helper format data
├── scrape_x_auto.py                    # Helper scraping tweet
├── dataset_timnas_siap_upload_1000.csv# Dataset default 1000 tweet
├── requirements.txt                    # Dependensi pustaka Python
├── install.bat                         # Batch script instalasi Windows
├── run.bat                             # Batch script jalankan aplikasi Windows
├── push_to_github.bat                  # Batch script otomatis push ke GitHub
└── README.md                           # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Prasyarat & Instalasi
Pastikan Python 3.8 - 3.12 telah terinstall.

```bash
# Clone repository
git clone https://github.com/faisaldzkr04/lstm_app.git
cd lstm_app

# Instalasi dependensi
pip install -r requirements.txt
```

Atau di Windows cukup klik dua kali file **`install.bat`**.

### 2. Jalankan Aplikasi
```bash
streamlit run app.py
```

Atau di Windows cukup klik dua kali file **`run.bat`**.
Buka browser Anda di `http://localhost:8501`.

---

## 📤 Langkah Push ke Repository Git

```bash
git init
git remote add origin https://github.com/faisaldzkr04/lstm_app.git
git branch -M main
git add .
git commit -m "Initial commit: Aplikasi Analisis Sentimen LSTM Timnas Indonesia"
git push -u origin main
```

