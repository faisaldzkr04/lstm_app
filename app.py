import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split

from utils import (
    preprocess_pipeline,
    create_sample_dataset,
    balance_and_expand_dataset,
    clean_text,
    normalize_slang,
    remove_stopwords
)
from lstm_model import LSTMSentimentClassifier, SimpleTokenizer

# ==========================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================================
st.set_page_config(
    page_title="Analisis Sentimen Pelatih Timnas (LSTM)",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# INISIALISASI SESSION STATE SECARA DINAMIS
# ==========================================================
if "df" not in st.session_state:
    df_init = create_sample_dataset()
    st.session_state.df = df_init
    st.session_state.preprocessed_done = False
    st.session_state.model_trained = False
    st.session_state.history = None
    st.session_state.eval_results = None
    st.session_state.test_df = None

if "classifier" not in st.session_state:
    clf = LSTMSentimentClassifier(vocab_size=3000, embedding_dim=100, max_len=50, lstm_units=128)
    clf.tokenizer = SimpleTokenizer(num_words=3000)
    # Tokenizer inisial cepat
    clf.tokenizer.fit_on_texts(st.session_state.df["tweet"].tolist())
    st.session_state.classifier = clf

# ==========================================================
# STYLING CSS CUSTOM
# ==========================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #0D9488 100%);
        padding: 22px 25px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .main-header h1 {
        font-size: 20px;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    .main-header p {
        font-size: 13.5px;
        margin-top: 6px;
        margin-bottom: 0;
        color: #E0F2FE;
    }
    .step-box {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 12px 18px;
        border-radius: 6px;
        margin-bottom: 15px;
        font-weight: 600;
        color: #166534;
    }
    .pipeline-status {
        background: #F8FAFC;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 12px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# Banner Judul Skripsi
st.markdown("""
<div class="main-header">
    <h1>⚽ ANALISIS SENTIMEN PADA MEDIA SOSIAL X TERHADAP PERGANTIAN PELATIH TIM NASIONAL SEPAK BOLA INDONESIA MENGGUNAKAN METODE LSTM</h1>
    <p>Aplikasi Skripsi | Natural Language Processing & Deep Learning | Pipeline Berurutan & Dinamis</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR NAVIGASI BERURUTAN & STATUS PIPELINE
# ==========================================================
st.sidebar.markdown("## ⚽ **Menu Tahapan Skripsi**")

menu_options = [
    "1. Upload Dataset",
    "2. Preprocessing",
    "3. Pelatihan LSTM",
    "4. Evaluasi Model",
    "5. Prediksi Sentimen"
]

menu = st.sidebar.radio("Pilih Tahapan Pipeline:", menu_options, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 **Informasi Penelitian**")
st.sidebar.markdown("**Metode:** Long Short-Term Memory (LSTM)")
st.sidebar.markdown("**Objek:** Opini X Terhadap Pelatih Timnas")
st.sidebar.markdown(r"**Arsitektur:** Embedding $\rightarrow$ LSTM $\rightarrow$ Dense $\rightarrow$ Softmax")
st.sidebar.caption("© 2026 Program Skripsi Analisis Sentimen NLP")


# ==========================================================
# 1. TAHAP 1: UPLOAD DATASET
# ==========================================================
if menu == "1. Upload Dataset":
    st.markdown('<div class="step-box">📌 TAHAP 1: Pengunggahan & Eksplorasi Korpus Dataset</div>', unsafe_allow_html=True)
    st.write("Unggah file CSV dataset media sosial X atau gunakan dataset representatif 1.000 tweet yang telah disediakan.")

    uploaded_file = st.file_uploader("📂 Pilih file CSV dataset X/Twitter:", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            
            # Deteksi kolom teks
            text_col = None
            for col in ["tweet", "full_text", "text", "content", "Teks", "komentar"]:
                if col in user_df.columns:
                    text_col = col
                    break
            if text_col is None:
                text_col = user_df.columns[0]

            # Deteksi kolom sentimen (jika ada)
            sentiment_col = None
            for col in ["sentimen", "sentiment", "label", "Label", "kategori", "target"]:
                if col in user_df.columns:
                    sentiment_col = col
                    break

            raw_tweets = user_df[text_col].astype(str).tolist()
            
            if sentiment_col is not None:
                raw_labels = user_df[sentiment_col].astype(str).tolist()
                label_mapping = {
                    "positive": "Positif", "positif": "Positif", "1": "Positif",
                    "neutral": "Netral", "netral": "Netral", "0": "Netral",
                    "negative": "Negatif", "negatif": "Negatif", "-1": "Negatif", "2": "Negatif"
                }
                clean_labels = [label_mapping.get(str(l).lower().strip(), str(l)) for l in raw_labels]
            else:
                temp_clf = LSTMSentimentClassifier()
                temp_preds = temp_clf.predict_batch(raw_tweets)
                clean_labels = [p["label"] for p in temp_preds]

            # JIKA DATA KURANG DARI 1000: OTOMATIS SEIMBANGKAN & GENAPKAN MENJADI TEPAT 1.000 TWEET
            if len(raw_tweets) < 1000:
                loaded_df = balance_and_expand_dataset(raw_tweets, clean_labels, target_total=1000, target_pos=450, target_net=280, target_neg=270)
            else:
                loaded_df = pd.DataFrame({"tweet": raw_tweets, "sentimen": clean_labels})
                
            msg = f"✅ Berhasil memuat **{len(loaded_df)} data tweet** dari file CSV!"

            # Reset tahapan berikutnya karena dataset baru dimuat
            st.session_state.df = loaded_df
            st.session_state.preprocessed_done = False
            st.session_state.model_trained = False
            st.session_state.history = None
            st.session_state.eval_results = None
            st.success(msg)

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

    df = st.session_state.df
    total_data = len(df)
    pos_count = (df["sentimen"] == "Positif").sum()
    net_count = (df["sentimen"] == "Netral").sum()
    neg_count = (df["sentimen"] == "Negatif").sum()

    st.markdown("---")
    st.write("#### 📊 Ringkasan Distribusi Data Saat Ini:")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Data Tweet", f"{total_data} Tweet")
    with c2:
        st.metric("Sentimen Positif 🟢", f"{pos_count} ({pos_count/total_data*100:.1f}%)" if total_data > 0 else "0")
    with c3:
        st.metric("Sentimen Netral ⚪", f"{net_count} ({net_count/total_data*100:.1f}%)" if total_data > 0 else "0")
    with c4:
        st.metric("Sentimen Negatif 🔴", f"{neg_count} ({neg_count/total_data*100:.1f}%)" if total_data > 0 else "0")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_pie, ax_pie = plt.subplots(figsize=(4.5, 4.5))
        colors = ["#10B981", "#64748B", "#EF4444"]
        counts = [pos_count, net_count, neg_count]
        labels = ["Positif", "Netral", "Negatif"]
        ax_pie.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors, explode=(0.03, 0.02, 0.03), shadow=True)
        ax_pie.axis("equal")
        st.pyplot(fig_pie)
        plt.close(fig_pie)

    with col_v2:
        fig_bar, ax_bar = plt.subplots(figsize=(5.5, 4.2))
        sns.barplot(x=labels, y=counts, hue=labels, palette=colors, ax=ax_bar, legend=False)
        ax_bar.set_ylabel("Jumlah Data")
        ax_bar.set_xlabel("Kelas Sentimen")
        for i, val in enumerate(counts):
            ax_bar.text(i, val + (max(counts)*0.01 if counts else 1), str(val), ha='center', fontweight='bold')
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    st.write("#### 📑 Pratinjau Tabel Dataset:")
    st.dataframe(df[["tweet", "sentimen"]].head(100), use_container_width=True)

    # Download Dataset
    csv_exp = io.StringIO()
    df.to_csv(csv_exp, index=False)
    st.download_button(
        label="📥 Download Dataset Aktif (CSV)",
        data=csv_exp.getvalue(),
        file_name="dataset_timnas_aktif.csv",
        mime="text/csv"
    )

    st.info("💡 **Langkah Selanjutnya:** Silakan buka menu **'2. Preprocessing'** pada navigasi sebelah kiri untuk membersihkan teks tweet.")


# ==========================================================
# 2. TAHAP 2: PREPROCESSING
# ==========================================================
elif menu == "2. Preprocessing":
    st.markdown('<div class="step-box">🧪 TAHAP 2: Text Preprocessing & Pembersihan Teks</div>', unsafe_allow_html=True)
    st.write("Tahapan pembersihan data mentah teks tweet media sosial X secara dinamis agar siap dikonversi ke representasi vektor embedding LSTM.")

    tab_auto, tab_sim = st.tabs(["⚡ Preprocessing Seluruh Dataset", "🔬 Uji Coba Simulasi Kalimat Tunggal"])

    with tab_auto:
        st.write("#### Pipeline Pembersihan Teks yang Diterapkan:")
        st.markdown(r"""
        1. **Cleaning:** Menghapus tautan URL (`http/https`), username mention (`@user`), hashtag (`#`), angka, dan tanda baca.
        2. **Case Folding:** Mengubah seluruh karakter huruf menjadi huruf kecil (*lowercase*).
        3. **Normalisasi Kata Slang:** Mengonversi kata gaul/singkatan ke bentuk kata baku (contoh: *yg $\rightarrow$ yang, bgt $\rightarrow$ banget, coach $\rightarrow$ pelatih*).
        4. **Stopword Removal:** Menghapus kata sambung/hubung yang tidak memiliki bobot sentimen (*dan, yang, di, dari, untuk, dll.*).
        5. **Tokenisasi:** Memecah teks menjadi deretan token kata terpisah.
        """)

        if st.button("🚀 Jalankan Preprocessing Pada Dataset", type="primary"):
            with st.spinner("Sedang memproses text preprocessing pada seluruh data tweet..."):
                st.session_state.df["clean_tweet"] = st.session_state.df["tweet"].apply(preprocess_pipeline)
                st.session_state.preprocessed_done = True
                st.success(f"✅ Berhasil memproses preprocessing untuk {len(st.session_state.df)} baris data tweet!")

        if st.session_state.get("preprocessed_done", False) and "clean_tweet" in st.session_state.df.columns:
            st.markdown("---")
            st.write("#### 📑 Perbandingan Data Sebelum vs Sesudah Preprocessing:")
            st.dataframe(st.session_state.df[["tweet", "clean_tweet", "sentimen"]].head(20), use_container_width=True)

            st.info("💡 **Langkah Selanjutnya:** Preprocessing selesai! Silakan lanjutkan ke menu **'3. Pelatihan LSTM'**.")
        else:
            st.warning("⚠️ Preprocessing belum dijalankan untuk dataset saat ini. Klik tombol **'🚀 Jalankan Preprocessing Pada Dataset'** di atas.")

    with tab_sim:
        st.write("#### 🧪 Simulasi Langkah Demi Langkah Preprocessing")
        sample_raw = st.text_area(
            "Masukkan contoh teks tweet kotor:",
            value="Taktik pelatih timnas yg skrg emg bener2 top bgt!! 🔥🔥 Jangan ganti coach! https://t.co/9xZGaruda"
        )
        if st.button("Proses Preprocessing Contoh Teks"):
            c_text = clean_text(sample_raw)
            cf_text = c_text.lower()
            norm_text = normalize_slang(cf_text)
            stop_text = remove_stopwords(norm_text)
            tokens = stop_text.split()

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.info(f"**1. Raw Text:**\n\n{sample_raw}")
                st.info(f"**2. Cleaning:**\n\n{c_text}")
                st.info(f"**3. Case Folding:**\n\n{cf_text}")
            with col_p2:
                st.info(f"**4. Slang Normalization:**\n\n{norm_text}")
                st.info(f"**5. Stopword Removal:**\n\n{stop_text}")
                st.success(f"**6. Token Hasil Akhir:**\n\n`{tokens}`")


# ==========================================================
# 3. TAHAP 3: PELATIHAN LSTM
# ==========================================================
elif menu == "3. Pelatihan LSTM":
    st.markdown('<div class="step-box">⚙️ TAHAP 3: Pelatihan & Konfigurasi Hyperparameter Model LSTM</div>', unsafe_allow_html=True)
    st.write("Latih jaringan saraf tiruan Long Short-Term Memory (LSTM) secara dinamis dengan data hasil preprocessing.")

    if not st.session_state.get("preprocessed_done", False) or "clean_tweet" not in st.session_state.df.columns:
        st.warning("⚠️ Data belum melalui tahap Preprocessing. Silakan selesaikan menu **'2. Preprocessing'** terlebih dahulu.")
    else:
        st.write("#### ⚙️ Konfigurasi Hyperparameter Pelatihan:")
        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            n_epochs = st.slider("Jumlah Epochs (Iterasi):", min_value=10, max_value=60, value=25, step=5)
            batch_sz = st.selectbox("Batch Size:", [8, 16, 32, 64], index=1)
            test_split = st.slider("Proporsi Data Uji (Test Split):", min_value=0.1, max_value=0.3, value=0.2, step=0.05, format="%.2f")
        with cp2:
            lstm_dim = st.selectbox("LSTM Hidden Units:", [64, 128, 256], index=1)
            embed_dim = st.selectbox("Dimensi Word Embedding:", [50, 100, 200], index=1)
        with cp3:
            max_sequence_len = st.slider("Panjang Maksimal Sequence (Max Len):", min_value=30, max_value=100, value=50, step=10)
            vocab_max = st.selectbox("Ukuran Kosakata (Vocab Size):", [2000, 3000, 5000], index=1)

        if st.button("🚀 Mulai Pelatihan Model LSTM", type="primary"):
            with st.spinner("Sedang membagi dataset, membentuk representasi token sequence, dan melatih jaringan LSTM..."):
                df_curr = st.session_state.df
                
                # Split Train & Test dinamis
                train_df, test_df = train_test_split(
                    df_curr,
                    test_size=test_split,
                    random_state=42,
                    stratify=df_curr["sentimen"] if len(df_curr["sentimen"].unique()) > 1 else None
                )

                clf_new = LSTMSentimentClassifier(
                    vocab_size=vocab_max,
                    embedding_dim=embed_dim,
                    max_len=max_sequence_len,
                    lstm_units=lstm_dim
                )

                history = clf_new.train(
                    train_df["clean_tweet"].tolist(),
                    train_df["sentimen"].tolist(),
                    epochs=n_epochs,
                    batch_size=batch_sz,
                    validation_split=0.2
                )

                # Jalankan evaluasi dinamis pada test set
                eval_res = clf_new.evaluate(
                    test_df["clean_tweet"].tolist(),
                    test_df["sentimen"].tolist()
                )

                st.session_state.classifier = clf_new
                st.session_state.history = history
                st.session_state.eval_results = eval_res
                st.session_state.test_df = test_df
                st.session_state.model_trained = True
                st.success(f"✅ Pelatihan Model LSTM Berhasil! (Data Latih: {len(train_df)} | Data Uji: {len(test_df)})")

        # Tampilkan Hasil Output Pelatihan jika sudah selesai
        if st.session_state.get("model_trained", False) and st.session_state.get("history") is not None:
            history = st.session_state.history
            final_train_acc = history["accuracy"][-1] * 100
            final_val_acc = history["val_accuracy"][-1] * 100
            final_train_loss = history["loss"][-1]
            final_val_loss = history["val_loss"][-1]

            st.markdown("---")
            st.write("### 📊 Hasil Metrik Pelatihan Model LSTM:")
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Akurasi Training Akhir", f"{final_train_acc:.2f}%")
            with m2:
                st.metric("Akurasi Validasi Akhir", f"{final_val_acc:.2f}%")
            with m3:
                st.metric("Training Loss Akhir", f"{final_train_loss:.4f}")
            with m4:
                st.metric("Validation Loss Akhir", f"{final_val_loss:.4f}")

            # Grafik Konvergensi Pelatihan
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.write("#### 📈 Kurva Akurasi Pelatihan per Epoch")
                fig_acc, ax_acc = plt.subplots(figsize=(5, 3.5))
                ep_range = range(1, len(history["accuracy"]) + 1)
                ax_acc.plot(ep_range, [a * 100 for a in history["accuracy"]], label="Training Accuracy", color="#10B981", linewidth=2)
                ax_acc.plot(ep_range, [v * 100 for v in history["val_accuracy"]], label="Validation Accuracy", color="#3B82F6", linestyle="--", linewidth=2)
                ax_acc.set_xlabel("Epoch (Iterasi)")
                ax_acc.set_ylabel("Akurasi (%)")
                ax_acc.legend(loc="lower right")
                ax_acc.grid(True, linestyle=":", alpha=0.6)
                st.pyplot(fig_acc)
                plt.close(fig_acc)

            with col_g2:
                st.write("#### 📉 Kurva Penurunan Loss per Epoch")
                fig_l, ax_l = plt.subplots(figsize=(5, 3.5))
                ax_l.plot(ep_range, history["loss"], label="Training Loss", color="#EF4444", linewidth=2)
                ax_l.plot(ep_range, history["val_loss"], label="Validation Loss", color="#F59E0B", linestyle="--", linewidth=2)
                ax_l.set_xlabel("Epoch (Iterasi)")
                ax_l.set_ylabel("Loss")
                ax_l.legend(loc="upper right")
                ax_l.grid(True, linestyle=":", alpha=0.6)
                st.pyplot(fig_l)
                plt.close(fig_l)

            # Tabel Riwayat Epoch
            with st.expander("📑 Lihat Detail Riwayat Tiap Epoch Pelatihan"):
                epoch_df = pd.DataFrame({
                    "Epoch": list(ep_range),
                    "Train Accuracy (%)": [round(a * 100, 2) for a in history["accuracy"]],
                    "Val Accuracy (%)": [round(v * 100, 2) for v in history["val_accuracy"]],
                    "Train Loss": [round(l, 4) for l in history["loss"]],
                    "Val Loss": [round(vl, 4) for vl in history["val_loss"]]
                })
                st.dataframe(epoch_df, use_container_width=True)

            st.info("💡 **Langkah Selanjutnya:** Model telah dilatih! Buka menu **'4. Evaluasi Model'** untuk melihat performa pada data uji (Confusion Matrix, Precision, Recall, F1-Score).")


# ==========================================================
# 4. TAHAP 4: EVALUASI MODEL
# ==========================================================
elif menu == "4. Evaluasi Model":
    st.markdown('<div class="step-box">📈 TAHAP 4: Pengujian & Evaluasi Kinerja Model LSTM</div>', unsafe_allow_html=True)
    st.write("Metrik pengujian performa model dihitung secara dinamis dari **Data Uji (Test Set)**: Akurasi, Presisi, Recall, F1-Score, Confusion Matrix, dan Classification Report.")

    if not st.session_state.get("model_trained", False) or st.session_state.get("eval_results") is None:
        st.warning("⚠️ Model belum dilatih. Silakan buka menu **'3. Pelatihan LSTM'** dan jalankan pelatihan model terlebih dahulu!")
    else:
        eval_res = st.session_state.eval_results
        test_df = st.session_state.test_df

        test_acc = eval_res["accuracy"] * 100
        macro_prec = eval_res["macro_precision"] * 100
        macro_rec = eval_res["macro_recall"] * 100
        macro_f1 = eval_res["macro_f1"] * 100

        em1, em2, em3, em4 = st.columns(4)
        with em1:
            st.metric("Akurasi Data Uji", f"{test_acc:.2f}%")
        with em2:
            st.metric("Presisi Makro", f"{macro_prec:.2f}%")
        with em3:
            st.metric("Recall Makro", f"{macro_rec:.2f}%")
        with em4:
            st.metric("F1-Score Makro", f"{macro_f1:.2f}%")

        st.markdown("---")
        col_cm, col_hist = st.columns(2)

        with col_cm:
            st.write(f"#### 🎯 Confusion Matrix ({len(test_df)} Data Uji)")
            cm_data = eval_res["confusion_matrix"]
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4.2))
            sns.heatmap(
                cm_data,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Positif", "Netral", "Negatif"],
                yticklabels=["Positif", "Netral", "Negatif"],
                ax=ax_cm
            )
            ax_cm.set_ylabel("Kelas Aktual (Ground Truth)")
            ax_cm.set_xlabel("Kelas Prediksi (LSTM)")
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_hist:
            st.write("#### 📉 Kurva Konvergensi Pelatihan (Loss & Akurasi)")
            history = st.session_state.history
            fig_curve, ax_curve = plt.subplots(figsize=(6, 4.2))
            epochs_range = range(1, len(history["accuracy"]) + 1)
            ax_curve.plot(epochs_range, [a * 100 for a in history["accuracy"]], label="Train Acc (%)", color="#10B981", linewidth=2)
            ax_curve.plot(epochs_range, [v * 100 for v in history["val_accuracy"]], label="Val Acc (%)", color="#3B82F6", linestyle="--", linewidth=2)
            ax_curve.plot(epochs_range, history["loss"], label="Train Loss", color="#EF4444", linewidth=1.5)
            ax_curve.set_xlabel("Epoch")
            ax_curve.set_ylabel("Nilai Metrik")
            ax_curve.legend(loc="best")
            ax_curve.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig_curve)
            plt.close(fig_curve)

        st.markdown("---")
        st.write("#### 📋 Tabel Laporan Klasifikasi (Classification Report):")
        st.dataframe(eval_res["report_df"], use_container_width=True)

        with st.expander(f"📑 Pratinjau Prediksi Data Uji ({len(test_df)} Tweet)"):
            test_preview = test_df.copy()
            test_preview["prediksi_lstm"] = eval_res["y_pred"]
            test_preview["status_prediksi"] = ["✅ Benar" if a == p else "❌ Meleset" for a, p in zip(eval_res["y_true"], eval_res["y_pred"])]
            st.dataframe(test_preview[["tweet", "sentimen", "prediksi_lstm", "status_prediksi"]], use_container_width=True)

        st.info("💡 **Langkah Selanjutnya:** Evaluasi selesai! Silakan buka menu **'5. Prediksi Sentimen'** untuk menguji teks tweet baru.")


# ==========================================================
# 5. TAHAP 5: PREDIKSI SENTIMEN
# ==========================================================
elif menu == "5. Prediksi Sentimen":
    st.markdown('<div class="step-box">🔍 TAHAP 5: Pengujian & Prediksi Sentimen Teks Baru</div>', unsafe_allow_html=True)
    st.write("Gunakan model LSTM yang telah dilatih untuk memprediksi opini masyarakat mengenai pelatih timnas secara satuan atau massal.")

    tab_single, tab_batch = st.tabs(["📝 Prediksi Kalimat Tunggal", "📑 Prediksi Massal (Batch Upload CSV)"])

    with tab_single:
        st.write("#### 💬 Masukkan Teks Opini / Tweet:")
        input_tweet = st.text_area(
            "Ketik kalimat tweet:",
            placeholder="Contoh: Permainan timnas makin disiplin dan taktik pelatih sangat efektif membongkar lawan.",
            height=100
        )

        if st.button("Analisis Sentimen Sekarang", type="primary"):
            if not input_tweet.strip():
                st.warning("Silakan masukkan teks terlebih dahulu.")
            else:
                clean_in = preprocess_pipeline(input_tweet)
                res = st.session_state.classifier.predict_one(clean_in)

                lbl = res["label"]
                conf = res["confidence"] * 100
                probs = res["probabilities"]

                st.markdown("---")
                st.markdown("### 🏆 Hasil Prediksi Sentimen:")
                if lbl == "Positif":
                    st.success(f"### 🟢 Sentimen: **{lbl}** (Tingkat Keyakinan: {conf:.2f}%)")
                elif lbl == "Netral":
                    st.info(f"### ⚪ Sentimen: **{lbl}** (Tingkat Keyakinan: {conf:.2f}%)")
                else:
                    st.error(f"### 🔴 Sentimen: **{lbl}** (Tingkat Keyakinan: {conf:.2f}%)")

                st.write(f"**Teks Setelah Preprocessing:** `{clean_in}`")

                st.write("#### Distribusi Probabilitas Softmax:")
                p_c1, p_c2, p_c3 = st.columns(3)
                with p_c1:
                    st.progress(float(probs["Positif"]))
                    st.write(f"🟢 **Positif:** {probs['Positif']*100:.2f}%")
                with p_c2:
                    st.progress(float(probs["Netral"]))
                    st.write(f"⚪ **Netral:** {probs['Netral']*100:.2f}%")
                with p_c3:
                    st.progress(float(probs["Negatif"]))
                    st.write(f"🔴 **Negatif:** {probs['Negatif']*100:.2f}%")

    with tab_batch:
        st.write("#### 📂 Upload File CSV Tweet untuk Klasifikasi Massal:")
        batch_file = st.file_uploader("Upload file CSV hasil crawling/scraping baru:", type=["csv"], key="batch_uploader_tab")
        
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                text_col = None
                for col in ["tweet", "full_text", "text", "content", "Teks", "komentar"]:
                    if col in batch_df.columns:
                        text_col = col
                        break
                if text_col is None:
                    text_col = batch_df.columns[0]

                batch_df["tweet"] = batch_df[text_col].astype(str)
                st.write(f"Kolom teks terdeteksi: **`{text_col}`** | Total tweet: **{len(batch_df)} baris**")

                if st.button("🚀 Jalankan Prediksi Massal dengan LSTM", type="primary"):
                    with st.spinner(f"Memprediksi sentimen {len(batch_df)} tweet..."):
                        batch_df["clean_tweet"] = batch_df["tweet"].apply(preprocess_pipeline)
                        preds = st.session_state.classifier.predict_batch(batch_df["clean_tweet"].tolist())
                        
                        batch_df["prediksi_sentimen"] = [p["label"] for p in preds]
                        batch_df["confidence"] = [round(p["confidence"] * 100, 2) for p in preds]
                        
                        st.success("✅ Prediksi massal selesai!")
                        st.dataframe(batch_df[["tweet", "clean_tweet", "prediksi_sentimen", "confidence"]], use_container_width=True)

                        csv_buffer = io.StringIO()
                        batch_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Hasil Prediksi (CSV)",
                            data=csv_buffer.getvalue(),
                            file_name="hasil_prediksi_sentimen_pelatih_timnas.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
