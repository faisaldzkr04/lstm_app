"""
Script Otomatis Pengambil Data X (Twitter) Skripsi
Judul: ANALISIS SENTIMEN PERGANTIAN PELATIH TIMNAS INDONESIA (LSTM)
Fitur: Otomatis mengambil 1000+ tweet dan langsung menyimpannya ke format CSV siap pakai Streamlit.
"""

import sys
import pandas as pd
from datetime import datetime

def scrape_with_ntscraper(keyword="pelatih timnas indonesia", limit=1000, filename="dataset_timnas_x_1000.csv"):
    """
    Mengambil data tweet menggunakan ntscraper tanpa memerlukan API key / login manual.
    """
    print("=" * 65)
    print(f"🚀 Memulai Pengambilan Data Otomatis dari Media Sosial X...")
    print(f"📌 Kata Kunci: '{keyword}'")
    print(f"📊 Target Jumlah Tweet: {limit} Tweet")
    print("=" * 65)

    try:
        # pyrefly: ignore [missing-import]
        from ntscraper import Nitter
        scraper = Nitter(log_level=1, skip_instance_check=False)
        
        # Scrape tweets berdasarkan keyword
        tweets_data = scraper.get_tweets(keyword, mode='term', number=limit)
        
        raw_tweets = tweets_data.get('tweets', [])
        print(f"\n✅ Berhasil mengumpulkan {len(raw_tweets)} tweet!")

        if not raw_tweets:
            print("⚠️ Tidak ada tweet yang ditemukan atau server sedang sibuk. Silakan coba metode Colab.")
            return

        # Ekstraksi hanya teks tweet yang bersih dan tidak duplikat
        tweet_list = []
        for t in raw_tweets:
            text = t.get('text', '')
            if text and len(text.strip()) > 15:
                tweet_list.append(text)

        df = pd.DataFrame(tweet_list, columns=['tweet'])
        df = df.drop_duplicates(subset=['tweet']).reset_index(drop=True)

        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"💾 File berhasil disimpan: '{filename}' ({len(df)} tweet unik)")
        print("🎉 File ini SIAP LANGSUNG di-upload ke menu 'Prediksi Massal' di Streamlit!")

    except ImportError:
        print("\n[INFO] Library 'ntscraper' belum terpasang.")
        print("Sedang menginstall library ntscraper...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ntscraper"])
        print("Silakan jalankan script ini kembali: python scrape_x_auto.py")
    except Exception as e:
        print(f"\n❌ Terjadi kendala: {e}")

if __name__ == "__main__":
    keyword = "pelatih timnas indonesia"
    limit = 1000
    scrape_with_ntscraper(keyword, limit)
