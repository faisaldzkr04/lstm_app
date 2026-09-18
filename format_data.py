"""
Script Pemformat Data Media Sosial X (Twitter)
Judul: Analisis Sentimen Pergantian Pelatih Timnas (LSTM)
"""
import pandas as pd

def format_csv(input_path: str, output_path: str = "dataset_siap_upload.csv"):
    df = pd.read_csv(input_path)
    text_col = None
    for col in ["full_text", "text", "content", "tweet", "Teks"]:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError("Kolom teks tweet tidak ditemukan. Pastikan ada kolom teks/full_text.")
    
    out_df = pd.DataFrame()
    out_df["tweet"] = df[text_col].dropna().drop_duplicates()
    
    if "sentimen" in df.columns:
        out_df["sentimen"] = df["sentimen"]
        
    out_df.to_csv(output_path, index=False)
    print(f"File berhasil diformat ke '{output_path}' ({len(out_df)} tweet).")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        format_csv(sys.argv[1])
    else:
        print("Gunakan: python format_data.py <nama_file_mentah.csv>")
