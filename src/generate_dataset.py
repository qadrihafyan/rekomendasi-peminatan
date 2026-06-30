"""
Generate dataset dummy nilai mahasiswa Teknik Informatika untuk
sistem rekomendasi peminatan (Software Engineering, Data Science,
Jaringan & Keamanan, Sistem Informasi & Mobile).

Logika: tiap peminatan punya "mata kuliah penciri" yang nilainya
sengaja dibuat cenderung lebih tinggi untuk mahasiswa yang label-nya
peminatan tsb, ditambah noise acak supaya realistis (gak terlalu mudah
ditebak / overfit).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# =========================================================
# 1. Definisi mata kuliah (10 MK inti, dipakai sebagai fitur)
# =========================================================
MATA_KULIAH = [
    "Algoritma_Pemrograman",
    "Struktur_Data",
    "Basis_Data",
    "Rekayasa_Perangkat_Lunak",
    "Matematika_Diskrit",
    "Statistika_Probabilitas",
    "Jaringan_Komputer",
    "Sistem_Operasi",
    "Keamanan_Informasi",
    "Pemrograman_Mobile",
]

PEMINATAN = [
    "Software Engineering",
    "Data Science & AI",
    "Jaringan & Keamanan Siber",
    "Sistem Informasi & Mobile",
]

# Mata kuliah "penciri" tiap peminatan -> nilai cenderung lebih tinggi
PENCIRI = {
    "Software Engineering": ["Algoritma_Pemrograman", "Struktur_Data", "Rekayasa_Perangkat_Lunak", "Basis_Data"],
    "Data Science & AI": ["Matematika_Diskrit", "Statistika_Probabilitas", "Basis_Data", "Algoritma_Pemrograman"],
    "Jaringan & Keamanan Siber": ["Jaringan_Komputer", "Sistem_Operasi", "Keamanan_Informasi"],
    "Sistem Informasi & Mobile": ["Pemrograman_Mobile", "Rekayasa_Perangkat_Lunak", "Basis_Data"],
}

N_PER_KELAS = 80  # jumlah mahasiswa dummy per peminatan -> total 320 baris


def generate_nilai(mata_kuliah_list, penciri_list):
    """Generate satu baris nilai mahasiswa untuk satu peminatan."""
    nilai = {}
    for mk in mata_kuliah_list:
        if mk in penciri_list:
            # mata kuliah penciri -> nilai cenderung tinggi (mean 82, std 7)
            skor = np.random.normal(loc=82, scale=7)
        else:
            # mata kuliah non-penciri -> nilai sedang/acak (mean 70, std 10)
            skor = np.random.normal(loc=70, scale=10)
        nilai[mk] = int(np.clip(skor, 40, 100))  # nilai dibatasi 40-100
    return nilai


rows = []
for peminatan in PEMINATAN:
    penciri_list = PENCIRI[peminatan]
    for i in range(N_PER_KELAS):
        nilai = generate_nilai(MATA_KULIAH, penciri_list)
        nilai["Peminatan"] = peminatan
        rows.append(nilai)

df = pd.DataFrame(rows)

# Tambah kolom ID & NIM dummy, acak urutan baris biar gak berurutan per kelas
df.insert(0, "NIM", [f"21110{str(i+1).zfill(4)}" for i in range(len(df))])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Reorder kolom: NIM, mata kuliah, lalu label paling akhir
df = df[["NIM"] + MATA_KULIAH + ["Peminatan"]]

output_path = "/home/claude/rekomendasi-peminatan/data/dataset_nilai_mahasiswa.csv"
df.to_csv(output_path, index=False)

print(f"Dataset tersimpan: {output_path}")
print(f"Total baris: {len(df)}")
print(f"Distribusi label:\n{df['Peminatan'].value_counts()}")
print("\nContoh 5 baris pertama:")
print(df.head())
