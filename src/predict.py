"""
Fungsi inference: ambil input nilai mahasiswa baru, kembalikan
rekomendasi peminatan beserta ranking probabilitas tiap peminatan.
"""

import pickle
import numpy as np

MODEL_DIR = "/home/claude/rekomendasi-peminatan/model"

with open(f"{MODEL_DIR}/knn_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(f"{MODEL_DIR}/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


def predict_peminatan(nilai_dict: dict) -> dict:
    """
    nilai_dict: dict {nama_mata_kuliah: nilai}, harus mencakup semua
    kolom di feature_columns.

    Return: dict berisi rekomendasi utama + ranking probabilitas
    semua peminatan.
    """
    missing = [c for c in feature_columns if c not in nilai_dict]
    if missing:
        raise ValueError(f"Nilai mata kuliah berikut belum diisi: {missing}")

    X_new = np.array([[nilai_dict[c] for c in feature_columns]])
    X_new_scaled = scaler.transform(X_new)

    pred_label = model.predict(X_new_scaled)[0]
    proba = model.predict_proba(X_new_scaled)[0]
    classes = model.classes_

    ranking = sorted(
        [{"peminatan": c, "skor_kecocokan": round(float(p) * 100, 1)} for c, p in zip(classes, proba)],
        key=lambda x: x["skor_kecocokan"],
        reverse=True,
    )

    return {
        "rekomendasi_utama": pred_label,
        "ranking_semua_peminatan": ranking,
    }


if __name__ == "__main__":
    # Contoh uji coba: mahasiswa dengan nilai bagus di mata kuliah jaringan
    contoh_nilai = {
        "Algoritma_Pemrograman": 70,
        "Struktur_Data": 68,
        "Basis_Data": 72,
        "Rekayasa_Perangkat_Lunak": 65,
        "Matematika_Diskrit": 60,
        "Statistika_Probabilitas": 62,
        "Jaringan_Komputer": 90,
        "Sistem_Operasi": 88,
        "Keamanan_Informasi": 85,
        "Pemrograman_Mobile": 60,
    }

    hasil = predict_peminatan(contoh_nilai)
    print("Rekomendasi utama:", hasil["rekomendasi_utama"])
    print("\nRanking semua peminatan:")
    for r in hasil["ranking_semua_peminatan"]:
        print(f"  {r['peminatan']}: {r['skor_kecocokan']}%")
