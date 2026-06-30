# Sistem Rekomendasi Peminatan Mahasiswa (KNN)

Sistem yang merekomendasikan peminatan/konsentrasi mata kuliah pilihan bagi
mahasiswa Teknik Informatika berdasarkan nilai mata kuliah yang sudah
ditempuh, menggunakan algoritma **K-Nearest Neighbors (KNN)**.

## Struktur Folder
```
rekomendasi-peminatan/
├── data/
│   └── dataset_nilai_mahasiswa.csv   ← dataset dummy (320 mahasiswa, 4 peminatan)
├── src/
│   ├── generate_dataset.py            ← generate dataset dummy
│   ├── train_model.py                 ← training + evaluasi model KNN
│   └── predict.py                     ← contoh inference standalone
├── backend/
│   ├── main.py                        ← FastAPI, endpoint /api/recommend
│   ├── requirements.txt
│   ├── Procfile                       ← untuk deploy ke Railway
│   └── model/                         ← model & scaler hasil training (.pkl)
└── frontend/
    └── index.html                     ← form input nilai + tampilan hasil
```

## Peminatan yang Tersedia
1. **Software Engineering** — kuat di Algoritma Pemrograman, Struktur Data, RPL, Basis Data
2. **Data Science & AI** — kuat di Matematika Diskrit, Statistika, Basis Data, Algoritma
3. **Jaringan & Keamanan Siber** — kuat di Jaringan Komputer, Sistem Operasi, Keamanan Informasi
4. **Sistem Informasi & Mobile** — kuat di Pemrograman Mobile, RPL, Basis Data

## Cara Menjalankan (Lokal)

### 1. (Opsional) Generate ulang dataset / training ulang model
```bash
cd src
python generate_dataset.py   # bikin dataset dummy baru
python train_model.py        # training ulang + evaluasi
```
Model hasil training otomatis tersimpan ke `backend/model/`.

### 2. Jalankan backend
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Backend jalan di `http://localhost:8001`.

### 3. Jalankan frontend
Buka `frontend/index.html` langsung di browser. Pastikan `API_BASE` di
dalam file menunjuk ke `http://localhost:8001/api`.

## Hasil Evaluasi Model (dataset dummy)
- **K optimal**: 9
- **Accuracy**: 81.25%
- Precision/recall per kelas bervariasi 0.67–0.94 — kelas yang paling sering
  "tertukar" adalah Software Engineering vs Data Science & AI, karena
  keduanya sama-sama mengandalkan nilai tinggi di Algoritma Pemrograman dan
  Basis Data. Ini bisa jadi bahan analisis di bab pembahasan skripsi.

## Catatan untuk Skripsi

- **Dataset ini dummy/simulasi** — sebelum sidang, sebaiknya dijelaskan
  secara eksplisit di BAB Metodologi bahwa data dibangkitkan secara
  terprogram dengan pola distribusi nilai yang merepresentasikan
  karakteristik tiap peminatan, ditambah noise acak (gaussian) untuk
  mensimulasikan variasi nilai mahasiswa di dunia nyata.
- Kalau nanti dapat akses data riil dari prodi (SIAKAD), tinggal ganti isi
  `data/dataset_nilai_mahasiswa.csv` dengan data asli (kolom harus sama
  persis namanya), lalu jalankan ulang `train_model.py`.
- Bagian evaluasi (`classification_report`, `confusion_matrix`) sudah
  otomatis tercetak saat `train_model.py` dijalankan — tinggal di-screenshot
  atau disalin ke bab pengujian.
- Untuk deploy production, ikuti pola yang sama seperti project
  sebelumnya: backend ke Railway, frontend ke Vercel.
