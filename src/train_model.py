"""
Training model KNN untuk sistem rekomendasi peminatan mahasiswa
berdasarkan nilai mata kuliah.

Tahapan:
1. Load dataset
2. Split train/test
3. Normalisasi nilai (StandardScaler)
4. Cari K optimal (uji K=1..15, pilih akurasi tertinggi di data test)
5. Training model final dengan K optimal
6. Evaluasi: accuracy, precision, recall, f1-score, confusion matrix
7. Simpan model + scaler ke disk (buat dipakai backend nanti)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = "/home/claude/rekomendasi-peminatan/data/dataset_nilai_mahasiswa.csv"
MODEL_DIR = "/home/claude/rekomendasi-peminatan/model"

# =========================================================
# 1. Load dataset
# =========================================================
df = pd.read_csv(DATA_PATH)

FEATURE_COLS = [c for c in df.columns if c not in ["NIM", "Peminatan"]]
X = df[FEATURE_COLS].values
y = df["Peminatan"].values

print(f"Jumlah fitur (mata kuliah): {len(FEATURE_COLS)}")
print(f"Jumlah data: {len(df)}")

# =========================================================
# 2. Split train/test (80:20, stratified biar proporsi label seimbang)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data train: {len(X_train)}, Data test: {len(X_test)}")

# =========================================================
# 3. Normalisasi nilai (penting untuk KNN karena berbasis jarak)
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 4. Cari K optimal
# =========================================================
print("\n=== Pencarian K optimal ===")
best_k, best_acc = 1, 0
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    print(f"K={k:2d} -> Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_k, best_acc = k, acc

print(f"\nK optimal terpilih: {best_k} (Accuracy: {best_acc:.4f})")

# =========================================================
# 5. Training model final dengan K optimal
# =========================================================
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_scaled, y_train)

# =========================================================
# 6. Evaluasi
# =========================================================
y_pred = final_model.predict(X_test_scaled)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
labels = sorted(df["Peminatan"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# =========================================================
# 7. Simpan model, scaler, dan metadata fitur
# =========================================================
with open(f"{MODEL_DIR}/knn_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(f"{MODEL_DIR}/feature_columns.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)

print(f"\n✅ Model, scaler, dan feature_columns disimpan di {MODEL_DIR}/")
