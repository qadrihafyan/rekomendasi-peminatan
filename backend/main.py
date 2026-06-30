"""
Backend FastAPI: Sistem Rekomendasi Peminatan Mahasiswa
Endpoint utama: POST /api/recommend
"""

import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = "model"  # relatif terhadap folder backend, sesuaikan saat deploy

app = FastAPI(title="Sistem Rekomendasi Peminatan Mahasiswa")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model sekali saat startup
with open(f"{MODEL_DIR}/knn_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open(f"{MODEL_DIR}/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


class NilaiInput(BaseModel):
    Algoritma_Pemrograman: float = Field(..., ge=0, le=100)
    Struktur_Data: float = Field(..., ge=0, le=100)
    Basis_Data: float = Field(..., ge=0, le=100)
    Rekayasa_Perangkat_Lunak: float = Field(..., ge=0, le=100)
    Matematika_Diskrit: float = Field(..., ge=0, le=100)
    Statistika_Probabilitas: float = Field(..., ge=0, le=100)
    Jaringan_Komputer: float = Field(..., ge=0, le=100)
    Sistem_Operasi: float = Field(..., ge=0, le=100)
    Keamanan_Informasi: float = Field(..., ge=0, le=100)
    Pemrograman_Mobile: float = Field(..., ge=0, le=100)
    nim: str | None = None
    nama: str | None = None


@app.get("/")
def root():
    return {"status": "ok", "message": "Sistem Rekomendasi Peminatan jalan 🚀"}


@app.post("/api/recommend")
def recommend(data: NilaiInput):
    try:
        nilai_dict = data.dict(exclude={"nim", "nama"})
        X_new = np.array([[nilai_dict[c] for c in feature_columns]])
        X_new_scaled = scaler.transform(X_new)

        pred_label = model.predict(X_new_scaled)[0]
        proba = model.predict_proba(X_new_scaled)[0]
        classes = model.classes_

        ranking = sorted(
            [
                {"peminatan": c, "skor_kecocokan": round(float(p) * 100, 1)}
                for c, p in zip(classes, proba)
            ],
            key=lambda x: x["skor_kecocokan"],
            reverse=True,
        )

        return {
            "nim": data.nim,
            "nama": data.nama,
            "rekomendasi_utama": pred_label,
            "ranking_semua_peminatan": ranking,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
