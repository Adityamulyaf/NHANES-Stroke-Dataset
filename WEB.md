# Panduan Pengembangan & Deployment Web App (FastAPI + React/Vite)

Dokumen ini menjelaskan langkah-langkah membangun, menjalankan, dan mendeploy aplikasi web klasifikasi risiko stroke NHANES menggunakan arsitektur modern standar industri:
1. **Backend**: **FastAPI** (Python) untuk melayani *inference* prediksi model machine learning.
2. **Frontend**: **React + Vite** (JavaScript/CSS) untuk antarmuka pengguna (*User Interface*) yang interaktif dan responsif.

---

## 📂 Struktur Folder Proyek yang Direkomendasikan

Buat folder `backend/` dan `frontend/` di root repositori Anda dengan struktur berikut:

```text
NHANES-Stroke-Dataset/
│
├── backend/                       # [BARU] Folder Kode Backend
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py                # Kode API Utama FastAPI
│   └── requirements.txt           # Dependensi Backend (fastapi, uvicorn, joblib, scikit-learn, etc.)
│
├── frontend/                      # [BARU] Folder Kode Frontend
│   ├── src/
│   │   ├── App.jsx                # Logika & Form Input UI React
│   │   ├── index.css              # Custom Styling (Vanilla CSS Premium)
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── models/                        # Folder Model Terlatih (Root)
│   └── feature_selection/
│       ├── rf_best_model.pkl
│       └── scaler.pkl
│
└── WEB.md                         # Dokumen Panduan Ini
```

---

## ⚡ 1. Pembuatan Backend (FastAPI)

### Langkah 1.1: Buat Dependensi Backend
Buat berkas [backend/requirements.txt](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/backend/requirements.txt) dan isi dengan library berikut:
```text
fastapi
uvicorn
joblib
numpy
pandas
scikit-learn
pydantic
```

### Langkah 1.2: Tulis Kode Server API FastAPI
Buat berkas [backend/app/main.py](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/backend/app/main.py) dan isi dengan kode lengkap di bawah ini. Kode ini memuat model, scaler, menyelaraskan urutan fitur, menstandarisasi input, dan mengembalikan hasil prediksi.

```python
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NHANES Stroke Prediction API",
    description="API untuk prediksi risiko stroke berdasarkan data klinis dan gaya hidup NHANES.",
    version="1.0.0"
)

# Aktifkan CORS agar frontend React (dari domain berbeda) dapat mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Saat produksi, Anda bisa mengganti "*" dengan URL Vercel frontend Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolusi path relatif ke folder root proyek
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'feature_selection' / 'rf_best_model.pkl'
SCALER_PATH = BASE_DIR / 'models' / 'feature_selection' / 'scaler.pkl'
FEATURE_GROUPS_PATH = BASE_DIR / 'data' / 'processed' / 'feature_selection' / 'feature_groups.pkl'

# Load artifacts model
try:
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Berkas model atau scaler tidak ditemukan di folder models/.")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_groups = joblib.load(FEATURE_GROUPS_PATH)
    
    # Ambil daftar fitur sesuai urutan latih agar tidak terjadi misalignment kolom
    CLINICAL = feature_groups['CLINICAL']
    SLEEP = feature_groups['SLEEP']
    STRESS = feature_groups['STRESS']
    PHYSICAL = feature_groups['PHYSICAL']
    FEATURE_ORDER = CLINICAL + SLEEP + STRESS + PHYSICAL
    print("Model, Scaler, dan Metadata Fitur berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model, scaler, FEATURE_ORDER = None, None, []

# Schema input data dari Frontend menggunakan Pydantic (24 fitur hasil seleksi fitur)
class StrokeInput(BaseModel):
    # Klinis (12 fitur)
    age: float
    education: float
    income_ratio: float
    waist_circ: float
    systolic_bp: float
    diastolic_bp: float
    hypertension: float
    diabetes: float
    heart_failure: float
    coronary_disease: float
    heart_attack: float
    ever_smoked: float
    # Tidur (4 fitur)
    snoring_freq: float
    sleep_apnea: float
    sleep_problem_doctor: float
    daytime_sleepy: float
    # Stres (5 fitur)
    stress_anhedonia: float
    stress_depressed: float
    stress_fatigue: float
    stress_concentration: float
    stress_self_esteem: float
    # Aktivitas Fisik (3 fitur)
    vigorous_leisure: float
    vigorous_leisure_min: float
    moderate_leisure: float

@app.get("/")
def home():
    return {"status": "running", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: StrokeInput):
    if not model or not scaler:
        raise HTTPException(
            status_code=500, 
            detail="Model ML belum berhasil dimuat di server. Hubungi administrator."
        )
    
    try:
        # 1. Konversi Pydantic schema ke dictionary
        input_dict = data.dict()
        
        # 2. Urutkan fitur sesuai urutan kolom yang dipakai saat fitting model
        input_vector = [input_dict[feat] for feat in FEATURE_ORDER]
        
        # 3. Standardisasi menggunakan scaler (.transform)
        input_scaled = scaler.transform([input_vector])
        
        # 4. Prediksi probabilitas (kelas positif/stroke ada di indeks 1)
        proba = model.predict_proba(input_scaled)[0, 1]
        
        # 5. Penentuan hasil berdasarkan Tuned Threshold Random Forest E (0.2315)
        threshold = 0.2315
        prediction = int(proba >= threshold)
        
        return {
            "prediction": prediction,
            "probability": round(float(proba) * 100, 2),  # Persentase (0 - 100%)
            "threshold": round(threshold * 100, 2),
            "risk_level": "RISIKO TINGGI (High Risk)" if prediction == 1 else "RISIKO RENDAH (Low Risk)",
            "message": "Disarankan untuk melakukan konsultasi medis lebih lanjut." if prediction == 1 else "Tetap pertahankan pola hidup sehat Anda!"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Terjadi kesalahan pemrosesan: {str(e)}")
```

---

## 🎨 2. Pembuatan Frontend (React/Vite)

### Langkah 2.1: Inisialisasi Proyek React Vite
Buka terminal baru di root proyek Anda, lalu jalankan perintah berikut untuk menginisialisasi proyek React dengan template Vite:
```bash
# Buat proyek di folder frontend
npm create vite@latest frontend -- --template react
```

> [!NOTE]
> Setelah inisialisasi selesai, masuk ke folder `frontend/` dan pasang pustaka pendukung untuk komunikasi API (seperti Axios):
> ```bash
> cd frontend
> npm install axios
> ```

### Langkah 2.2: Tulis Tampilan Formulir React
Ubah isi berkas `frontend/src/App.jsx` menjadi form interaktif premium. Berikut template UI elegan dengan validasi dan efek visual:

```jsx
import React, { useState } from 'react';
import axios from 'axios';
import './index.css';

// Tentukan alamat API (Local fallback ke localhost, prod membaca ENV)
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [formData, setFormData] = useState({
    age: 40, education: 3, income_ratio: 2.5, waist_circ: 90,
    systolic_bp: 120, diastolic_bp: 80, hypertension: 0, diabetes: 0,
    heart_failure: 0, coronary_disease: 0, heart_attack: 0, ever_smoked: 0,
    snoring_freq: 0, sleep_apnea: 0, sleep_problem_doctor: 0, daytime_sleepy: 0,
    stress_anhedonia: 0, stress_depressed: 0, stress_fatigue: 0,
    stress_concentration: 0, stress_self_esteem: 0,
    vigorous_leisure: 0, vigorous_leisure_min: 0, moderate_leisure: 0
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Koneksi ke server backend gagal.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Stroke Risk Calculator</h1>
        <p>NHANES Explainable Machine Learning Classifier</p>
      </header>

      <div className="main-content">
        {/* FORM INPUT */}
        <form onSubmit={handleSubmit} className="form-card">
          <h2>Kuesioner Kesehatan & Gaya Hidup</h2>
          
          <div className="form-section">
            <h3>1. Data Demografi & Klinis</h3>
            <div className="grid-2">
              <label>
                Usia: {formData.age} Tahun
                <input type="range" name="age" min="18" max="100" value={formData.age} onChange={handleInputChange} />
              </label>
              <label>
                Rasio Pendapatan (Income Ratio): {formData.income_ratio}
                <input type="range" name="income_ratio" min="0" max="5" step="0.1" value={formData.income_ratio} onChange={handleInputChange} />
              </label>
              <label>
                Tekanan Darah Sistolik: {formData.systolic_bp} mmHg
                <input type="range" name="systolic_bp" min="80" max="200" value={formData.systolic_bp} onChange={handleInputChange} />
              </label>
              <label>
                Tekanan Darah Diastolik: {formData.diastolic_bp} mmHg
                <input type="range" name="diastolic_bp" min="50" max="120" value={formData.diastolic_bp} onChange={handleInputChange} />
              </label>
              <label>
                Lingkar Pinggang: {formData.waist_circ} cm
                <input type="range" name="waist_circ" min="50" max="150" value={formData.waist_circ} onChange={handleInputChange} />
              </label>
              <label>
                Tingkat Pendidikan:
                <select name="education" value={formData.education} onChange={handleInputChange}>
                  <option value="1">SD / Tidak Sekolah</option>
                  <option value="2">SMP / Sederajat</option>
                  <option value="3">SMA / Sederajat</option>
                  <option value="4">Diploma (D3)</option>
                  <option value="5">Sarjana (S1 ke atas)</option>
                </select>
              </label>
            </div>
            <div className="grid-3 checkboxes">
              <label><input type="checkbox" name="hypertension" checked={formData.hypertension === 1} onChange={(e) => setFormData(p => ({...p, hypertension: e.target.checked ? 1 : 0}))} /> Hipertensi</label>
              <label><input type="checkbox" name="diabetes" checked={formData.diabetes === 1} onChange={(e) => setFormData(p => ({...p, diabetes: e.target.checked ? 1 : 0}))} /> Diabetes</label>
              <label><input type="checkbox" name="heart_failure" checked={formData.heart_failure === 1} onChange={(e) => setFormData(p => ({...p, heart_failure: e.target.checked ? 1 : 0}))} /> Gagal Jantung</label>
              <label><input type="checkbox" name="coronary_disease" checked={formData.coronary_disease === 1} onChange={(e) => setFormData(p => ({...p, coronary_disease: e.target.checked ? 1 : 0}))} /> Penyakit Koroner</label>
              <label><input type="checkbox" name="heart_attack" checked={formData.heart_attack === 1} onChange={(e) => setFormData(p => ({...p, heart_attack: e.target.checked ? 1 : 0}))} /> Serangan Jantung</label>
              <label><input type="checkbox" name="ever_smoked" checked={formData.ever_smoked === 1} onChange={(e) => setFormData(p => ({...p, ever_smoked: e.target.checked ? 1 : 0}))} /> Riwayat Merokok</label>
            </div>
          </div>

          <div className="form-section">
            <h3>2. Pola & Masalah Tidur</h3>
            <div className="grid-2">
              <label>
                Frekuensi Mendengkur:
                <select name="snoring_freq" value={formData.snoring_freq} onChange={handleInputChange}>
                  <option value="0">Tidak Pernah</option>
                  <option value="1">Jarang (1-2x seminggu)</option>
                  <option value="3">Sering (3-4x seminggu)</option>
                  <option value="5">Sangat Sering (>= 5x seminggu)</option>
                </select>
              </label>
              <label>
                Apnea Tidur (Napas Berhenti):
                <select name="sleep_apnea" value={formData.sleep_apnea} onChange={handleInputChange}>
                  <option value="0">Tidak Pernah</option>
                  <option value="1">Jarang (1-2x seminggu)</option>
                  <option value="3">Sering/Sangat Sering</option>
                </select>
              </label>
              <label>
                Tingkat Kantuk Siang Hari:
                <select name="daytime_sleepy" value={formData.daytime_sleepy} onChange={handleInputChange}>
                  <option value="0">Tidak Pernah Mengantuk</option>
                  <option value="1">Ringan (1-2x sebulan)</option>
                  <option value="2">Sedang (2-4x seminggu)</option>
                  <option value="3">Berat (5-7x seminggu)</option>
                </select>
              </label>
              <div className="checkboxes" style={{marginTop: '25px'}}>
                <label><input type="checkbox" name="sleep_problem_doctor" checked={formData.sleep_problem_doctor === 1} onChange={(e) => setFormData(p => ({...p, sleep_problem_doctor: e.target.checked ? 1 : 0}))} /> Konsultasi Masalah Tidur ke Dokter</label>
              </div>
            </div>
          </div>

          <div className="form-section">
            <h3>3. Tingkat Stres (PHQ-9 2-Minggu Terakhir)</h3>
            <p className="subtitle">Skala: 0 = Tidak sama sekali | 1 = Beberapa hari | 2 = > Setengah hari | 3 = Hampir setiap hari</p>
            <div className="grid-2">
              <label>
                Kehilangan minat melakukan sesuatu:
                <input type="range" name="stress_anhedonia" min="0" max="3" step="1" value={formData.stress_anhedonia} onChange={handleInputChange} />
              </label>
              <label>
                Merasa depresi / putus asa:
                <input type="range" name="stress_depressed" min="0" max="3" step="1" value={formData.stress_depressed} onChange={handleInputChange} />
              </label>
              <label>
                Merasa lelah / kurang energi:
                <input type="range" name="stress_fatigue" min="0" max="3" step="1" value={formData.stress_fatigue} onChange={handleInputChange} />
              </label>
              <label>
                Kesulitan konsentrasi:
                <input type="range" name="stress_concentration" min="0" max="3" step="1" value={formData.stress_concentration} onChange={handleInputChange} />
              </label>
              <label>
                Merasa rendah diri / gagal:
                <input type="range" name="stress_self_esteem" min="0" max="3" step="1" value={formData.stress_self_esteem} onChange={handleInputChange} />
              </label>
            </div>
          </div>

          <div className="form-section">
            <h3>4. Aktivitas Fisik</h3>
            <div className="grid-2">
              <label>
                Durasi Olahraga Berat (Menit/Minggu): {formData.vigorous_leisure_min} m
                <input type="range" name="vigorous_leisure_min" min="0" max="600" step="10" value={formData.vigorous_leisure_min} onChange={handleInputChange} />
              </label>
              <div className="checkboxes" style={{marginTop: '25px'}}>
                <label><input type="checkbox" name="vigorous_leisure" checked={formData.vigorous_leisure === 1} onChange={(e) => setFormData(p => ({...p, vigorous_leisure: e.target.checked ? 1 : 0}))} /> Olahraga Berat Rutin</label>
                <label><input type="checkbox" name="moderate_leisure" checked={formData.moderate_leisure === 1} onChange={(e) => setFormData(p => ({...p, moderate_leisure: e.target.checked ? 1 : 0}))} /> Olahraga Sedang Rutin</label>
              </div>
            </div>
          </div>

          <button type="submit" disabled={loading} className="btn-predict">
            {loading ? 'Menghitung Risiko...' : 'Prediksi Risiko Stroke'}
          </button>
        </form>

        {/* OUTPUT HASIL PREDIKSI */}
        <div className="result-panel">
          {error && <div className="card error-card">{error}</div>}

          {result && (
            <div className={`card result-card ${result.prediction === 1 ? 'high-risk' : 'low-risk'}`}>
              <h2>Hasil Analisis Risiko</h2>
              
              <div className="gauge-container">
                <div className="gauge-value">{result.probability}%</div>
                <p>Probabilitas Risiko Terakumulasi</p>
              </div>

              <div className="badge">{result.risk_level}</div>
              <p className="description">{result.message}</p>
              
              <div className="divider"></div>
              <p className="notes">
                *Tuned threshold klasifikasi model diatur pada <strong>{result.threshold}%</strong> untuk mendeteksi pasien berisiko stroke secara optimal pada dataset perwakilan populasi umum NHANES.
              </p>
            </div>
          )}

          {!result && !error && !loading && (
            <div className="card info-card">
              <h3>Petunjuk Penggunaan</h3>
              <p>Silakan isi kuesioner parameter klinis dan gaya hidup Anda di panel kiri, kemudian klik tombol <strong>Prediksi Risiko Stroke</strong>.</p>
              <p>Hasil kalkulasi model kecerdasan buatan akan langsung muncul di panel ini.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
```

### Langkah 2.3: Desain Styling CSS Premium
Tulis visualisasi antarmuka web di berkas `frontend/src/index.css` agar terlihat modern dan *wow* (menggunakan skema warna HSL gelap/terang, desain kartu melayang, gradien warna halus, dan bayangan lembut):

```css
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

:root {
  --primary-color: #2C3E93;
  --accent-color: #10B981;
  --warning-color: #E74C3C;
  --bg-color: #f3f5fa;
  --card-bg: rgba(255, 255, 255, 0.85);
  --text-dark: #1E293B;
  --text-light: #64748B;
  --shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
  --border-radius: 16px;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Outfit', sans-serif;
  background: radial-gradient(circle at 10% 20%, rgb(239, 246, 255) 0%, rgb(219, 234, 254) 90%);
  color: var(--text-dark);
  min-height: 100vh;
}

.app-container {
  max-width: 1280px;
  margin: 0 auto;
  padding: 40px 20px;
}

.app-header {
  text-align: center;
  margin-bottom: 40px;
}

.app-header h1 {
  font-size: 2.8rem;
  font-weight: 700;
  color: var(--primary-color);
  margin-bottom: 8px;
}

.app-header p {
  color: var(--text-light);
  font-size: 1.1rem;
}

.main-content {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 30px;
  align-items: start;
}

@media (max-width: 900px) {
  .main-content {
    grid-template-columns: 1fr;
  }
}

.form-card {
  background: var(--card-bg);
  backdrop-filter: blur(10px);
  padding: 30px;
  border-radius: var(--border-radius);
  box-shadow: var(--shadow);
  border: 1px solid rgba(255, 255, 255, 0.6);
}

.form-card h2 {
  font-size: 1.8rem;
  margin-bottom: 25px;
  color: var(--primary-color);
}

.form-section {
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid rgba(100, 116, 139, 0.1);
}

.form-section h3 {
  font-size: 1.2rem;
  margin-bottom: 15px;
  font-weight: 600;
}

.subtitle {
  font-size: 0.85rem;
  color: var(--text-light);
  margin-bottom: 15px;
}

.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.grid-3 {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 15px;
}

label {
  display: flex;
  flex-direction: column;
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--text-dark);
}

input[type="range"] {
  margin-top: 8px;
  accent-color: var(--primary-color);
  width: 100%;
}

select {
  margin-top: 8px;
  padding: 10px;
  border: 1px solid rgba(100, 116, 139, 0.2);
  border-radius: 8px;
  background: white;
  font-family: inherit;
  font-size: 0.95rem;
}

.checkboxes {
  margin-top: 15px;
}

.checkboxes label {
  flex-direction: row;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.btn-predict {
  background: linear-gradient(135deg, var(--primary-color) 0%, #4f46e5 100%);
  color: white;
  border: none;
  padding: 15px;
  width: 100%;
  border-radius: 10px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 5px 15px rgba(79, 70, 229, 0.3);
  transition: all 0.3s ease;
}

.btn-predict:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4);
}

.btn-predict:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.card {
  padding: 30px;
  border-radius: var(--border-radius);
  box-shadow: var(--shadow);
  background: white;
  text-align: center;
}

.info-card {
  background: rgba(255, 255, 255, 0.6);
  border: 1px dashed rgba(100, 116, 139, 0.3);
}

.info-card h3 {
  margin-bottom: 15px;
  color: var(--text-light);
}

.info-card p {
  font-size: 0.95rem;
  color: var(--text-light);
  margin-bottom: 10px;
}

.error-card {
  background: #fef2f2;
  color: var(--warning-color);
  border: 1px solid #fee2e2;
}

.result-card h2 {
  margin-bottom: 20px;
  font-size: 1.6rem;
}

.gauge-container {
  width: 150px;
  height: 150px;
  border-radius: 50%;
  margin: 0 auto 20px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  box-shadow: inset 0 0 15px rgba(0,0,0,0.05);
}

.result-card.low-risk .gauge-container {
  border: 10px solid var(--accent-color);
  background: rgba(16, 185, 129, 0.05);
}

.result-card.high-risk .gauge-container {
  border: 10px solid var(--warning-color);
  background: rgba(231, 76, 60, 0.05);
}

.gauge-value {
  font-size: 2.2rem;
  font-weight: 700;
}

.gauge-container p {
  font-size: 0.75rem;
  color: var(--text-light);
}

.badge {
  display: inline-block;
  padding: 8px 16px;
  border-radius: 20px;
  font-weight: 600;
  margin-bottom: 15px;
  font-size: 0.9rem;
}

.low-risk .badge {
  background: #d1fae5;
  color: #065f46;
}

.high-risk .badge {
  background: #fee2e2;
  color: #991b1b;
}

.description {
  font-size: 1.1rem;
  font-weight: 500;
  margin-bottom: 20px;
}

.divider {
  height: 1px;
  background: rgba(0, 0, 0, 0.08);
  margin: 20px 0;
}

.notes {
  font-size: 0.8rem;
  color: var(--text-light);
  line-height: 1.4;
}
```

---

## 💻 3. Cara Menjalankan Lokal

### Langkah 3.1: Jalankan Backend (FastAPI)
Buka terminal baru di root folder proyek, lalu jalankan:
```bash
# Aktifkan virtual environment Anda terlebih dahulu jika ada
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```
API server backend akan berjalan di http://localhost:8000. Anda juga bisa menguji API tersebut di http://localhost:8000/docs (Swagger UI).

### Langkah 3.2: Jalankan Frontend (React/Vite)
Buka terminal baru yang lain, masuk ke folder `frontend/`, dan jalankan:
```bash
npm install
npm run dev
```
Aplikasi frontend web Anda akan berjalan di http://localhost:5173.

---

## 🚀 4. Langkah Deployment ke Server Publik

### Langkah 4.1: Deploy Backend ke Render (Gratis)
1. Buat akun di [Render](https://render.com/).
2. Buat layanan baru: **Web Service**.
3. Hubungkan repositori GitHub Anda.
4. Konfigurasikan pengaturan build berikut:
   * **Name**: `nhanes-stroke-backend`
   * **Environment**: `Python 3`
   * **Build Command**: `pip install -r backend/requirements.txt`
   * **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
5. Salin URL Web Service Anda yang dibuat oleh Render (misalnya: `https://nhanes-stroke-backend.onrender.com`).

---

### Langkah 4.2: Deploy Frontend ke Vercel (Gratis)
1. Buat akun di [Vercel](https://vercel.com/).
2. Buat proyek baru dengan menghubungkan repositori GitHub Anda.
3. Konfigurasikan pengaturan berikut:
   * **Root Directory**: Pilih/masukkan `frontend`.
   * **Framework Preset**: Pilih **Vite**.
4. Di bagian **Environment Variables**, tambahkan variabel berikut:
   * **Key**: `VITE_API_URL`
   * **Value**: Masukkan URL Render Backend Anda (dari Langkah 4.1).
5. Klik **Deploy**. Selesai! Aplikasi web React Anda siap diakses secara publik.
