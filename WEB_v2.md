# Arsitektur Web App

Dokumen ini menjelaskan alur kerja dan pembagian tugas untuk pengembangan web app prediksi/rekomendasi berbasis `Next.js` dan `FastAPI`.

## Ringkasan Konsep

Web app dibuat sederhana seperti kuis:

1. User membuka halaman web.
2. User mengisi beberapa pertanyaan berdasarkan fitur terpilih.
3. Frontend mengirim jawaban ke backend.
4. Backend melakukan validasi, preprocessing, dan inferensi model.
5. Backend mengirim hasil berupa output, penjelasan, dan rekomendasi.
6. Frontend menampilkan hasil ke user.

## Stack Yang Dipakai

- `Frontend`: Next.js
- `Backend`: FastAPI
- `Model`: machine learning model yang sudah dilatih
- `Feature selection`: chi-square
- `Database`: tidak dipakai untuk versi awal

## Kenapa Tanpa Database

Proyek ini aman dijalankan tanpa database karena alurnya sederhana:

- user mengisi kuis
- sistem memproses input
- hasil langsung ditampilkan
- tidak ada login
- tidak ada riwayat yang wajib disimpan

Database bisa ditambahkan nanti kalau dibutuhkan fitur seperti:

- history hasil kuis
- akun user
- dashboard admin

## Pembagian Kerja

### 1. Frontend `Next.js`

Tugas frontend adalah menangani tampilan dan interaksi user.

Yang dikerjakan frontend:

- halaman landing
- halaman kuis / assessment
- validasi ringan di UI
- tombol submit
- loading state
- error state
- halaman hasil
- menampilkan output, penjelasan, dan rekomendasi

Frontend tidak perlu menangani logic model.

### 2. Backend `FastAPI`

Tugas backend adalah memproses data dan menjalankan model.

Yang dikerjakan backend:

- menerima request dari frontend
- validasi data utama
- preprocessing input
- feature selection berdasarkan fitur terpilih
- menjalankan model prediksi
- mengembalikan hasil dalam format JSON

Backend menjadi pusat logika sistem.

### 3. Model / Machine Learning Layer

Bagian ini berisi komponen analisis dan prediksi.

Isinya:

- dataset hasil preprocessing
- fitur terpilih dari chi-square
- model yang sudah dilatih
- file artefak model bila diperlukan

Bagian ini biasanya dikelola dari sisi backend atau pipeline training.

## API Contract

API contract adalah kesepakatan format data antara frontend dan backend.

### Endpoint Minimal

- `GET /health`
- `POST /predict`

### Contoh Request `POST /predict`

```json
{
  "feature_1": 1,
  "feature_2": 0,
  "feature_3": 3
}
```

### Contoh Response

```json
{
  "prediction": "high_risk",
  "explanation": [
    "Beberapa fitur menunjukkan risiko yang lebih tinggi.",
    "Hasil dihitung dari fitur terpilih."
  ],
  "recommendation": "Disarankan konsultasi ke tenaga medis."
}
```

## Alur Sistem

```text
User
  ↓
Next.js UI
  ↓
FastAPI endpoint
  ↓
Preprocessing
  ↓
Chi-square selected features
  ↓
Model prediction
  ↓
JSON response
  ↓
Next.js result page
```

## Struktur Folder Yang Disarankan

```text
Dataset-NHANES/
├─ frontend/
├─ backend/
├─ data/
├─ notebooks/
├─ experiments/
├─ reports/
└─ docs/
```

### Penjelasan Folder

- `frontend/`: source code Next.js
- `backend/`: source code FastAPI
- `data/`: dataset mentah dan hasil olahan
- `notebooks/`: eksperimen dan eksplorasi
- `experiments/`: pengujian model atau skenario
- `reports/`: visualisasi dan tabel hasil
- `docs/`: dokumentasi proyek

## Pembagian Tugas Tim

### Frontend Developer

- membuat UI kuis
- membuat halaman hasil
- memastikan tampilan responsif
- menghubungkan UI ke API

### Backend Developer

- membuat endpoint API
- memproses input
- menjalankan model
- mengatur format response

### ML / Data Pipeline

- menyiapkan fitur
- melakukan chi-square feature selection
- melatih model
- menyimpan artefak model

## Catatan Implementasi

- Versi awal sebaiknya fokus ke MVP sederhana.
- Tidak perlu database kalau belum ada kebutuhan simpan data.
- Frontend dan backend bisa dikembangkan terpisah.
- Vercel bisa dipakai untuk deploy frontend dengan root directory mengarah ke folder `frontend`.
