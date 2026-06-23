# Panduan Pengembangan (Development Guide)

Dokumen ini ditujukan bagi anggota tim atau kontributor yang ingin mengedit, mengembangkan, atau memelihara kode aplikasi **NHANES Stroke Assessment** baik di bagian **Frontend (Next.js)** maupun **Backend (FastAPI)**.

---

## 📂 Struktur Utama Proyek

* **[frontend/](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/frontend)**: Berisi kode aplikasi antarmuka Next.js (React, Tailwind CSS, TypeScript).
* **[backend/](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/backend)**: Berisi kode logika API FastAPI (Python) yang memuat model ML dan menghitung XAI.
* **[models/](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/models)**: Tempat penyimpanan file serialisasi model Random Forest (`rf_best_model.pkl`) dan Standard Scaler (`scaler.pkl`).
* **[data/](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/data)**: Dataset mentah dan olahan. Metadata kelompok fitur tersimpan di `data/processed/feature_selection/feature_groups.pkl`.

---

## 💻 1. Mengembangkan & Mengedit Frontend (Next.js)

Aplikasi frontend Next.js dirancang secara dinamis menggunakan TypeScript. Jika Anda ingin mengubah tampilan kuis atau cara kerja antarmuka:

### Lokasi File Penting
* **Daftar Pertanyaan Kuis**: Dikustomisasi di berkas [frontend/src/lib/quiz-config.ts](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/frontend/src/lib/quiz-config.ts).
  > [!WARNING]
  > Jika Anda menambah, menghapus, atau mengubah nama kunci (`key`) pertanyaan di berkas ini, Anda **wajib** menyesuaikannya dengan Pydantic Schema dan urutan fitur di Backend (`backend/app/main.py`), karena model ML membutuhkan 24 fitur dengan nama dan tipe data yang presisi.
* **Halaman Form Kuis**: Antarmuka halaman pengisian kuis berada di [frontend/src/app/quiz/page.tsx](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/frontend/src/app/quiz/page.tsx).
* **Halaman Hasil Prediksi & XAI**: Antarmuka visualisasi skor kesehatan, diagnosis tingkat risiko, dan daftar penjelasan XAI dikelola di [frontend/src/app/result/page.tsx](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/frontend/src/app/result/page.tsx).
* **Fungsi Pemanggilan API**: Fungsi untuk mengirim data jawaban kuis ke backend didefinisikan di [frontend/src/lib/api.ts](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/frontend/src/lib/api.ts).

### Menjalankan Frontend secara Lokal
1. Masuk ke folder frontend:
   ```bash
   cd frontend
   ```
2. Instal pustaka pendukung:
   ```bash
   npm install
   ```
3. Konfigurasi Endpoint Backend (buat berkas `.env.local` di dalam folder `frontend/`):
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```
4. Jalankan aplikasi web:
   ```bash
   npm run dev
   ```
   Buka browser pada [http://localhost:3000](http://localhost:3000).

---

## 🐍 2. Mengembangkan & Mengedit Backend (FastAPI)

Backend berfungsi sebagai API inferensi yang menerima jawaban kuis, melakukan normalisasi data (*scaling*), memprediksi probabilitas risiko menggunakan Random Forest, serta menyusun narasi penjelasan klinis (Explainable AI).

### Lokasi File Penting
* **Logika Utama API**: Seluruh rute API (`/` dan `/predict`), definisi skema input Pydantic, proses normalisasi data, serta logika teks penjelasan klinis (XAI) dan rekomendasi berada dalam berkas [backend/app/main.py](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/backend/app/main.py).
* **Dependensi Backend**: Tercantum di [backend/requirements.txt](file:///c:/Users/ACER/OneDrive/Dokumen/NHANES-Stroke-Dataset/backend/requirements.txt).

### Cara Mengubah Logika Explainable AI (XAI)
Penjelasan klinis dihitung berdasarkan respons jawaban spesifik pengguna pada fungsi `predict_stroke` di berkas `main.py`. 
Jika Anda ingin menambah atau mengubah aturan klinis, cari blok komentar `# 6. Logika Explainable AI (XAI) Dinamis` dan tambahkan aturan kondisional baru, contoh:
```python
if data.systolic_bp >= 140:
    explanations.append("Hipertensi Stadium 2: Tekanan darah sistolik Anda sangat tinggi...")
```

### Menjalankan Backend secara Lokal
1. Pastikan python 3.10+ telah terinstal. Pasang dependensi di root direktori:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Jalankan server lokal:
   ```bash
   python -m uvicorn backend.app.main:app --reload --port 8000
   ```
   Backend akan aktif di [http://localhost:8000](http://localhost:8000). Akses `/docs` untuk membuka dokumentasi Swagger API.

---

## 🚀 3. Mempublikasikan Pembaruan (Deployment)

### Cara Deploy Update Frontend (Next.js)
Karena terhubung dengan **Vercel**, setiap kali Anda melakukan `git push` perubahan ke repositori GitHub utama Anda (`origin/main`), Vercel akan otomatis melakukan proses *rebuild* dan memperbarui website publik Anda dalam waktu beberapa menit.

### Cara Deploy Update Backend (Hugging Face Spaces)
Hugging Face memproses aplikasi backend menggunakan Docker. Karena Hugging Face **menolak berkas biner besar** (seperti dataset `.xpt` di folder `data/raw`), kita tidak bisa mem-push branch utama secara langsung ke Hugging Face.

Ikuti prosedur berikut di terminal komputer Anda untuk mem-push pembaruan backend dengan aman:

1. **Buat Branch Bersih Sementara (Orphan Branch)**:
   ```powershell
   git checkout --orphan deploy-hf
   ```
2. **Hapus Berkas Mentah yang Tidak Diperlukan API**:
   ```powershell
   git rm -rf data/raw reports notebooks experiments docs
   ```
3. **Inisialisasi & Lacak Model Biner dengan Git LFS**:
   ```powershell
   git lfs install
   git lfs track "*.pkl"
   git add .gitattributes
   git add .
   ```
4. **Komit & Force Push ke Hugging Face**:
   ```powershell
   git commit -m "deploy: update backend"
   git push hf deploy-hf:main --force
   ```
   *(Gunakan Access Token Hugging Face Anda dengan otorisasi WRITE sebagai password).*
5. **Kembali ke Branch Utama & Bersihkan Branch Sementara**:
   ```powershell
   git checkout main
   git branch -D deploy-hf
   ```

Setelah langkah ini selesai, Hugging Face secara otomatis akan membangun ulang (*building*) kontainer Docker baru Anda di internet.
