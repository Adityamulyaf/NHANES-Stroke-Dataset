# Slide Presentasi - Progress Week 3
**Proyek:** Analisis Faktor Gaya Hidup pada Prediksi Risiko Stroke menggunakan Explainable Machine Learning  
**Topik:** Optimalisasi Feature Selection, Pengembangan Web App, dan Deployment  

---

### Slide 1: Cover (Halaman Judul)
*   **PROGRESS WEEK 3: OPTIMALISASI FITUR & DEPLOYMENT APLIKASI**
*   *Integrasi Sistem Prediksi Risiko Stroke Berbasis Explainable Machine Learning (NHANES 2015-2016)*
*   **Oleh Kelompok:**
    *   Fadhil Rusadi (L0124013)
    *   Firizqi Aditya Mulya (L0124016)
    *   Nurman Aqil Wicaksono (L0124139)

---

### Slide 2: Agenda Progress & Pencapaian
*   **Poin Utama Progress Week 3:**
    1.  **Optimalisasi Fitur (Feature Selection):** Transisi dari pemilihan manual (31 fitur) ke seleksi statistik (24 fitur).
    2.  **Pengembangan Web App:** Pembuatan API backend (FastAPI) dan antarmuka interaktif (Next.js).
    3.  **Integrasi Model & Explainable AI (XAI):** Kalibrasi ambang batas (*threshold*) probabilitas dan penerapan aturan klinis dinamis.
    4.  **Deployment Publik:** Publikasi frontend di Vercel dan backend kontainer Docker di Hugging Face Spaces.

---

### Slide 3: Optimalisasi Feature Selection (Seleksi Fitur)
*   **Perbedaan Pendekatan:**
    *   *Sebelumnya:* 31 Fitur dipilih manual berdasarkan literatur subjektif (masih banyak *noise*).
    *   *Sekarang:* 24 Fitur terpilih secara statistik melalui seleksi univariat terstandarisasi dengan signifikansi $p < 0.05$.
*   **Metode Seleksi Statistik:**
    *   **Uji Chi-Square:** Untuk fitur kategorikal (riwayat penyakit, status merokok, dll).
    *   **ANOVA F-Test:** Untuk fitur kontinu (usia, tekanan darah, durasi aktivitas harian).
*   **Fitur Noise yang Dieliminasi (7 Fitur):**
    *   *Klinis:* Jenis Kelamin (`gender`), Ras (`race`), BMI (`bmi`), Perokok Aktif (`current_smoker`).
    *   *Tidur:* Durasi Tidur (`sleep_hours`).
    *   *Aktivitas Fisik:* Kerja Fisik Berat (`vigorous_work`), Menit Sedentary (`sedentary_min`).

---

### Slide 4: Dampak Seleksi Fitur terhadap Kinerja Model
*   **Perbandingan AUC-ROC Sebelum vs Sesudah Feature Selection:**
    | Skenario Eksperimen | Model Terbaik | AUC-ROC (31 Fitur) | AUC-ROC (24 Fitur) | Selisih Performa |
    | :--- | :--- | :---: | :---: | :---: |
    | **Skenario A (Klinis)** | XGBoost | 0.6961 | **0.8232** | **+0.1271** (Naik Masif!) |
    | **Skenario B (Klinis + Tidur)** | XGBoost | 0.7375 | **0.8139** | **+0.0764** |
    | **Skenario C (Klinis + Stres)** | XGBoost | 0.7361 | **0.8064** | **+0.0703** |
    | **Skenario D (Klinis + Aktivitas)** | XGBoost | 0.7007 | **0.7831** | **+0.0824** |
*   **Efek Optimasi Dimensi:**
    *   Peningkatan kualitas SMOTE di ruang fitur yang lebih sempit tanpa terganggu variabel *noise*.
    *   Lonjakan **Tuned F1-Score Skenario C** sebesar **+81.5%** (dari 0.1157 menjadi **0.2101**) menggunakan model XGBoost.

---

### Slide 5: Arsitektur Aplikasi & API Contract
*   **Alur Arsitektur Sistem (Stateless):**
    `User Input (Quiz UI Next.js)` $\rightarrow$ `POST /predict (JSON Data)` $\rightarrow$ `FastAPI Backend (Preprocessing & Inference)` $\rightarrow$ `JSON Response (Result & XAI Explanation)`
*   **API Contract (`POST /predict`):**
    *   *Request Body:* JSON dengan 24 nilai parameter kesehatan & gaya hidup.
    *   *Response Body:*
        *   `prediction`: Kategori risiko (`"high_risk"` / `"low_risk"`).
        *   `probability`: Nilai desimal probabilitas prediksi stroke (`0.0` s.d `1.0`).
        *   `explanation`: Kumpulan teks alasan medis individual (array of strings).
        *   `recommendation`: Rekomendasi gaya hidup & medis praktis (string).

---

### Slide 6: Backend ML: Penanganan Imbalance & Explainable AI (XAI)
*   **Penalaan Threshold Optimal (Youden's J-Statistic):**
    *   *Masalah:* Kasus stroke sangat langka pada populasi NHANES (~3.6%). Threshold default `0.50` menghasilkan Recall buruk (**17.14%**).
    *   *Solusi:* Menurunkan ambang batas keputusan probabilitas menjadi **`0.2344`**.
    *   *Hasil:* Sensitivitas/Recall model naik drastis menjadi **`77.14%`** pada Random Forest (sangat aman untuk tes penapisan medis awal).
*   **Logika Explainable AI (XAI) Dinamis:**
    *   Menghitung penjelasan berbasis aturan klinis berdasarkan profil jawaban personal.
    *   *Contoh:* Sistolik $\ge 130$ mmHg $\rightarrow$ Trigger penjelasan bahaya hipertensi merusak arteri. PHQ-9 $\ge 8 \rightarrow$ Trigger penjelasan dampak stres kronis pada beban kardiovaskular.

---

### Slide 7: Tampilan Antarmuka Frontend (Next.js)
*   **Komponen UI Interaktif:**
    *   **Kuis Bertahap (Multi-step Quiz):** Pertanyaan 24 fitur dipecah berdasarkan kategori (Klinis, Tidur, Mental, Aktivitas Fisik) agar nyaman diisi.
    *   **Dashboard Hasil Visual:**
        *   *Risk Gauge:* Indikator visual persentase tingkat risiko stroke.
        *   *XAI Accordion:* Tampilan kartu drop-down dinamis yang mengurai detail faktor risiko individu pengguna.
        *   *Actionable Advice:* Rekomendasi medis terarah sesuai tingkat risiko yang diperoleh.

---

### Slide 8: Strategi Deployment & CI/CD
*   **Frontend (Next.js):**
    *   Dihosting di **Vercel** ([nhanes-stroke-dataset.vercel.app](https://nhanes-stroke-dataset.vercel.app)).
    *   Continuous Delivery otomatis setiap kali kode di-push ke branch `main`.
*   **Backend (FastAPI & Docker):**
    *   Dihosting di **Hugging Face Spaces** ([huggingface.co/spaces/adityamulyaf/nhanes-stroke-api](https://huggingface.co/spaces/adityamulyaf/nhanes-stroke-api)).
    *   *Mengatasi Batas Ukuran File:* Menggunakan **Git Orphan Branch (`deploy-hf`)** untuk mengunggah berkas bersih tanpa data mentah CDC (.xpt) yang besar, dan melacak model biner (`.pkl`) menggunakan **Git LFS**.

---

### Slide 9: Demo Aplikasi & Rencana Selanjutnya
*   **Tautan Live Demo:**
    *   *Web App:* [nhanes-stroke-dataset.vercel.app](https://nhanes-stroke-dataset.vercel.app)
    *   *Dokumentasi API Swagger:* [adityamulyaf-nhanes-stroke-api.hf.space/docs](https://adityamulyaf-nhanes-stroke-api.hf.space/docs)
*   **TERIMA KASIH - SESI TANYA JAWAB**

