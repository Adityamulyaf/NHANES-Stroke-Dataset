# Draft Konten Slide Presentasi Akhir (PPT)

**Judul Proyek:** Penerapan Explainable Machine Learning untuk Prediksi Risiko Stroke Berbasis Faktor Gaya Hidup
**Dataset:** NHANES 2015-2016

---

## Slide 1: Cover / Halaman Judul
**[Visual/Desain]**
- Logo Universitas Sebelas Maret (UNS).
- Judul Proyek dengan *font* yang jelas dan tegas.
- Nama Anggota Kelompok.

**[Teks di Slide]**
**PENERAPAN EXPLAINABLE MACHINE LEARNING UNTUK PREDIKSI RISIKO STROKE BERBASIS FAKTOR GAYA HIDUP**

**Disusun oleh:**
- Fadhil Rusadi (L0124013)
- Firizqi Aditya Mulya (L0124016)
- Nurman Aqil Wicaksono (L0124139)

**Dosen:** Prof. Dr. Wiharto, S.T., M.Kom.

---

## Slide 2: Latar Belakang & Permasalahan
**[Visual/Desain]**
- Ikon medis atau representasi otak / jantung.
- Tiga poin utama dibuat berbentuk *bullet points* dengan ikon di masing-masing poin.

**[Teks di Slide]**
**Mengapa Proyek Ini Penting?**
1. **Clinical vs Lifestyle:** Deteksi dini seringkali hanya fokus pada riwayat klinis dan mengabaikan faktor keseharian (kualitas tidur, stres, sedentari).
2. **Class Imbalance:** Pasien stroke jauh lebih sedikit dari pasien sehat (~3.6%), menyebabkan bias mayoritas (Model sering *False Negative*).
3. **Black Box AI:** Algoritma *machine learning* modern sangat rumit. Dokter dan pengguna butuh alasan rasional dan transparan di balik hasil prediksi.

---

## Slide 3: Metodologi & Preprocessing
**[Visual/Desain]**
- Flowchart singkat alur pemrosesan data (Stratified Split $\rightarrow$ Imputasi $\rightarrow$ Standardisasi $\rightarrow$ SMOTE).
- Tabel perbandingan jumlah fitur sebelum dan sesudah seleksi.

**[Teks di Slide]**
**Alur Pemrosesan & Optimalisasi Fitur**

- **Dataset:** NHANES (2015-2016).
- **Seleksi Fitur (Statistik Univariat):**
  - Uji Chi-Square (Kategorikal) & ANOVA F-Test (Kontinu).
  - Mengurangi dari **31 Fitur $\rightarrow$ 24 Fitur**.
  - Mengeliminasi fitur *noise* (contoh: Ras, BMI) tanpa mengurangi informasi krusial.
- **SMOTE (Oversampling):** Mengatasi *class imbalance* hanya pada *data training* untuk mencegah *data leakage*.

---

## Slide 4: Hasil Eksperimen & Penalaan Threshold
**[Visual/Desain]**
- Grafik Batang / Tabel yang menyoroti Random Forest sebagai model terbaik.
- Kurva ROC Skenario E (AUC-ROC: 0.7678).

**[Teks di Slide]**
**Performa Model & Ambang Batas Medis**

- **Model Terbaik:** Random Forest Skenario E (24 Fitur Gabungan: Klinis + Tidur + Aktivitas Fisik + Stres).
- **AUC-ROC:** 0.7678.
- **Penalaan Youden's J-Statistic Threshold (0.2344):**
  - *Threshold default (0.50):* Recall hanya 17.14%.
  - **Threshold optimal (0.2344): Recall meroket menjadi 77.14%.**
- *Implikasi:* Model jauh lebih sensitif mendeteksi penderita stroke demi meminimalkan kelalaian penanganan medis (*false negative*).

---

## Slide 5: Explainable AI (SHAP) & Analisis Global
**[Visual/Desain]**
- Gambar/Grafik *SHAP Global Feature Importance* atau *Beeswarm Plot* (Gambar 3.4 / 3.5 dari Laporan Akhir).

**[Teks di Slide]**
**Membuka "Kotak Hitam" AI dengan SHAP**

- **SHAP (SHapley Additive exPlanations):** Alokasi kontribusi yang adil untuk setiap fitur berdasarkan teori permainan kooperatif.
- **Top Faktor Risiko (Global):**
  1. Usia (*Age*)
  2. Riwayat Hipertensi (*Hypertension*)
  3. Keluhan Henti Napas saat Tidur (*Sleep Apnea*)
  4. Kurang Olahraga / Sedentari (*Moderate/Vigorous Leisure*)
  5. Stres Psikologis

---

## Slide 6: Demo Web App & SHAP Lokal
**[Visual/Desain]**
- *Screenshot* Dashboard Hasil dari Aplikasi MediTrust (menampilkan *Risk Gauge* dan *Dual-Direction SHAP Bar Chart* dari Gambar 3.7 & 3.8 laporan akhir).

**[Teks di Slide]**
**Implementasi Sistem & Antarmuka Interaktif**

- **Frontend:** Next.js (Vercel) | **Backend:** FastAPI (Hugging Face).
- **Risk Gauge:** Probabilitas prediksi dengan indikator batas Youden (23.44%).
- **Bagan Batang SHAP Dua Arah (Real-Time):**
  - **Batang Merah:** Mendorong/menaikkan risiko stroke.
  - **Batang Hijau:** Mengurangi/melindungi dari risiko stroke.
- *Transparan, Tepercaya, dan Edukatif untuk Pengguna Independen.*

---

## Slide 7: Kesimpulan
**[Visual/Desain]**
- Ringkasan 3 Poin *Takeaway*.
- Link Aplikasi / GitHub.

**[Teks di Slide]**
**Kesimpulan & Penutup**

1. **Holistik:** Integrasi parameter gaya hidup secara kolektif meningkatkan deteksi stroke.
2. **Sensitif & Akurat:** Optimasi statistik (Seleksi Fitur) & Threshold 0.2344 mengamankan metrik Recall medis.
3. **Transparan (XAI):** SHAP secara efektif mengubah algoritma "Kotak Hitam" menjadi alasan yang bisa dipahami (*Actionable Advice*).

**Live Website:** [nhanes-stroke-dataset.vercel.app](https://nhanes-stroke-dataset.vercel.app)
**Repository:** Tersedia secara publik di GitHub.

***TERIMA KASIH - SESI TANYA JAWAB***
