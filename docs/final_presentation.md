# Draft Konten Slide Presentasi Akhir

**Judul Proyek:** Penerapan Explainable Machine Learning untuk Prediksi Risiko Stroke Berbasis Faktor Gaya Hidup  
**Dataset:** NHANES 2015-2016

---

## Slide 1: Cover / Halaman Judul
**[Visual/Desain]**
- Logo Universitas Sebelas Maret (UNS).
- Judul dibuat besar, tegas, dan bersih.
- Nama anggota kelompok di bagian bawah.

**[Teks di Slide]**
**PENERAPAN EXPLAINABLE MACHINE LEARNING UNTUK PREDIKSI RISIKO STROKE BERBASIS FAKTOR GAYA HIDUP**

**Disusun oleh:**
- Fadhil Rusadi (L0124013)
- Firizqi Aditya Mulya (L0124016)
- Nurman Aqil Wicaksono (L0124139)

**Dosen:** Prof. Dr. Wiharto, S.T., M.Kom.
**Program Studi Informatika, Fakultas Teknologi Informasi dan Sains Data, Universitas Sebelas Maret**

---

## Slide 2: Latar Belakang Masalah
**[Visual/Desain]**
- Ikon medis atau ilustrasi otak/pembuluh darah.
- Tiga poin utama ditampilkan singkat dan visual.

**[Teks di Slide]**
**Mengapa Topik Ini Penting?**
1. **Stroke** merupakan penyebab kematian dan disabilitas jangka panjang yang tinggi, sehingga deteksi dini sangat penting.
2. **Faktor gaya hidup** seperti kualitas tidur, stres, dan aktivitas fisik sering belum dimanfaatkan optimal dalam skrining risiko.
3. **Model machine learning** modern dapat akurat, tetapi sering bersifat *black box* sehingga kurang mudah dipercaya pengguna.

---

## Slide 3: Dataset & Preprocessing
**[Visual/Desain]**
- Flowchart: Dataset mentah -> Split -> Imputasi -> Standardisasi -> SMOTE.
- Diagram singkat penggabungan 10 modul NHANES.

**[Teks di Slide]**
**Alur Pengolahan Data**

- **Dataset:** NHANES 2015-2016, digabung dari 10 modul survei berdasarkan `SEQN`.
- **Split data:** Stratified train-test split 80:20.
- **Imputasi:** Median untuk numerik, modus untuk kategorikal/biner, dan 0 untuk variabel aktivitas fisik tertentu.
- **Standardisasi:** `StandardScaler` hanya di-fit pada data training.
- **SMOTE:** Diterapkan hanya pada data training untuk menangani *class imbalance* dan mencegah *data leakage*.

---

## Slide 4: Feature Selection & Desain Eksperimen
**[Visual/Desain]**
- Tabel kecil perbandingan 31 fitur menjadi 24 fitur.
- Diagram 5 skenario eksperimen.

**[Teks di Slide]**
**Seleksi Fitur dan Skenario Uji**

- Seleksi fitur dilakukan menggunakan:
  - **Chi-Square** untuk fitur kategorikal.
  - **ANOVA F-Test** untuk fitur kontinu.
- Dari **31 fitur awal**, tersisa **24 fitur signifikan**.
- Tujuh fitur yang dieliminasi dianggap kurang informatif untuk model akhir.
- Tiga model diuji:
  - Decision Tree
  - Random Forest
  - XGBoost
- Lima skenario eksperimen:
  - A) Klinis saja
  - B) Klinis + Tidur
  - C) Klinis + Stres
  - D) Klinis + Aktivitas Fisik
  - E) Klinis + Semua Gaya Hidup

---

## Slide 5: Hasil Eksperimen & Threshold
**[Visual/Desain]**
- Grafik AUC-ROC atau tabel ringkas hasil model.
- Kurva ROC untuk model terbaik.

**[Teks di Slide]**
**Model Terbaik dan Ambang Keputusan**

- **Model terbaik:** Random Forest pada Skenario E.
- **AUC-ROC:** 0.7678.
- **Threshold default 0.50** menghasilkan recall yang masih rendah.
- **Threshold optimal 0.2344** dipilih dengan **Youden's J-Statistic**.
- Dampak utama:
  - Recall naik signifikan menjadi **77.14%**
  - Model lebih aman untuk skrining karena mengurangi risiko *false negative*

---

## Slide 6: Explainable AI (SHAP)
**[Visual/Desain]**
- Bar chart SHAP global.
- Beeswarm plot SHAP.

**[Teks di Slide]**
**Interpretasi Model dengan SHAP**

- **SHAP** digunakan untuk menjelaskan kontribusi setiap fitur terhadap prediksi.
- Kelompok fitur paling dominan:
  1. **Klinis**
  2. **Kualitas Tidur**
  3. **Aktivitas Fisik**
  4. **Stres**
- Fitur penting yang sering muncul:
  - Usia
  - Hipertensi
  - Sleep apnea
  - Riwayat merokok
  - Aktivitas fisik rendah

---

## Slide 7: Aplikasi Web
**[Visual/Desain]**
- Screenshot dashboard aplikasi.
- Highlight gauge risiko dan panel penjelasan SHAP lokal.

**[Teks di Slide]**
**Implementasi Sistem Interaktif**

- **Frontend:** Next.js di Vercel.
- **Backend:** FastAPI di Hugging Face Spaces.
- Aplikasi menampilkan:
  - **Risk Gauge** untuk probabilitas risiko stroke
  - **Penjelasan SHAP lokal** secara real-time
  - **Rekomendasi** berdasarkan profil pengguna
- Tujuan utama: membuat hasil model lebih mudah dipahami dan lebih siap digunakan.

---

## Slide 8: Kesimpulan
**[Visual/Desain]**
- Ringkasan 3 poin utama.
- Tautan demo dan repository.

**[Teks di Slide]**
**Kesimpulan**

1. Integrasi faktor gaya hidup dengan data klinis membantu meningkatkan kualitas prediksi risiko stroke.
2. Seleksi fitur dan penentuan threshold membuat model lebih efektif untuk skenario skrining.
3. SHAP membantu mengubah model *black box* menjadi sistem yang lebih transparan dan dapat dijelaskan.

**Live Demo:** [nhanes-stroke-dataset.vercel.app](https://nhanes-stroke-dataset.vercel.app)  
**Repository:** GitHub proyek NHANES Stroke Dataset

***TERIMA KASIH***
