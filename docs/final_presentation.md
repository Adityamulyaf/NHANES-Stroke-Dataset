# Draft Konten Slide Presentasi Akhir - Terintegrasi Lengkap (Berdasarkan Laporan & AI Reference PDF)

**Judul Proyek:** Analisis Faktor Gaya Hidup pada Prediksi Risiko Stroke Menggunakan Explainable Machine Learning  
**Dataset:** NHANES 2015-2016

---

## Slide 1: Cover / Halaman Judul
**[Visual/Desain]**
- Logo Universitas Sebelas Maret (UNS).
- Judul dibuat besar, tegas, bersih, dan menggunakan font modern.
- Nama anggota kelompok di bagian bawah.

**[Teks di Slide]**
**ANALISIS FAKTOR GAYA HIDUP PADA PREDIKSI RISIKO STROKE MENGGUNAKAN EXPLAINABLE MACHINE LEARNING**

**Disusun oleh:**
- Fadhil Rusadi (L0124013)
- Firizqi Aditya Mulya (L0124016)
- Nurman Aqil Wicaksono (L0124139)

**Dosen Pengampu:** Prof. Dr. Wiharto, S.T., M.Kom.  
**Program Studi Informatika, Fakultas Teknologi Informasi dan Sains Data, Universitas Sebelas Maret**

---

## Slide 2: Latar Belakang & Research Gap
**[Visual/Desain]**
- Pembagian visual dua kolom: Kolom Kiri (Fakta Masalah) dengan ikon/foto medis, Kolom Kanan (Research Gap) dengan warna latar kontras.

**[Teks di Slide]**
- **Fakta Masalah:**  
  Stroke merupakan salah satu penyebab utama kematian dan kecacatan di dunia. Identifikasi risiko stroke sejak dini dapat membantu pencegahan dan pengambilan keputusan klinis yang lebih efektif.
- **Research Gap:**  
  Walaupun pola tidur, kondisi mental, dan aktivitas fisik secara medis berkaitan dengan risiko stroke, kontribusi masing-masing kelompok faktor gaya hidup terhadap performa model prediksi masih belum dievaluasi secara komprehensif dan terukur.

---

## Slide 3: Penelitian Lain & Inovasi
**[Visual/Desain]**
- Dua kolom: Kolom Kiri berisi daftar pustaka penelitian lain, Kolom Kanan berisi poin inovasi yang diusung dengan penyorotan warna.

**[Teks di Slide]**
- **Penelitian Lain:**
  - **Shobayo et al. (2023):** *Prediction of Stroke Disease with Demographic and Behavioural Data Using Random Forest Algorithm*.
  - **Huang & Liu (2025):** *Development and validation of a machine learning model to predict stroke risk based on the NHANES database*.
  - **Nguyen et al. (2022):** *Lifestyle practices and associated factors among adults with hypertension: Conquering Hypertension in Vietnam-solutions at the grassroots level study*.
- **Inovasi Proyek Kami:**
  - Membandingkan 3 algoritma Machine Learning (Decision Tree, Random Forest, XGBoost) pada 5 skenario kombinasi fitur.
  - Mengukur kontribusi masing-masing kelompok gaya hidup (tidur, stres, aktivitas fisik) secara independen.
  - Menyediakan interpretasi hasil prediksi secara lokal & global menggunakan **SHAP (Explainable AI)** secara real-time di web app.

---

## Slide 4: Rumusan Masalah
**[Visual/Desain]**
- Layout grid berisi 5 pertanyaan masalah utama dengan penomoran yang rapi dan ikon interogatif.

**[Teks di Slide]**
**Rumusan Masalah Utama:**
1. Bagaimana performa Decision Tree, Random Forest, dan XGBoost dalam memprediksi risiko stroke menggunakan dataset NHANES?
2. Seberapa besar peningkatan performa model ketika variabel pola tidur ditambahkan ke fitur klinis standar?
3. Seberapa besar peningkatan performa model ketika variabel stres/depresi ditambahkan ke fitur klinis standar?
4. Seberapa besar peningkatan performa model ketika variabel aktivitas fisik ditambahkan ke fitur klinis standar?
5. Kelompok faktor gaya hidup manakah yang memberikan kontribusi terbesar terhadap prediksi risiko stroke berdasarkan analisis SHAP?

---

## Slide 5: Tujuan Penelitian
**[Visual/Desain]**
- Grid 2x2 yang memvisualisasikan 4 tujuan utama proyek dengan box warna-warni yang senada.

**[Teks di Slide]**
**Tujuan Penelitian Proyek:**
- **Membangun Model:** Mengembangkan model klasifikasi risiko stroke menggunakan Decision Tree, Random Forest, dan XGBoost berbasis data survei nasional NHANES.
- **Perbandingan Algoritma:** Membandingkan kinerja ketiga algoritma menggunakan metrik evaluasi Accuracy, Precision, Recall, F1-Score, dan AUC-ROC.
- **Evaluasi Gaya Hidup:** Mengevaluasi kontribusi masing-masing kelompok faktor gaya hidup (tidur, stres, aktivitas fisik) terhadap peningkatan performa prediksi model.
- **SHAP Analysis:** Mengidentifikasi kelompok fitur dan variabel gaya hidup yang paling signifikan memengaruhi prediksi stroke menggunakan metode SHAP (*SHapley Additive exPlanations*).

---

## Slide 6: Manfaat Penelitian
**[Visual/Desain]**
- Box berisi 3 manfaat utama dengan penataan spasi yang nyaman dibaca dan ikon pendukung.

**[Teks di Slide]**
**Manfaat yang Diharapkan:**
- Memberikan bukti empiris mengenai kontribusi masing-masing kelompok faktor gaya hidup (tidur, stres, aktivitas fisik) terhadap prediksi risiko stroke berbasis machine learning.
- Memberikan analisis interpretatif mengenai faktor-faktor yang memengaruhi keputusan model menggunakan metode Explainable AI (SHAP) untuk transparansi pengguna.
- Menjadi dasar pengembangan sistem skrining mandiri (*self-assessment*) risiko stroke dini yang mempertimbangkan faktor klinis dasar serta perilaku hidup sehari-hari yang dapat dimodifikasi.

---

## Slide 7: Metode Penelitian & Preprocessing
**[Visual/Desain]**
- Flowchart alur pemrosesan data (Dataset -> Split -> Imputasi -> Standardisasi -> SMOTE) untuk mencegah data leakage.

**[Teks di Slide]**
**Alur Pengolahan Data (Stateless & Terstruktur):**
- **Dataset:**CDC NHANES 2015-2016 (Modul: DEMO_I, MCQ_I, BMX_I, BPX_I, SLQ_I, DPQ_I, PAQ_I) digabung berdasarkan responden ID (`SEQN`).
- **Data Preprocessing:**
  - *Split Data:* Stratified train-test split 80:20 (menjaga distribusi kelas target).
  - *Imputasi:* Median (numerik kontinu), modus (kategorikal/biner), dan `0` untuk aktivitas fisik tertentu.
  - *Standardisasi:* `StandardScaler` (hanya di-fit pada data training).
  - *SMOTE:* Oversampling penyeimbangan kelas (~3.6% stroke vs ~96.4% sehat) *hanya dilakukan pada data training* untuk mencegah *data leakage*.

---

## Slide 8: Seleksi Fitur & Skenario Eksperimen
**[Visual/Desain]**
- Tabel kecil perbandingan jumlah fitur sebelum vs sesudah seleksi fitur statistik.
- Skema diagram alir 5 skenario kombinasi fitur.

**[Teks di Slide]**
**Optimalisasi Dimensi & Skenario Fitur**
- **Seleksi Fitur Statistik Univariat ($p < 0.05$):**
  - **Uji Chi-Square:** Untuk fitur kategorikal.
  - **Uji ANOVA F-Test:** Untuk fitur kontinu.
  - *Hasil:* Memangkas **7 fitur noise**, dari **31 fitur awal menjadi 24 fitur signifikan** (12 klinis, 4 tidur, 5 stres, 3 aktivitas fisik).
- **Desain 5 Skenario Kombinasi Fitur:**
  - Skenario A: Klinis saja (Baseline)
  - Skenario B: Klinis + Tidur
  - Skenario C: Klinis + Stres
  - Skenario D: Klinis + Aktivitas Fisik
  - Skenario E: Klinis + Semua Gaya Hidup

---

## Slide 9: Hasil Eksperimen & Threshold Tuning
**[Visual/Desain]**
- Grafik perbandingan AUC-ROC model atau kurva ROC untuk skenario terbaik (AUC-ROC Random Forest = 0.7678).
- Tabel perbandingan performa model terbaik pada threshold default 0.50 vs tuned threshold Youden's J-Statistic (0.2315).

**[Teks di Slide]**
**Model Terbaik dan Kalibrasi Threshold Medis**
- **Model Terbaik:** Random Forest pada Skenario E (24 Fitur Gabungan).
- **AUC-ROC:** 0.7678 (mengungguli XGBoost 0.7286 dan Decision Tree 0.6624).
- **Threshold Tuning (Youden's J-Statistic = 0.2315):**
  - *Threshold Default (0.50):* Recall hanya **24.32%** (melewatkan 75% kasus positif aktual).
  - *Threshold Optimal (0.2315):* **Recall meroket menjadi 75.68%** (Akurasi global stabil di 67.61%).
- *Implikasi:* Penurunan threshold meningkatkan sensitivitas skrining medis awal secara drastis demi meminimalkan kelalaian penanganan (*false negative*).

---

## Slide 10: Explainable AI (SHAP Global)
**[Visual/Desain]**
- Gambar/Chart *SHAP Global Feature Importance* (Gambar 3.4) dan *SHAP Beeswarm Plot* (Gambar 3.5).

**[Teks di Slide]**
**Membuka "Kotak Hitam" AI secara Global**
- **SHAP (SHapley Additive exPlanations):** Alokasi nilai kontribusi adil tiap fitur terhadap pergeseran output prediksi model.
- **Top 5 Faktor Risiko Global (Model Random Forest Skenario E):**
  1. **Usia (*Age*):** Faktor alami non-modifikasi terpenting (usia tua menaikkan risiko).
  2. **Riwayat Hipertensi (*Hypertension*):** Faktor klinis utama pemicu kerusakan arteri.
  3. **Henti Napas saat Tidur (*Sleep Apnea*):** Keluhan tidur paling berbahaya.
  4. **Kurang Olahraga / Sedentari:** Fitur protektif jika aktif, pemicu risiko jika diabaikan.
  5. **Stres Psikologis (PHQ-5):** Beban emosional kronis kardiovaskular.

---

## Slide 11: Demo Web App & SHAP Lokal (Real-Time)
**[Visual/Desain]**
- Tangkapan layar antarmuka dashboard website MediTrust (Gauge Batas Youden dan Horizontal SHAP Bar Chart dua arah).

**[Teks di Slide]**
**Implementasi Sistem & Antarmuka Interaktif**
- **Hosting & Platform:** Frontend Next.js (Vercel) $\rightarrow$ API Backend FastAPI (Hugging Face Spaces).
- **Risk Gauge:** Visualisasi persentase risiko prediksi dengan penanda batas Youden's Threshold (23.15%).
- **Horizontal SHAP Bar Chart (Dua Arah):**
  - *Batang Merah (Kanan):* Parameter yang menaikkan risiko (misal: Tekanan Darah Sistolik +3.6%).
  - *Batang Hijau (Kiri):* Parameter pelindung yang menurunkan risiko (misal: Olahraga Berat -14.0%).
- *Memberikan penjelasan yang transparan, tepercaya, dan edukatif bagi pengguna.*

---

## Slide 12: Kesimpulan & Referensi
**[Visual/Desain]**
- Ringkasan 3 poin kesimpulan utama.
- Link web demo dan repository.
- Daftar referensi utama.

**[Teks di Slide]**
**Kesimpulan:**
1. Integrasi faktor gaya hidup bersama data klinis secara kolektif meningkatkan keandalan deteksi dini stroke.
2. Pemangkasan fitur noise (24 fitur) & penalaan Youden's threshold (0.2315) mengamankan sensitivitas (Recall 75.68%) skrining awal.
3. SHAP secara interaktif menerjemahkan output model yang rumit menjadi alasan medis yang transparan dan dapat ditindaklanjuti oleh individu.

- **Live Demo:** [nhanes-stroke-dataset.vercel.app](https://nhanes-stroke-dataset.vercel.app)
- **Referensi Utama:**
  - Shobayo et al. (2023) | Huang & Liu (2025) | Nguyen et al. (2022) | Liu et al. (2022) | Lundberg & Lee (2017) | Chen & Guestrin (2016)

***TERIMA KASIH***
