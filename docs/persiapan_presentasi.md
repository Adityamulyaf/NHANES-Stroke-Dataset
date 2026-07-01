# 📋 Persiapan Presentasi Akhir — NHANES Stroke Prediction
**Proyek:** Analisis Faktor Gaya Hidup pada Prediksi Risiko Stroke Menggunakan Explainable Machine Learning

---

## Daftar Isi
1. [Alur Program dari Awal hingga Akhir](#1-alur-program-dari-awal-hingga-akhir)
2. [Library yang Digunakan & Fungsinya](#2-library-yang-digunakan--fungsinya)
3. [Mekanisme Semua Algoritma](#3-mekanisme-semua-algoritma)
4. [Daftar Pertanyaan yang Mungkin Muncul & Jawabannya](#4-daftar-pertanyaan-yang-mungkin-muncul--jawabannya)
5. [Cheat Sheet — Angka & Fakta Kunci](#5-cheat-sheet--angka--fakta-kunci)

---

## 1. Alur Program dari Awal hingga Akhir

```
[DATA MENTAH]                         [NOTEBOOK]                       [PRODUKSI]
NHANES CDC XPT  ──►  Preprocessing  ──►  Feature Selection  ──►  Training Model  ──►  Backend FastAPI  ──►  Frontend Next.js
(10 Modul .xpt)       (Notebook)          (Notebook)              (Notebook)          (main.py)             (Vercel)
```

### Fase 1 — Konstruksi Dataset (preprocessing_nhanes.ipynb)
1. **Load 10 Modul XPT dari CDC NHANES 2015-2016** menggunakan `pyreadstat`/`pandas`.
   - `DEMO_I` (Demografi), `MCQ_I` (Riwayat Medis), `BMX_I` (Fisik), `BPX_I` (Tekanan Darah)
   - `SLQ_I` (Tidur), `DPQ_I` (Depresi PHQ-9), `PAQ_I` (Aktivitas Fisik), dll.
2. **Merge/Gabung** seluruh dataset berdasarkan kunci ID responden `SEQN`.
3. **Recoding** variabel: mengubah kode angka NHANES ke nilai biner atau kategorikal yang bermakna.
4. **Definisi variabel target**: `stroke` = 1 jika responden pernah didiagnosis stroke, 0 jika tidak.
5. **Ekspor** dataset yang telah digabung dan dikode ke format `.csv`.

### Fase 2 — Feature Selection (feature_selection.ipynb)
1. **Stratified Train-Test Split (80:20)** — split dilakukan di awal sebelum apapun untuk mencegah *data leakage*.
2. **Imputasi Missing Value** pada data training:
   - Numerik kontinu → **Median**
   - Kategorikal/biner → **Modus**
   - Menit olahraga (jika tidak berolahraga) → **0**
3. **Standardisasi** menggunakan `StandardScaler` → fit HANYA di training set, transform di test set.
4. **SMOTE** (*Synthetic Minority Over-sampling Technique*) → diterapkan HANYA di training set.
5. **Uji Chi-Square** → untuk fitur kategorikal (p < 0.05 terhadap `stroke`).
6. **ANOVA F-Test** → untuk fitur kontinu (p < 0.05 terhadap `stroke`).
7. **Hasil**: 7 fitur noise dieliminasi → sisa **24 fitur signifikan** disimpan ke `feature_groups.pkl`.

### Fase 3 — Training Model (training_nhanes.ipynb)
1. **5 Skenario Eksperimen** menggunakan 24 fitur:
   - Skenario A: Klinis (12 fitur)
   - Skenario B: Klinis + Tidur (16 fitur)
   - Skenario C: Klinis + Stres (17 fitur)
   - Skenario D: Klinis + Aktivitas Fisik (15 fitur)
   - Skenario E: Klinis + Semua Gaya Hidup (24 fitur)
2. **Training 3 Model** per skenario: Decision Tree, Random Forest, XGBoost.
3. **Evaluasi** menggunakan: Accuracy, Precision, Recall, F1-Score, AUC-ROC.
4. **Threshold Tuning** → Youden's J-Statistic → threshold optimal = `0.2344`.
5. **SHAP Analysis** → Global Feature Importance & Beeswarm Plot.
6. **Simpan artefak terbaik**: `rf_best_model.pkl` dan `scaler.pkl` ke folder `models/`.

### Fase 4 — Backend API (backend/app/main.py)
1. **Startup**: Load `rf_best_model.pkl`, `scaler.pkl`, `feature_groups.pkl`.
2. **Inisialisasi SHAP**: `shap.TreeExplainer(model)`.
3. **Endpoint `POST /predict`** menerima 24 nilai JSON dari frontend.
4. **Preprocessing real-time**: urutan fitur → `scaler.transform()`.
5. **Prediksi**: `model.predict_proba()` → ambil probabilitas kelas 1 (stroke).
6. **Threshold**: jika `proba >= 0.2344` → `"high_risk"`, jika tidak → `"low_risk"`.
7. **SHAP Lokal**: hitung kontribusi tiap fitur secara real-time.
8. **Kirim respons JSON** ke frontend.

### Fase 5 — Frontend Web App (Next.js / Vercel)
1. Pengguna mengisi **Multi-step Quiz** (24 pertanyaan berbasis 4 kategori, dikonfigurasi di `quiz-config.ts`).
2. Jawaban user dikumpulkan sebagai objek `Record<string, number>`.
3. Fungsi `mapAnswers()` di `api.ts` mengonversi nilai indeks quiz ke nilai numerik NHANES yang sesungguhnya:
   - `education` (indeks 0–4) → nilai NHANES `DMDEDUC2` (1–5)
   - `income_ratio` (indeks 0–4) → nilai PIR float (0.5 – 5.0)
   - `snoring_freq` & `sleep_apnea` (indeks 0–4) → nilai NHANES (1–5, dengan `+1`)
   - `daytime_sleepy` (indeks 0–3) → nilai NHANES (1–4, dengan `+1`)
4. Fungsi `predict()` mengirim **`POST /predict`** ke Hugging Face Spaces backend dengan payload JSON 24 fitur.
5. Respons JSON disimpan sementara di **`sessionStorage`** (bukan `localStorage`) dengan key `"predict_result"`.
6. Browser di-redirect ke halaman `/result` — data di-load dari `sessionStorage` dan ditampilkan:
   - **Skor Kesehatan** (animasi `easeOutQuart` selama 1.5 detik) = `100 - probability%`
   - **Risk Threshold Slider** dengan penanda batas merah di **23.44%** (Youden's J).
   - **Dual-Direction SHAP Bar Chart** — top 6 fitur diurutkan berdasarkan `|nilai SHAP|` terbesar.
   - **Penjelasan Klinis** & **Rekomendasi** gaya hidup dari backend.

---

### Detail Alur Data Frontend (Lengkap)

```
[quiz-config.ts]          [Halaman /quiz]           [api.ts]                [Backend FastAPI]
Konfigurasi 24 Soal  →   User isi jawaban    →   mapAnswers()         →   POST /predict
(key, label, type,        (Record<key,index>)      (konversi indeks        (24 field JSON)
 scaleLabels, unit)                                ke nilai NHANES)
                                                        |
                                                   predict() ──fetch──►  Response JSON:
                                                                         { prediction,
[Halaman /result]   ◄── sessionStorage.getItem ◄── sessionStorage          probability,
 normalizeResult()        ("predict_result")       .setItem(result)        explanation[],
 (validasi tipe data)                                                      recommendation,
                                                                           shap_contributions[] }
     |
     ▼
 Tampil UI:
 - Skor Kesehatan (progress bar animasi)
 - Risk Slider (threshold 23.44%)
 - SHAP Bar Chart (top 6, dua arah)
 - Penjelasan Klinis & Rekomendasi
```

---

## 2. Library yang Digunakan & Fungsinya

### Python — Machine Learning & Data Science

| Library | Fungsi Utama dalam Proyek |
|---|---|
| **pandas** | Load file `.xpt`, merge dataset antar modul, manipulasi DataFrame. |
| **numpy** | Operasi array numerik, kalkulasi probabilitas & SHAP values. |
| **scikit-learn** | `StandardScaler`, `StratifiedShuffleSplit`, `DecisionTreeClassifier`, `RandomForestClassifier`, `chi2`, `f_classif`, metrik evaluasi (AUC-ROC, F1, dll). |
| **imbalanced-learn** | `SMOTE` — Oversampling sintetis untuk menyeimbangkan kelas stroke (3.6%) vs non-stroke (96.4%). |
| **xgboost** | `XGBClassifier` — Extreme Gradient Boosting, model ensemble berbasis boosting. |
| **shap** | `TreeExplainer` — Menghitung nilai kontribusi (SHAP values) dari setiap fitur terhadap keputusan model. |
| **joblib** | Serialisasi & deserialisasi model (`.pkl`) — menyimpan dan memuat model terlatih. |
| **matplotlib / seaborn** | Visualisasi: kurva ROC, heatmap AUC, bar plot perbandingan, grafik SHAP. |

### Python — Backend API

| Library | Fungsi Utama dalam Proyek |
|---|---|
| **fastapi** | Framework web API modern Python. Mendefinisikan endpoint `GET /` dan `POST /predict`. |
| **uvicorn** | ASGI server — menjalankan aplikasi FastAPI di production/lokal. |
| **pydantic** | Validasi otomatis tipe data input (class `StrokeInput` dengan 24 field bertipe `float`). |

### JavaScript/TypeScript — Frontend

| Library / API | Fungsi Utama dalam Proyek |
|---|---|
| **Next.js 15** | Framework React dengan App Router. Membangun halaman `/` (landing), `/quiz` (kuis multi-step), dan `/result` (halaman hasil prediksi). |
| **TypeScript** | Memberikan type-safety pada seluruh kode frontend. Mendefinisikan tipe `PredictResponse`, `QuizQuestion`, `QuizSection`, dll. |
| **Tailwind CSS** | Styling komponen UI secara utility-first. Semua class seperti `rounded-2xl`, `bg-teal-light`, `flex`, `gap-6` berasal dari Tailwind. |
| **Fetch API (native)** | `fetch()` bawaan browser digunakan di `api.ts` untuk mengirim `POST /predict` ke backend — tidak memerlukan library tambahan seperti `axios`. |
| **sessionStorage (Web API)** | Menyimpan sementara hasil prediksi JSON saat berpindah halaman dari `/quiz` ke `/result`. Dipilih karena data hanya diperlukan selama sesi browser. |
| **requestAnimationFrame (Web API)** | Digunakan di halaman `/result` untuk animasi progress bar Skor Kesehatan menggunakan fungsi easing `easeOutQuart` selama 1500ms. |

### Struktur Response API dari Backend (5 Field)

```json
{
  "prediction": "high_risk" | "low_risk",
  "probability": 0.0 – 1.0,
  "explanation": ["String penjelasan klinis 1", "String 2", ...],
  "recommendation": "String rekomendasi lengkap",
  "shap_contributions": [
    { "feature": "age", "value": 0.045 },
    { "feature": "systolic_bp", "value": 0.036 },
    ...
  ]
}
```

> ⚠️ **Catatan penting**: `probability` dikirim sebagai desimal (0.0–1.0). Frontend mengalikan `× 100` untuk tampilan persen. Jika backend mengirim dalam format persen (25.0), maka `Math.min(1, 25.0) * 100 = 100%` — gauge akan selalu menunjukkan 100%!

---

## 3. Mekanisme Semua Algoritma

### 3.1 SMOTE (Synthetic Minority Over-sampling Technique)
- **Masalah yang Diselesaikan**: Kelas stroke hanya ~3.6% data — model akan bias memprediksi semua orang sehat.
- **Cara Kerja**:
  1. Untuk setiap sampel minoritas (stroke = 1), cari **K tetangga terdekat** dari sesama kelas minoritas.
  2. Pilih tetangga secara acak, lalu **buat titik sintetis** di sepanjang garis yang menghubungkan sampel asli dan tetangganya.
  3. Hasilkan sampel baru hingga jumlah kelas seimbang.
- **Penting**: SMOTE **HANYA diterapkan pada data training**, bukan data uji — agar evaluasi model mencerminkan distribusi populasi riil.

### 3.2 Feature Selection — Uji Chi-Square (χ²)
- **Digunakan untuk**: Fitur **kategorikal** (biner: ya/tidak, dll).
- **Cara Kerja**: Mengukur apakah distribusi kategori suatu fitur berbeda secara signifikan antara kelompok stroke dan non-stroke.
  - χ² = Σ((Observasi - Ekspektasi)² / Ekspektasi)
- **Interpretasi**: Jika nilai **p-value < 0.05** → fitur signifikan → **dipertahankan**.
- **Contoh Dipertahankan**: `hypertension`, `diabetes`, `sleep_apnea`, `ever_smoked`.
- **Contoh Dieliminasi**: `gender`, `race` (p ≥ 0.05 → tidak signifikan).

### 3.3 Feature Selection — ANOVA F-Test
- **Digunakan untuk**: Fitur **kontinu** (angka, seperti usia, tekanan darah, dll).
- **Cara Kerja**: F = Variansi Antar-Grup / Variansi Dalam-Grup. Membandingkan rata-rata nilai numerik antara kelompok stroke dan non-stroke.
- **Interpretasi**: Nilai F besar (p-value < 0.05) → fitur mampu membedakan kedua kelas → **dipertahankan**.
- **Contoh Dieliminasi**: `bmi` (p ≥ 0.05), `sedentary_min`.

### 3.4 Decision Tree (Pohon Keputusan)
- **Cara Kerja**:
  1. Algoritma memilih fitur terbaik untuk dibagi (*split*) berdasarkan metrik **Gini Impurity**: `Gini = 1 - Σ(p_k²)`. Nilai Gini 0 berarti node sudah murni (satu kelas).
  2. Proses dilanjutkan secara rekursif hingga mencapai batas `max_depth` atau node murni.
- **Parameter**: `max_depth=6`, `class_weight='balanced'`, `random_state=42`.
- **Kelemahan**: Mudah *overfitting* jika tidak dibatasi kedalamannya.

### 3.5 Random Forest (Hutan Acak) ← **Model Terbaik Proyek**
- **Konsep Dasar**: *Ensemble Bagging* — gabungan banyak Decision Tree secara paralel.
- **Cara Kerja**:
  1. Buat **200 Decision Tree** independen.
  2. Setiap tree dilatih pada **sampel bootstrap** (sub-sampel acak dengan pengembalian).
  3. Setiap split pada setiap tree hanya mempertimbangkan **subset fitur acak** → keragaman antar-tree.
  4. Prediksi akhir: **Voting Mayoritas** dari 200 tree.
- **Parameter**: `n_estimators=200`, `max_depth=8`, `class_weight='balanced'`, `n_jobs=-1`.
- **Keunggulan**: Variansi rendah, tidak mudah overfitting, tahan terhadap noise. AUC-ROC = **0.7678**.

### 3.6 XGBoost (Extreme Gradient Boosting)
- **Konsep Dasar**: *Ensemble Boosting* — tree dibangun secara **sekuensial**, bukan paralel.
- **Cara Kerja**:
  1. Bangun Decision Tree pertama sebagai model awal.
  2. Hitung **residual error** dari tree sebelumnya.
  3. Tree berikutnya dilatih khusus untuk memperbaiki residual tersebut menggunakan *Gradient Descent* pada fungsi Log Loss.
  4. Prediksi akhir: penjumlahan bertahap semua tree dibobot `learning_rate`.
- **Parameter**: `n_estimators=200`, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=10`, `eval_metric='logloss'`.
- **Perbedaan dengan RF**: RF = paralel (bagging), XGB = sekuensial (boosting). XGBoost lebih ekspresif tapi lebih sensitif terhadap tuning parameter.

### 3.7 Youden's J-Statistic (Penalaan Threshold)
- **Masalah**: Threshold default `0.50` menghasilkan Recall hanya **17%** — model melewatkan 83% pasien stroke.
- **Formula**: J(threshold) = True Positive Rate + True Negative Rate − 1
- **Cara Kerja**: Hitung nilai J untuk setiap titik threshold pada Kurva ROC. Pilih threshold di mana J **paling maksimal**.
- **Hasil**: Threshold optimal = **`0.2344`** → Recall naik menjadi **77.14%**.
- **Logika Medis**: Dalam konteks skrining kesehatan, lebih baik sedikit *false alarm* (false positive) daripada melewatkan pasien stroke yang sesungguhnya (*false negative*).

### 3.8 SHAP (SHapley Additive exPlanations)
- **Tujuan**: Menjawab "Mengapa model memprediksi risiko ini untuk pengguna ini?" secara individual.
- **Dasar Teori**: Nilai Shapley dari **teori permainan kooperatif** — mengalokasikan kontribusi prediksi secara adil ke setiap fitur.
- **Formula**: g(z') = φ₀ + Σ(φᵢ × z'ᵢ) — di mana φ₀ = baseline rata-rata model, φᵢ = kontribusi fitur ke-i.
- **TreeExplainer**: Implementasi SHAP khusus untuk model berbasis pohon (RF, XGBoost). Jauh lebih cepat karena mengeksploitasi struktur tree.
- **Global vs Lokal**:
  - **Global**: Rata-rata |SHAP| seluruh data → fitur terpenting secara keseluruhan (Gambar 3.4 laporan).
  - **Lokal (Real-time)**: Nilai SHAP khusus untuk input pengguna saat ini → ditampilkan sebagai bar chart dua arah di web app.

---

## 4. Daftar Pertanyaan yang Mungkin Muncul & Jawabannya

### 🔴 Tentang Dataset

**Q: Mengapa memilih dataset NHANES?**
> Dataset NHANES adalah survei nasional representatif populasi umum Amerika yang dirilis CDC. Keunggulannya: (1) mencakup variabel klinis DAN gaya hidup secara bersamaan, (2) data publik dan terstandar baik, (3) sudah digunakan penelitian lain sehingga bisa dibandingkan (Huang & Liu 2025), dan (4) gratis tanpa hambatan lisensi.

**Q: Mengapa menggunakan data NHANES Amerika, bukan data Indonesia?**
> Data stroke Indonesia yang publik, lengkap, dan mencakup variabel gaya hidup multi-domain belum tersedia. NHANES merupakan standar emas survei kesehatan populasi dan metodologinya dapat diadaptasi untuk konteks Indonesia jika datanya tersedia.

**Q: Berapa jumlah data yang digunakan?**
> Dataset final berisi sekitar **9.700+ responden** setelah merge dan pembersihan. Dari total tersebut, sekitar **3.6% adalah kasus stroke positif** (~350 kasus) — inilah penyebab masalah class imbalance.

---

### 🟡 Tentang Preprocessing & Feature Selection

**Q: Mengapa tidak menggunakan semua 31 fitur asli saja?**
> Fitur tidak relevan (*noise*) merusak performa model — model bingung belajar dari informasi yang tidak bermakna. Setelah seleksi statistik, 7 fitur terbukti tidak signifikan (p ≥ 0.05) terhadap prediksi stroke. Menghapusnya meningkatkan AUC-ROC secara dramatis, terutama Skenario A (+12.71%).

**Q: Kenapa BMI dihapus? Bukankah BMI penting untuk kesehatan?**
> Secara medis BMI memang berkaitan dengan kesehatan, namun dalam dataset NHANES ini BMI **tidak menunjukkan perbedaan signifikan (p ≥ 0.05)** antara kelompok stroke dan non-stroke. Ini bisa karena korelasi tingginya dengan lingkar pinggang (`waist_circ`) yang sudah ada. Kita mengikuti hasil data, bukan asumsi subjektif.

**Q: Mengapa SMOTE hanya di data training?**
> Jika SMOTE diterapkan di data uji, kita mengevaluasi model pada data sintetis yang tidak mencerminkan populasi asli. Data uji harus mencerminkan kondisi riil agar evaluasi valid. Menerapkan SMOTE di test set adalah *data leakage* yang membuat hasil tidak dipercaya.

**Q: Mengapa StandardScaler hanya di-fit pada training set?**
> Jika scaler di-fit pada seluruh data, kita "membocorkan" informasi statistik test set (mean & std) ke proses training — ini juga bentuk *data leakage*. Scaler harus belajar dari training set saja, lalu hasilnya diterapkan ke test set.

---

### 🟢 Tentang Model & Algoritma

**Q: Mengapa Random Forest menjadi model terbaik, bukan XGBoost?**
> Pada dataset survei populasi umum seperti NHANES yang memiliki banyak variabel dengan pola tidak linier, Random Forest lebih stabil karena sifat bagging-nya mengurangi variansi. XGBoost yang berbasis boosting lebih sensitif terhadap noise dan parameter tuning. Pada Skenario E, RF mencapai AUC-ROC 0.7678 vs XGBoost 0.7286.

**Q: Apa bedanya Bagging (Random Forest) dan Boosting (XGBoost)?**
> **Bagging** (RF): membangun banyak tree secara **paralel dan independen**, hasil akhir voting mayoritas — mengurangi variansi. **Boosting** (XGBoost): membangun tree secara **sekuensial**, setiap tree baru fokus memperbaiki kesalahan tree sebelumnya — mengurangi bias.

**Q: Apa itu Gini Impurity pada Decision Tree?**
> Gini Impurity mengukur seberapa "tidak murni" sebuah node. Rumus: `Gini = 1 - Σ(p_k²)`. Jika semua sampel di node berasal dari satu kelas, Gini = 0 (murni sempurna). Algoritma Decision Tree selalu mencari split yang menghasilkan Gini paling kecil.

**Q: Apakah akurasi 70% itu cukup baik?**
> Untuk konteks skrining medis dengan data imbalanced, akurasi bukan metrik yang tepat. Yang lebih penting adalah **AUC-ROC (0.7678)** yang mengukur kemampuan pemisahan kelas, dan **Recall (77.14%)** yang mengukur seberapa banyak kasus stroke terdeteksi. Akurasi 70% adalah *trade-off* saat threshold diturunkan untuk meningkatkan sensitivitas.

---

### 🔵 Tentang SHAP & XAI

**Q: Apa bedanya SHAP global dan SHAP lokal?**
> **SHAP Global**: Rata-rata nilai absolut SHAP dari seluruh data test — menunjukkan fitur mana yang *secara umum* paling berpengaruh. **SHAP Lokal**: Nilai SHAP yang dihitung spesifik untuk **satu pengguna** — menunjukkan mengapa *orang ini* diprediksi berisiko tinggi/rendah berdasarkan nilai inputnya secara personal.

**Q: Bagaimana SHAP dihitung di web app secara real-time?**
> Saat backend FastAPI startup, `shap.TreeExplainer(model)` diinisialisasi sekali. Ketika ada request `POST /predict`, backend memanggil `explainer.shap_values(input_scaled)` untuk menghitung kontribusi 24 fitur pengguna. Hasilnya dikirim sebagai array JSON `shap_contributions` ke frontend Next.js untuk divisualisasikan.

**Q: Mengapa kita perlu SHAP padahal Random Forest sudah punya Feature Importance bawaan?**
> Feature Importance bawaan RF (berbasis Impurity Decrease) hanya memberikan gambaran **global rata-rata** dan bias terhadap fitur dengan banyak nilai unik. SHAP lebih unggul karena: (1) dapat menjelaskan prediksi **individual/lokal**, (2) konsisten secara teori (berbasis Shapley Value dari game theory), dan (3) dapat menunjukkan **arah pengaruh** (positif/negatif) dari nilai fitur tersebut.

---

### 🟣 Tentang Backend & Deployment

**Q: Mengapa menggunakan FastAPI, bukan Flask?**
> FastAPI memiliki keunggulan: (1) validasi data input otomatis dengan Pydantic, (2) dokumentasi Swagger UI otomatis di `/docs`, (3) performa lebih cepat (berbasis ASGI/Starlette vs WSGI Flask), dan (4) mendukung async programming secara native.

**Q: Bagaimana cara deploy model .pkl yang besar ke cloud?**
> Backend di-deploy di **Hugging Face Spaces** menggunakan **Docker container** dengan strategi: (1) **Git Orphan Branch** (`deploy-hf`) — branch bersih tanpa file data mentah NHANES (.xpt) yang sangat besar, dan (2) **Git LFS** (*Large File Storage*) — melacak file model biner (`.pkl`) secara terpisah dari repo utama.

**Q: Apa itu Docker dan mengapa digunakan?**
> Docker adalah platform kontainerisasi yang membungkus aplikasi beserta semua dependensinya dalam satu "container" terisolasi. Manfaatnya: "Works on my machine" tidak lagi jadi masalah — siapapun dapat menjalankan backend ini dengan `docker run` tanpa harus menginstall Python dan semua library secara manual.

**Q: Mengapa probabilitas dikirim sebagai desimal (0.0 - 1.0), bukan persentase?**
> Frontend Next.js melakukan konversi: `Math.round(result.probability * 100)`. Jika backend mengirim `25.0` (persen), maka `Math.min(1, 25.0) = 1`, lalu `1 × 100 = 100%` — gauge selalu 100%! Mengirim desimal `0.25` adalah cara benar agar gauge menampilkan 25%.

**Q: Apa itu CORS dan mengapa FastAPI menggunakannya?**
> CORS (*Cross-Origin Resource Sharing*) adalah mekanisme keamanan browser yang memblokir request dari domain berbeda. Karena frontend di Vercel (`nhanes-stroke-dataset.vercel.app`) mengirim request ke backend di Hugging Face (`...hf.space`), browser akan memblokir jika CORS tidak dikonfigurasi. Oleh karena itu, backend menggunakan `CORSMiddleware` dari FastAPI dengan `allow_origins=["*"]` agar semua domain diizinkan.

**Q: Apa fungsi ASGI dan uvicorn?**
> **ASGI** (Asynchronous Server Gateway Interface) adalah standar interface antara web server dan aplikasi Python asinkron (pengganti WSGI yang sinkron). **Uvicorn** adalah implementasi ASGI server yang sangat cepat (berbasis `uvloop` dan `httptools`). Saat backend dijalankan dengan `uvicorn app.main:app`, uvicorn-lah yang mendengarkan port, menerima koneksi HTTP, dan meneruskannya ke FastAPI.

**Q: Kenapa model dimuat saat startup, bukan saat setiap request?**
> Memuat `rf_best_model.pkl`, `scaler.pkl`, dan menginisialisasi `shap.TreeExplainer` dilakukan **sekali saat startup** aplikasi karena proses ini membutuhkan waktu dan memori yang signifikan. Jika dimuat ulang setiap request, latency setiap prediksi akan melonjak drastis. Model disimpan di variabel global `model`, `scaler`, dan `explainer` di memori server.

**Q: Apa itu Git LFS (Large File Storage)?**
> File `.pkl` (model terlatih) berukuran besar — di atas batas file Git biasa (100 MB). Git LFS adalah ekstensi Git yang menyimpan file besar secara terpisah di server LFS (bukan di dalam repository Git utama), sementara repo Git hanya menyimpan *pointer* ke file tersebut. Perintahnya: `git lfs track "*.pkl"` lalu commit seperti biasa.

---

### 🟤 Tentang Frontend & Antarmuka Web

**Q: Bagaimana data quiz dikirim ke backend? Apakah langsung atau ada konversi?**
> Ada proses konversi wajib di fungsi `mapAnswers()` di `api.ts`. Alasannya: tampilan di quiz menggunakan indeks ramah-pengguna (0, 1, 2, 3...) untuk memudahkan pengisian, namun model ML dilatih dengan nilai asli NHANES. Misalnya: indeks pilihan "SD" (1) harus dikonversi ke nilai NHANES `DMDEDUC2 = 2`. Tanpa konversi ini, model akan menerima data yang salah skala dan prediksi akan tidak akurat.

**Q: Mengapa hasil disimpan di `sessionStorage`, bukan langsung di URL atau `localStorage`?**
> - **URL**: Data SHAP (array 24 objek) terlalu besar untuk URL params.
> - `localStorage` bersifat permanen (tersimpan walau browser ditutup-buka) — tidak sesuai untuk data sementara.
> - `sessionStorage` hanya aktif selama **satu sesi tab browser** — ketika tab ditutup, data hilang. Ini tepat untuk data prediksi yang sifatnya sementara.

**Q: Apa itu `normalizeResult()` di halaman result?**
> `normalizeResult()` adalah fungsi validasi defensif yang memastikan data dari `sessionStorage` memiliki tipe yang benar sebelum dirender ke UI. Misalnya: jika `prediction` bukan `string`, fallback ke `"low_risk"`. Ini penting karena `sessionStorage` menyimpan string JSON — jika parsing gagal atau field hilang, UI tidak boleh crash.

**Q: Bagaimana SHAP Bar Chart ditampilkan di frontend?**
> 1. Backend mengirim `shap_contributions` (array 24 objek `{feature, value}`).
> 2. Frontend **mengurutkan** berdasarkan `|value|` terbesar (`Math.abs(b.value) - Math.abs(a.value)`).
> 3. Ambil **top 6** kontributor terbesar.
> 4. Lebar bar = `(|value| / maxAbsVal) × 100%` — dinormalisasi relatif terhadap kontribusi terbesar.
> 5. Warna: `value > 0` → **merah** (menaikkan risiko), `value < 0` → **hijau** (menurunkan risiko).

**Q: Mengapa threshold di web menampilkan 23.44% bukan 23.15% seperti di slide presentasi?**
> Ada perbedaan kecil karena dua sumber angka berbeda: kode backend `main.py` menggunakan `threshold = 0.2344` (dari perhitungan Youden's J akhir), sementara slide menampilkan `0.2315`. Nilai yang **benar dan berlaku di produksi** adalah **0.2344** sesuai kode yang berjalan.

---

### ⚪ Pertanyaan Umum / Refleksi

**Q: Apa keterbatasan proyek ini?**
> (1) Dataset NHANES dari Amerika, belum tentu sepenuhnya mewakili populasi Indonesia. (2) Model tidak melakukan diagnosis klinis — hanya alat skrining awal mandiri. (3) Presisi masih rendah (8.52%) — wajar mengingat prevalensi stroke yang sangat kecil di populasi umum. (4) Pertanyaan kuis cukup teknis (tekanan darah dalam mmHg) — idealnya tersedia opsi pengukuran otomatis.

**Q: Apa yang bisa dikembangkan ke depannya?**
> (1) Validasi model pada data populasi Indonesia. (2) Integrasi biomarker sederhana (kadar kolesterol, gula darah mandiri). (3) Penerapan Deep Learning untuk meningkatkan AUC-ROC. (4) Fitur longitudinal — pemantauan risiko dari waktu ke waktu.

**Q: Apakah aplikasi ini bisa digunakan secara nyata di klinik?**
> Aplikasi ini dirancang sebagai **alat skrining mandiri (self-assessment)**, bukan pengganti diagnosis dokter. Hasil yang ditampilkan — khususnya jika "Risiko Tinggi" — sebaiknya segera dikonfirmasi ke tenaga medis profesional.

**Q: Mengapa Precision model sangat rendah (~8%)?**
> Precision rendah adalah konsekuensi **disengaja** dari menurunkan threshold ke 0.2344. Artinya, dari semua yang diprediksi "risiko tinggi", sekitar 8.52% adalah kasus stroke sungguhan (sisanya *false positive*). Ini trade-off yang **wajar dan disengaja** dalam konteks medis: jauh lebih baik memiliki banyak false positive (yang bisa diperiksa ulang oleh dokter) daripada melewatkan satu kasus stroke sesungguhnya (*false negative* yang bisa fatal).

---

## 5. Cheat Sheet — Angka & Fakta Kunci

> Ringkasan cepat untuk review terakhir sebelum presentasi dimulai.

### 📊 Dataset
| Fakta | Nilai |
|---|---|
| Sumber data | CDC NHANES 2015-2016 |
| Jumlah modul XPT yang digabung | 10 modul |
| ID penggabung | `SEQN` |
| Total responden setelah merge | ~9.700+ |
| Prevalensi stroke | ~3.6% (~350 kasus) |
| Fitur awal (sebelum seleksi) | 31 fitur |
| Fitur setelah seleksi statistik | 24 fitur |
| Fitur noise yang dieliminasi | 7 fitur |
| Rasio train:test split | 80:20 (Stratified) |

### 🤖 Model & Performa
| Fakta | Nilai |
|---|---|
| Model terbaik | Random Forest, Skenario E (24 fitur) |
| Jumlah estimators RF | 200 tree |
| Max depth RF | 8 |
| AUC-ROC RF Skenario E | **0.7678** |
| AUC-ROC XGBoost Skenario E | 0.7286 |
| AUC-ROC Decision Tree Skenario E | 0.6624 |
| Threshold default (sebelum tuning) | 0.50 |
| Recall dengan threshold 0.50 | ~17-24% |
| Threshold optimal (Youden's J) | **0.2344** |
| Recall dengan threshold 0.2344 | **77.14%** |
| Precision dengan threshold 0.2344 | ~8.52% |

### 🧠 Algoritma
| Algoritma | Tipe Ensemble | Cara Bangun Tree | Keunggulan |
|---|---|---|---|
| Decision Tree | Tidak ada | Tunggal, rekursif Gini | Interpretabel, cepat |
| Random Forest | Bagging (paralel) | 200 tree independen | Variansi rendah, stabil |
| XGBoost | Boosting (sekuensial) | Perbaikan residual bertahap | Bias rendah, ekspresif |

### 🔢 24 Fitur Final (Skenario E)
| Kelompok | Jumlah | Fitur |
|---|---|---|
| Klinis | 12 | `age`, `education`, `income_ratio`, `waist_circ`, `systolic_bp`, `diastolic_bp`, `hypertension`, `diabetes`, `heart_failure`, `coronary_disease`, `heart_attack`, `ever_smoked` |
| Tidur | 4 | `snoring_freq`, `sleep_apnea`, `sleep_problem_doctor`, `daytime_sleepy` |
| Stres (PHQ-9) | 5 | `stress_anhedonia`, `stress_depressed`, `stress_fatigue`, `stress_concentration`, `stress_self_esteem` |
| Aktivitas Fisik | 3 | `vigorous_leisure`, `vigorous_leisure_min`, `moderate_leisure` |

### 🌐 Tech Stack
| Komponen | Teknologi |
|---|---|
| Notebook | Jupyter Notebook + Python |
| Backend API | FastAPI + Uvicorn + Pydantic |
| Model serialisasi | joblib (`.pkl`) |
| XAI | SHAP TreeExplainer |
| Frontend | Next.js 15 + TypeScript + Tailwind CSS |
| Hosting Backend | Hugging Face Spaces (Docker) |
| Hosting Frontend | Vercel |
| File besar | Git LFS untuk `.pkl` |

### 🏆 Top 5 Fitur Terpenting (SHAP Global)
1. **Usia (`age`)** — faktor risiko non-modifikasi utama
2. **Hipertensi (`hypertension`)** — riwayat diagnosis aktif
3. **Sleep Apnea (`sleep_apnea`)** — keluhan tidur paling berbahaya
4. **Olahraga Berat (`vigorous_leisure`)** — fitur protektif
5. **Stres Psikologis (`stress_*`)** — beban emosional kronis
