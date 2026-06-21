# Analisis Faktor Gaya Hidup pada Prediksi Risiko Stroke menggunakan Explainable Machine Learning

Repository ini berisi proyek analisis dan prediksi risiko stroke menggunakan dataset survei nasional **NHANES (National Health and Nutrition Examination Survey) 2015-2016** dari CDC Amerika Serikat. Proyek ini mengevaluasi kontribusi faktor gaya hidup (tidur, stres, aktivitas fisik) sebagai tambahan dari data klinis dasar.

## Anggota Tim
* **Fadhil Rusadi** (L0124013)
* **Firizqi Aditya Mulya** (L0124016)
* **Nurman Aqil Wicaksono** (L0124139)

---

## Progress Proyek
* **Progress 1: Dataset & Preprocessing**
  * Pemilihan dan pengunduhan dataset NHANES 2015-2016 (10 file modul `.xpt`).
  * Pembersihan data, penyelarasan (*recode*) kode respons, pembagian data (*train-test split* 80:20), imputasi *missing value*, standardisasi fitur (*StandardScaler*), dan penyeimbangan kelas (*SMOTE* pada training set saja) untuk mencegah *data leakage*.
* **Progress 2: Training & Evaluasi Model (Explainable AI)**
  * Melatih 3 model klasifikasi (Decision Tree, Random Forest, XGBoost) pada 5 skenario kombinasi fitur untuk menganalisis kontribusi tidur, stres, dan aktivitas fisik.
  * Evaluasi metrik performa lengkap (Accuracy, Precision, Recall, F1-Score, AUC-ROC) dan visualisasi grafik evaluasi.
  * Penerapan **SHAP values** untuk mengukur pengaruh dan signifikansi kontribusi kelompok fitur klinis vs gaya hidup secara kuantitatif.
* **Progress 3: Web App Deployment**
  * Mengemas model klasifikasi terbaik yang telah dilatih ke dalam bentuk aplikasi web interaktif untuk simulasi prediksi risiko stroke mandiri oleh pengguna.

---

## Dataset & Fitur
Dataset menggabungkan 10 file modul survei `.xpt` terpisah. Model dilatih menggunakan total **31 Fitur + 1 Target (Stroke)** yang dibagi menjadi 4 kelompok utama:

1. **Fitur Klinis (16 fitur)**:
   * Demografi: Usia (*age*), jenis kelamin (*gender*), ras (*race*), tingkat pendidikan (*education*), rasio pendapatan (*income_ratio*).
   * Antropometri: BMI (*bmi*), lingkar pinggang (*waist_circ*).
   * Tekanan Darah: Tekanan darah sistolik (*systolic_bp*) & diastolik (*diastolic_bp*).
   * Riwayat Penyakit: Hipertensi, diabetes, gagal jantung (*heart_failure*), penyakit koroner (*coronary_disease*), serangan jantung (*heart_attack*).
   * Kebiasaan Merokok: Pernah merokok (*ever_smoked*), perokok aktif (*current_smoker*).
2. **Fitur Tidur (5 fitur)**: Jam tidur (*sleep_hours*), frekuensi mendengkur (*snoring_freq*), apnea tidur (*sleep_apnea*), konsultasi masalah tidur ke dokter (*sleep_problem_doctor*), kantuk siang hari (*daytime_sleepy*).
3. **Fitur Stres (5 fitur)**: Kehilangan minat (*stress_anhedonia*), depresi (*stress_depressed*), kelelahan (*stress_fatigue*), gangguan konsentrasi (*stress_concentration*), harga diri rendah (*stress_self_esteem*).
4. **Fitur Aktivitas Fisik (5 fitur)**: Kerja fisik berat (*vigorous_work*), olahraga berat (*vigorous_leisure*), menit olahraga berat (*vigorous_leisure_min*), olahraga sedang (*moderate_leisure*), menit sedentary (*sedentary_min*).

---

## Alur Pipeline & Preprocessing
Untuk mencegah kebocoran data (*data leakage*), preprocessing dilakukan secara ketat dengan urutan berikut:
1. **Merge**: Menggabungkan 10 file modul berdasarkan responden ID (`SEQN`).
2. **Seleksi & Recode**: Menyelaraskan kode respons survei (misal: 1=Ya, 0=Tidak, kode 7/9 menjadi `NaN`).
3. **Split Data**: Pembagian data latih dan uji secara stratified (80% Train, 20% Test).
4. **Imputasi**: Imputasi `Median` untuk fitur numerik kontinyu, `Modus` untuk kategorik/biner, dan `0` untuk aktivitas fisik.
5. **Scaling**: Standardisasi skala fitur menggunakan `StandardScaler` (fit pada data train, transform pada test).
6. **SMOTE**: Penanganan *class imbalance* (~96% tidak stroke vs ~4% stroke) khusus pada data training saja.

---

## Model & Desain Eksperimen
Eksperimen membandingkan 3 algoritma Machine Learning: **Decision Tree**, **Random Forest**, dan **XGBoost** diuji pada 5 skenario kombinasi fitur:
* **Skenario A**: Klinis saja (Baseline)
* **Skenario B**: Klinis + Tidur
* **Skenario C**: Klinis + Stres
* **Skenario D**: Klinis + Aktivitas Fisik
* **Skenario E**: Klinis + Semua Gaya Hidup

Model terbaik diinterpretasikan secara global dan lokal menggunakan **SHAP (SHapley Additive exPlanations)** untuk melihat kontribusi relatif kelompok fitur.

---

## Struktur Notebook & Dokumentasi
* `notebooks/preprocessing_nhanes.ipynb`: Mengunduh dataset mentah, melakukan penggabungan, pembersihan, split data, standardisasi, SMOTE, dan mengekspor file siap latih ke `data/processed/`.
* `notebooks/training_nhanes.ipynb`: Memuat data preprocessed, melakukan perbandingan training & evaluasi pada 5 skenario eksperimen, menyimpan tabel ke `reports/tables/`, menyimpan gambar ke `reports/figures/`, dan menghitung SHAP values untuk interpretasi model.
* `notebooks/feature_selection.ipynb`: Melakukan eksplorasi seleksi fitur dari dataset bersih di `data/processed/nhanes_clean.csv`.

## Struktur Folder
* `data/raw/`: Dataset mentah NHANES dalam format `.xpt`.
* `data/processed/`: Dataset hasil preprocessing dan metadata fitur untuk training.
* `notebooks/`: Notebook pipeline utama.
* `reports/figures/`: Visualisasi evaluasi model dan SHAP.
* `reports/tables/`: Tabel hasil eksperimen.
* `experiments/with_feature_selection/`: Varian eksperimen yang menggunakan seleksi fitur.
* `docs/`: Dokumentasi progres dan referensi proyek.
* `external/`: Salinan repo/dataset eksternal sebagai referensi.
