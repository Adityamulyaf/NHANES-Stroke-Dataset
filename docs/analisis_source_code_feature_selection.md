# Analisis Source Code Pipeline Feature Selection NHANES Stroke

Dokumen ini menjelaskan alur source code proyek dari data mentah sampai model siap dipakai. Fokusnya adalah pipeline `feature_selection`, karena pipeline ini yang memakai seleksi fitur statistik dan menghasilkan model final 24 fitur.

File utama yang dibahas:

1. `notebooks/feature_selection/preprocessing_nhanes.ipynb`
2. `notebooks/feature_selection/feature_selection.ipynb`
3. `notebooks/feature_selection/training_nhanes.ipynb`
4. `experiments/run_train_feature_selection.py`
5. `backend/app/main.py`

Output utama:

1. `data/processed/feature_selection/nhanes_clean.csv`
2. `data/processed/feature_selection/nhanes_train_scaled.csv`
3. `data/processed/feature_selection/nhanes_test_scaled.csv`
4. `data/processed/feature_selection/feature_groups.pkl`
5. `models/feature_selection/scaler.pkl`
6. `models/feature_selection/rf_best_model.pkl`
7. `reports/feature_selection/tables/hasil_eksperimen.csv`
8. `reports/feature_selection/figures/*.png`

---

## 1. Gambaran Besar Alur Program

Secara source code, pipeline berjalan seperti ini:

```text
Data mentah NHANES .xpt
        |
        v
preprocessing_nhanes.ipynb
        |
        |-- load 10 file .xpt
        |-- merge berdasarkan SEQN
        |-- mapping target stroke
        |-- encoding fitur
        |-- pilih fitur hasil feature selection
        |-- train-test split
        |-- imputasi missing value
        |-- scaling
        |-- SMOTE khusus data train
        v
data/processed/feature_selection/*.csv
        |
        v
feature_selection.ipynb
        |
        |-- Chi-Square untuk fitur kategorikal
        |-- ANOVA F-Test untuk fitur kontinu
        |-- justifikasi fitur yang dipakai
        v
training_nhanes.ipynb atau experiments/run_train_feature_selection.py
        |
        |-- load data train/test
        |-- load kelompok fitur
        |-- susun 5 skenario eksperimen
        |-- train Decision Tree, Random Forest, XGBoost
        |-- evaluasi metrik
        |-- threshold tuning
        |-- SHAP
        |-- simpan model
        v
models/feature_selection/rf_best_model.pkl
        |
        v
backend/app/main.py
        |
        |-- load model dan scaler
        |-- terima input user 24 fitur
        |-- scaling input
        |-- predict probability
        |-- threshold 0.2344
        |-- tampilkan risiko dan penjelasan
```

Intinya, notebook preprocessing bertugas mengubah data mentah menjadi data siap training, notebook feature selection bertugas memberi justifikasi statistik, notebook training bertugas melatih dan mengevaluasi model, sedangkan backend memakai model yang sudah disimpan untuk prediksi user.

---

## 2. Library yang Dipakai dan Fungsinya

### 2.1 Library Umum Python

| Library | Dipakai di | Fungsi di proyek |
|---|---|---|
| `os` | preprocessing, training, backend | Mengakses path folder, cek file, membuat folder, dan operasi direktori. |
| `pathlib.Path` | preprocessing, training, backend | Membuat path yang lebih aman dan fleksibel antar environment, misalnya Colab, lokal, atau server. |
| `warnings` | training | Menyembunyikan warning yang tidak mengganggu hasil utama agar output notebook lebih rapi. |
| `copy` | training | Membuat salinan model sebelum dilatih di tiap skenario agar model antar eksperimen tidak saling tercampur. |

Contoh kode:

```python
from pathlib import Path
current_dir = Path(os.getcwd())
```

Fungsinya adalah mendeteksi lokasi project secara dinamis, sehingga notebook tetap bisa dijalankan walaupun lokasi folder berbeda.

---

### 2.2 Library Pengolahan Data

| Library | Fungsi utama | Contoh penggunaan |
|---|---|---|
| `pandas` | Membaca data, membuat DataFrame, merge data, menyimpan CSV. | `pd.read_sas()`, `pd.read_csv()`, `df.merge()`, `df.to_csv()` |
| `numpy` | Operasi numerik, array, NaN, argmax, sampling. | `np.nan`, `np.argmax()`, `np.abs()` |
| `joblib` | Menyimpan dan memuat object Python seperti model, scaler, dan feature groups. | `joblib.dump()`, `joblib.load()` |

Contoh kode load data mentah:

```python
dfs[name] = pd.read_sas(path, format='xport', encoding='utf-8')
```

Penjelasan:

`pd.read_sas()` dipakai karena file NHANES berformat `.xpt`, yaitu format SAS XPORT. Hasil pembacaan disimpan sebagai DataFrame pandas.

Contoh kode merge:

```python
df = df.merge(other, on='SEQN', how='left')
```

Penjelasan:

Semua modul NHANES digabung berdasarkan `SEQN`, yaitu ID responden. `how='left'` berarti semua responden dari data utama demografi tetap dipertahankan, lalu data modul lain ditempelkan jika tersedia.

---

### 2.3 Library Machine Learning dari Scikit-Learn

| Modul | Fungsi di proyek |
|---|---|
| `train_test_split` | Membagi data menjadi train dan test 80:20. |
| `SimpleImputer` | Mengisi missing value, di pipeline ini memakai median. |
| `StandardScaler` | Menyamakan skala fitur agar fitur besar seperti tekanan darah tidak mendominasi fitur kecil seperti biner 0/1. |
| `SelectKBest` | Menjalankan seleksi fitur univariat berdasarkan skor statistik. |
| `chi2` | Uji Chi-Square untuk fitur kategorikal, biner, atau ordinal. |
| `f_classif` | ANOVA F-Test untuk fitur numerik kontinu. |
| `DecisionTreeClassifier` | Model pohon keputusan. |
| `RandomForestClassifier` | Model ensemble banyak decision tree. |
| `accuracy_score` | Menghitung akurasi. |
| `precision_score` | Menghitung precision. |
| `recall_score` | Menghitung recall atau sensitivitas. |
| `f1_score` | Menghitung harmonic mean precision dan recall. |
| `roc_auc_score` | Menghitung AUC-ROC. |
| `roc_curve` | Menghasilkan FPR, TPR, dan threshold untuk tuning threshold. |

Contoh kode split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Penjelasan:

`test_size=0.2` berarti 20 persen data dipakai sebagai test set. `stratify=y` menjaga proporsi stroke dan tidak stroke tetap seimbang antara train dan test. Ini penting karena data stroke sangat imbalance.

Contoh kode imputasi:

```python
imputer = SimpleImputer(strategy='median')
X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])
```

Penjelasan:

Imputer di-fit hanya pada data train, lalu diterapkan ke test. Ini mencegah data leakage karena statistik dari data test tidak ikut dipakai saat training.

Contoh kode scaling:

```python
scaler = StandardScaler()
X_train_scaled_raw = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Penjelasan:

Scaler juga di-fit hanya pada train. Data test hanya di-transform menggunakan scaler yang sudah belajar dari train.

---

### 2.4 Library Imbalanced Learning

| Library | Fungsi |
|---|---|
| `imblearn.over_sampling.SMOTE` | Menyeimbangkan kelas minoritas dengan membuat sampel sintetis pada data training. |

Contoh kode:

```python
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_scaled, y_train_resampled = smote.fit_resample(X_train_scaled_raw, y_train)
```

Penjelasan:

Data stroke hanya sekitar 3,6 persen. Jika tidak ditangani, model bisa cenderung memprediksi semua orang sebagai tidak stroke. SMOTE hanya diterapkan ke data train, bukan test, agar evaluasi tetap realistis.

Output aktual:

```text
Data bersih: 5140 baris
Stroke positif: 186
Stroke negatif: 4954

Train setelah SMOTE:
0 = 3963
1 = 3963

Test tetap asli:
0 = 991
1 = 37
```

---

### 2.5 Library Model Tambahan

| Library | Fungsi |
|---|---|
| `xgboost.XGBClassifier` | Model gradient boosting yang kuat untuk data tabular. |
| `shap` | Explainable AI untuk menjelaskan kontribusi fitur terhadap prediksi model. |

Contoh XGBoost:

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=10,
    verbosity=0
)
```

Penjelasan:

`scale_pos_weight=10` membantu XGBoost lebih memperhatikan kelas stroke yang jumlahnya sedikit.

Contoh SHAP:

```python
explainer = shap.TreeExplainer(best['model'])
shap_values = explainer.shap_values(X_sample_shap)
```

Penjelasan:

`TreeExplainer` dipakai karena model yang dianalisis adalah model berbasis pohon seperti Random Forest dan XGBoost. SHAP menjelaskan fitur mana yang paling menaikkan atau menurunkan risiko stroke.

---

### 2.6 Library Visualisasi

| Library | Fungsi |
|---|---|
| `matplotlib.pyplot` | Membuat plot dasar dan menyimpan gambar. |
| `seaborn` | Membuat visualisasi statistik yang lebih rapi seperti barplot dan heatmap. |

Contoh kode:

```python
sns.barplot(x='Skenario', y='AUC-ROC', hue='Model', data=results_df)
plt.savefig('auc_comparison.png')
```

Penjelasan:

Kode ini membuat grafik perbandingan AUC-ROC antar skenario dan model, lalu menyimpannya sebagai file gambar di folder `reports`.

---

### 2.7 Library Backend

| Library | Fungsi |
|---|---|
| `fastapi` | Membuat API prediksi stroke. |
| `pydantic.BaseModel` | Mendefinisikan schema input 24 fitur dari user. |
| `CORSMiddleware` | Mengizinkan frontend Next.js mengakses backend. |
| `HTTPException` | Memberi response error jika model belum siap. |
| `uvicorn` | Menjalankan server FastAPI. |

Contoh schema input:

```python
class StrokeInput(BaseModel):
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
    snoring_freq: float
    sleep_apnea: float
    sleep_problem_doctor: float
    daytime_sleepy: float
    stress_anhedonia: float
    stress_depressed: float
    stress_fatigue: float
    stress_concentration: float
    stress_self_esteem: float
    vigorous_leisure: float
    vigorous_leisure_min: float
    moderate_leisure: float
```

Penjelasan:

Pydantic memastikan input dari frontend memiliki nama fitur dan tipe data yang sesuai dengan model.

---

## 3. Source Code Preprocessing: Dari Data Mentah ke Data Siap Training

File:

```text
notebooks/feature_selection/preprocessing_nhanes.ipynb
```

### 3.1 Deteksi Root Project

Kode mencari folder project secara dinamis:

```python
current_dir = Path(os.getcwd())
root_dir = None
```

Tujuannya agar notebook tetap bisa berjalan di beberapa environment:

1. Local laptop.
2. Google Colab.
3. Folder project yang berpindah lokasi.

### 3.2 Load Dataset Mentah

Kode mendefinisikan 10 file NHANES:

```python
files = {
    'demo': 'DEMO_I.xpt',
    'mcq':  'MCQ_I.xpt',
    'bpq':  'BPQ_I.xpt',
    'diq':  'DIQ_I.xpt',
    'bmx':  'BMX_I.xpt',
    'bpx':  'BPX_I.xpt',
    'smq':  'SMQ_I.xpt',
    'slq':  'SLQ_I.xpt',
    'dpq':  'DPQ_I.xpt',
    'paq':  'PAQ_I.xpt',
}
```

Makna setiap file:

| File | Isi data |
|---|---|
| `DEMO_I.xpt` | Demografi, umur, gender, ras, pendidikan, income ratio. |
| `MCQ_I.xpt` | Riwayat penyakit, termasuk stroke dan penyakit jantung. |
| `BPQ_I.xpt` | Riwayat hipertensi. |
| `DIQ_I.xpt` | Riwayat diabetes. |
| `BMX_I.xpt` | Body measurement seperti BMI dan lingkar pinggang. |
| `BPX_I.xpt` | Hasil pemeriksaan tekanan darah. |
| `SMQ_I.xpt` | Kebiasaan merokok. |
| `SLQ_I.xpt` | Tidur, mendengkur, sleep apnea, kantuk siang. |
| `DPQ_I.xpt` | Depression questionnaire, dipakai sebagai proxy stres. |
| `PAQ_I.xpt` | Aktivitas fisik. |

### 3.3 Seleksi Kolom Awal

Source code tidak langsung memakai semua kolom NHANES. Hanya kolom yang relevan diambil.

Contoh:

```python
demo_cols = ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'INDFMPIR']
demo_clean = dfs['demo'][demo_cols].copy()
```

Penjelasan:

`SEQN` wajib dipertahankan karena dipakai sebagai key untuk merge antar modul.

Contoh target:

```python
mcq_cols = ['SEQN', 'MCQ160F', 'MCQ160B', 'MCQ160C', 'MCQ160E']
```

Kolom penting:

| Kolom NHANES | Nama akhir | Makna |
|---|---|---|
| `MCQ160F` | `stroke` | Pernah stroke atau tidak. |
| `MCQ160B` | `heart_failure` | Gagal jantung. |
| `MCQ160C` | `coronary_disease` | Penyakit jantung koroner. |
| `MCQ160E` | `heart_attack` | Serangan jantung. |

### 3.4 Merge Data

```python
df = demo_clean
for other in [mcq_clean, bpq_clean, diq_clean, bmx_clean,
              bpx_clean, smq_clean, slq_clean, dpq_clean, paq_clean]:
    df = df.merge(other, on='SEQN', how='left')
```

Penjelasan:

Ini adalah proses ekstraksi dan integrasi data. Semua data modular NHANES disatukan menjadi satu tabel besar berdasarkan responden yang sama.

### 3.5 Membuat Target Stroke

```python
df['stroke'] = df['MCQ160F'].map({1.0: 1, 2.0: 0})
df = df[df['stroke'].notna()].copy()
```

Penjelasan:

NHANES memakai kode:

1. `1.0` berarti "Yes".
2. `2.0` berarti "No".

Kode tersebut diubah menjadi format machine learning:

1. `1` berarti stroke.
2. `0` berarti tidak stroke.

Baris dengan target kosong dibuang karena model supervised learning membutuhkan label target.

### 3.6 Filter Usia Dewasa

```python
df = df[df['RIDAGEYR'] >= 18].copy()
```

Penjelasan:

Model hanya memakai responden dewasa, karena risiko stroke dan variabel gaya hidup pada anak-anak tidak cocok dibandingkan langsung dengan orang dewasa.

### 3.7 Encoding Fitur

Contoh:

```python
binary_cols = {
    'BPQ020':  'hypertension',
    'DIQ010':  'diabetes',
    'MCQ160B': 'heart_failure',
    'MCQ160C': 'coronary_disease',
    'MCQ160E': 'heart_attack',
    'SMQ020':  'ever_smoked',
    'SLQ050':  'sleep_problem_doctor',
    'PAQ650':  'vigorous_leisure',
    'PAQ665':  'moderate_leisure',
}
```

Lalu:

```python
df[new] = df[old].map({1.0: 1, 2.0: 0, 3.0: 1})
```

Penjelasan:

Kode survei NHANES diubah menjadi format numerik. Umumnya:

1. `1.0` menjadi `1` atau "Ya".
2. `2.0` menjadi `0` atau "Tidak".
3. Pada diabetes, kode `3.0` berarti borderline, lalu digabung ke `1` karena tetap dianggap berisiko.

### 3.8 Rename Kolom

```python
rename_map = {
    'RIDAGEYR': 'age',
    'INDFMPIR': 'income_ratio',
    'BMXWAIST': 'waist_circ',
    'BPXSY1': 'systolic_bp',
    'BPXDI1': 'diastolic_bp',
    'SLQ030': 'snoring_freq',
    'SLQ040': 'sleep_apnea',
    'PAD660': 'vigorous_leisure_min',
}
```

Penjelasan:

Nama kolom NHANES sulit dibaca. Rename membuat fitur lebih mudah dipahami saat training, SHAP, dan frontend.

### 3.9 Kelompok Fitur Final

Pada pipeline feature selection, fitur final dibagi menjadi:

```python
CLINICAL = [
    'age', 'education', 'income_ratio', 'waist_circ', 'systolic_bp', 'diastolic_bp',
    'hypertension', 'diabetes', 'heart_failure', 'coronary_disease', 'heart_attack',
    'ever_smoked'
]

SLEEP = [
    'snoring_freq', 'sleep_apnea', 'sleep_problem_doctor', 'daytime_sleepy'
]

STRESS = [
    'stress_anhedonia', 'stress_depressed', 'stress_fatigue', 'stress_concentration', 'stress_self_esteem'
]

PHYSICAL = [
    'vigorous_leisure', 'vigorous_leisure_min', 'moderate_leisure'
]
```

Total:

| Kelompok | Jumlah fitur |
|---|---:|
| Klinis | 12 |
| Tidur | 4 |
| Stres | 5 |
| Aktivitas fisik | 3 |
| Total | 24 |

Catatan penting untuk menjawab dosen:

Beberapa dokumentasi lama masih menyebut 31 fitur atau masing-masing gaya hidup 5 fitur. Untuk pipeline `feature_selection` yang benar-benar dipakai training final, jumlah aktualnya adalah 24 fitur: 12 klinis, 4 tidur, 5 stres, dan 3 aktivitas fisik.

### 3.10 Missing Value, Split, Scaling, dan SMOTE

Urutannya:

1. Pisahkan `X` dan `y`.
2. Split train-test 80:20.
3. Imputasi missing value.
4. Scaling.
5. SMOTE pada train.
6. Simpan output.

Kode:

```python
X = df_model.drop(columns=[TARGET])
y = df_model[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Kemudian:

```python
imputer = SimpleImputer(strategy='median')
X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])
```

Kemudian:

```python
scaler = StandardScaler()
X_train_scaled_raw = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Kemudian:

```python
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_scaled, y_train_resampled = smote.fit_resample(X_train_scaled_raw, y_train)
```

Kenapa urutannya seperti ini?

1. Split dilakukan sebelum imputasi dan scaling agar tidak ada data leakage.
2. Imputer dan scaler hanya belajar dari train.
3. SMOTE hanya diterapkan pada train agar data test tetap mencerminkan kondisi asli populasi.

### 3.11 Output Preprocessing

```python
df_model.to_csv('nhanes_clean.csv', index=False)
joblib.dump(feature_groups, 'feature_groups.pkl')
joblib.dump(scaler, 'scaler.pkl')
df_train_scaled.to_csv('nhanes_train_scaled.csv', index=False)
df_test_scaled.to_csv('nhanes_test_scaled.csv', index=False)
```

Penjelasan output:

| File | Fungsi |
|---|---|
| `nhanes_clean.csv` | Data bersih sebelum SMOTE, untuk dokumentasi dan analisis. |
| `nhanes_train_scaled.csv` | Data train yang sudah scaling dan SMOTE. |
| `nhanes_test_scaled.csv` | Data test yang sudah scaling tetapi tanpa SMOTE. |
| `feature_groups.pkl` | Metadata daftar fitur per kelompok. |
| `scaler.pkl` | Object scaler yang nanti dipakai backend. |

---

## 4. Source Code Feature Selection

File:

```text
notebooks/feature_selection/feature_selection.ipynb
```

Tujuan notebook ini adalah memberi justifikasi objektif kenapa fitur tertentu dipakai.

### 4.1 Kenapa Chi-Square?

Chi-Square dipakai untuk fitur kategorikal, biner, atau ordinal.

Contoh fitur:

1. `hypertension`
2. `diabetes`
3. `heart_failure`
4. `coronary_disease`
5. `sleep_apnea`
6. `stress_depressed`
7. `vigorous_leisure`

Kode:

```python
chi_selector = SelectKBest(score_func=chi2, k='all')
chi_selector.fit(X_train_cat, y_train)
```

Penjelasan:

`SelectKBest` dengan `chi2` menghitung skor hubungan setiap fitur kategorikal dengan target `stroke`. Semakin tinggi skor dan semakin kecil p-value, semakin kuat hubungan fitur dengan target.

### 4.2 Kenapa ANOVA F-Test?

ANOVA F-Test dipakai untuk fitur numerik kontinu.

Contoh fitur:

1. `age`
2. `income_ratio`
3. `waist_circ`
4. `systolic_bp`
5. `diastolic_bp`
6. `vigorous_leisure_min`

Kode:

```python
anova_selector = SelectKBest(score_func=f_classif, k='all')
anova_selector.fit(X_train_cont, y_train)
```

Penjelasan:

ANOVA menguji apakah rata-rata nilai fitur berbeda signifikan antara kelompok stroke dan tidak stroke.

### 4.3 Hasil Seleksi Fitur yang Kuat

Dari data feature-selection terbaru, fitur Chi-Square terkuat adalah:

| Fitur | Makna |
|---|---|
| `coronary_disease` | Riwayat penyakit jantung koroner. |
| `sleep_apnea` | Gangguan napas saat tidur. |
| `heart_failure` | Riwayat gagal jantung. |
| `heart_attack` | Riwayat serangan jantung. |
| `snoring_freq` | Frekuensi mendengkur. |
| `hypertension` | Riwayat hipertensi. |
| `diabetes` | Riwayat diabetes. |
| `stress_depressed` | Indikator depresi. |
| `stress_self_esteem` | Indikator harga diri rendah. |
| `vigorous_leisure` | Aktivitas fisik berat saat waktu luang. |

Fitur ANOVA terkuat:

| Fitur | Makna |
|---|---|
| `age` | Usia. |
| `systolic_bp` | Tekanan darah sistolik. |
| `income_ratio` | Rasio pendapatan keluarga. |
| `waist_circ` | Lingkar pinggang. |
| `diastolic_bp` | Tekanan darah diastolik. |

### 4.4 Hal yang Perlu Dijelaskan Jika Dosen Mengulik

Feature selection dilakukan pada data train, bukan seluruh data. Alasannya agar informasi dari test set tidak bocor ke proses pemilihan fitur. Jika test ikut dipakai memilih fitur, evaluasi model bisa terlihat lebih bagus dari kondisi sebenarnya.

Catatan teknis:

Notebook `feature_selection.ipynb` masih memiliki beberapa daftar fitur lama seperti `gender`, `race`, `bmi`, `sleep_hours`, `current_smoker`, `vigorous_work`, dan `sedentary_min`. Fitur-fitur tersebut adalah bagian dari kandidat awal 31 fitur, tetapi tidak masuk ke output final 24 fitur pada pipeline `feature_selection`. Untuk presentasi, gunakan daftar final dari `feature_groups.pkl`.

---

## 5. Source Code Training dan Evaluasi Model

File:

```text
notebooks/feature_selection/training_nhanes.ipynb
experiments/run_train_feature_selection.py
```

Kedua file berisi logika training yang sama. Notebook dipakai untuk eksplorasi dan visualisasi interaktif, sedangkan file `.py` lebih mudah dijalankan ulang sebagai script.

### 5.1 Load Data Training dan Test

```python
train_df = pd.read_csv(DATA_DIR / 'nhanes_train_scaled.csv')
test_df = pd.read_csv(DATA_DIR / 'nhanes_test_scaled.csv')
```

Penjelasan:

Training tidak memakai data mentah lagi. Model dilatih dari output preprocessing yang sudah scaling dan sudah dipisahkan train-test.

### 5.2 Load Kelompok Fitur

```python
feature_groups = joblib.load(DATA_DIR / 'feature_groups.pkl')
CLINICAL = feature_groups['CLINICAL']
SLEEP = feature_groups['SLEEP']
STRESS = feature_groups['STRESS']
PHYSICAL = feature_groups['PHYSICAL']
```

Penjelasan:

`feature_groups.pkl` menjaga agar daftar fitur di training konsisten dengan preprocessing dan backend.

### 5.3 Desain 5 Skenario Eksperimen

```python
SCENARIOS = {
    'A: Klinis saja': CLINICAL,
    'B: Klinis + Tidur': CLINICAL + SLEEP,
    'C: Klinis + Stres': CLINICAL + STRESS,
    'D: Klinis + Aktivitas': CLINICAL + PHYSICAL,
    'E: Klinis + Semua Gaya Hidup': CLINICAL + SLEEP + STRESS + PHYSICAL,
}
```

Penjelasan:

Skenario ini dibuat untuk menjawab pertanyaan penelitian: apakah fitur gaya hidup memberi tambahan informasi dibanding fitur klinis saja?

Makna skenario:

| Skenario | Tujuan |
|---|---|
| A | Baseline klinis saja. |
| B | Menguji tambahan fitur tidur. |
| C | Menguji tambahan fitur stres. |
| D | Menguji tambahan fitur aktivitas fisik. |
| E | Menguji semua fitur klinis dan gaya hidup. |

### 5.4 Model yang Dibandingkan

```python
MODELS = {
    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=10,
        verbosity=0
    ),
}
```

Penjelasan parameter:

| Parameter | Fungsi |
|---|---|
| `max_depth` | Membatasi kedalaman pohon agar tidak terlalu overfitting. |
| `random_state=42` | Membuat hasil eksperimen lebih reproducible. |
| `class_weight='balanced'` | Memberi bobot lebih besar ke kelas minoritas. |
| `n_estimators=200` | Jumlah pohon pada Random Forest atau boosting rounds pada XGBoost. |
| `n_jobs=-1` | Memakai semua core CPU untuk mempercepat training Random Forest. |
| `learning_rate=0.05` | Mengatur kecepatan belajar XGBoost. |
| `scale_pos_weight=10` | Membantu XGBoost menangani class imbalance. |
| `eval_metric='logloss'` | Metrik internal XGBoost saat training. |

### 5.5 Fungsi Evaluasi Model

Kode utama:

```python
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = model.predict(X_test)
```

Penjelasan:

1. `model.fit()` melatih model.
2. `predict_proba()` menghasilkan probabilitas stroke.
3. `predict()` menghasilkan label default dengan threshold 0.5.

Metrik default:

```python
acc_def = accuracy_score(y_test, y_pred_default)
prec_def = precision_score(y_test, y_pred_default)
rec_def = recall_score(y_test, y_pred_default)
f1_def = f1_score(y_test, y_pred_default)
auc_roc = roc_auc_score(y_test, y_proba)
```

Penjelasan metrik:

| Metrik | Makna |
|---|---|
| Accuracy | Persentase prediksi benar secara keseluruhan. |
| Precision | Dari semua yang diprediksi stroke, berapa yang benar-benar stroke. |
| Recall | Dari semua pasien stroke asli, berapa yang berhasil terdeteksi. |
| F1 | Keseimbangan precision dan recall. |
| AUC-ROC | Kemampuan model membedakan kelas stroke dan tidak stroke pada berbagai threshold. |

Untuk kasus medis seperti stroke, recall sangat penting karena false negative berbahaya.

### 5.6 Threshold Tuning dengan Youden's J-Statistic

Kode:

```python
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = thresholds[best_idx]
y_pred_opt = (y_proba >= best_thresh).astype(int)
```

Penjelasan:

Threshold default model adalah 0.5. Namun pada data imbalance, threshold 0.5 sering terlalu tinggi dan membuat model gagal mendeteksi kelas stroke. Youden's J mencari threshold yang memaksimalkan:

```text
J = TPR - FPR
```

TPR adalah recall/sensitivity. FPR adalah false positive rate.

Pada backend, threshold Random Forest Skenario E yang dipakai adalah:

```python
threshold = 0.2344
```

Artinya, user diklasifikasikan risiko tinggi jika probabilitas stroke dari model >= 0.2344.

### 5.7 Loop Eksperimen

```python
for scenario_name, feat_cols in SCENARIOS.items():
    X_train_scen = train_df[feat_cols]
    X_test_scen = test_df[feat_cols]

    for model_name, model in MODELS.items():
        m = copy.deepcopy(model)
        metrics, trained_m = evaluate_model(m, X_train_scen, X_test_scen, y_train, y_test)
```

Penjelasan:

Program melatih 3 model pada 5 skenario, sehingga total eksperimen:

```text
5 skenario x 3 model = 15 eksperimen
```

Setiap hasil disimpan ke `results_df`.

### 5.8 Hasil Evaluasi

```python
results_df.to_csv('hasil_eksperimen.csv', index=False)
```

File hasil:

```text
reports/feature_selection/tables/hasil_eksperimen.csv
```

Model dengan AUC tertinggi berdasarkan file hasil:

| Ranking | Skenario | Model | AUC-ROC |
|---:|---|---|---:|
| 1 | A: Klinis saja | XGBoost | 0.8232 |
| 2 | B: Klinis + Tidur | XGBoost | 0.8139 |
| 3 | A: Klinis saja | Random Forest | 0.8106 |
| 4 | C: Klinis + Stres | XGBoost | 0.8064 |
| 5 | C: Klinis + Stres | Random Forest | 0.8026 |

Catatan penting:

Source code memilih:

```python
best = scenario_E_trained_models['Random Forest']
```

Jadi model yang disimpan adalah Random Forest Skenario E, bukan model dengan AUC tertinggi. Alasan yang aman untuk dijelaskan:

Random Forest Skenario E dipilih sebagai model final deployment karena mencakup seluruh kelompok fitur klinis dan gaya hidup, sehingga lebih sesuai dengan tujuan proyek dan interpretasi SHAP. Namun, jika hanya berdasarkan AUC-ROC, performa tertinggi diperoleh XGBoost Skenario A.

### 5.9 Simpan Model

```python
joblib.dump(best['model'], 'models/feature_selection/rf_best_model.pkl')
```

Penjelasan:

Model disimpan agar tidak perlu training ulang saat backend melakukan prediksi.

---

## 6. Source Code SHAP Explainability

SHAP dipakai untuk menjawab pertanyaan "fitur apa yang paling berpengaruh terhadap prediksi model?"

Kode:

```python
explainer = shap.TreeExplainer(best['model'])
shap_values = explainer.shap_values(X_sample_shap)
```

Output SHAP:

1. `shap_summary_bar.png`
2. `shap_beeswarm.png`
3. `shap_groups.png`

Makna visualisasi:

| File | Fungsi |
|---|---|
| `shap_summary_bar.png` | Ranking fitur paling penting berdasarkan rata-rata nilai absolut SHAP. |
| `shap_beeswarm.png` | Menunjukkan arah dan sebaran pengaruh fitur. |
| `shap_groups.png` | Menjumlahkan kontribusi SHAP berdasarkan kelompok klinis, tidur, stres, aktivitas fisik. |

Contoh agregasi SHAP per kelompok:

```python
groups = {
    'Klinis': [c for c in CLINICAL if c in available_feats],
    'Tidur': [c for c in SLEEP if c in available_feats],
    'Stres': [c for c in STRESS if c in available_feats],
    'Aktivitas Fisik': [c for c in PHYSICAL if c in available_feats],
}
```

Penjelasan:

Kode ini mengelompokkan kontribusi setiap fitur agar tidak hanya terlihat fitur individual, tetapi juga kontribusi domain gaya hidup secara keseluruhan.

---

## 7. Source Code Backend: Dari Model ke Prediksi User

File:

```text
backend/app/main.py
```

Backend bertugas memakai model yang sudah dilatih untuk memprediksi input user dari frontend.

### 7.1 Load Model dan Scaler

```python
MODEL_PATH = BASE_DIR / 'models' / 'feature_selection' / 'rf_best_model.pkl'
SCALER_PATH = BASE_DIR / 'models' / 'feature_selection' / 'scaler.pkl'
FEATURE_GROUPS_PATH = BASE_DIR / 'data' / 'processed' / 'feature_selection' / 'feature_groups.pkl'
```

Kemudian:

```python
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
```

Penjelasan:

Backend tidak melatih model. Backend hanya memuat model yang sudah jadi.

### 7.2 Urutan Fitur Harus Sama

```python
FEATURE_ORDER = (
    feature_groups['CLINICAL'] +
    feature_groups['SLEEP'] +
    feature_groups['STRESS'] +
    feature_groups['PHYSICAL']
)
```

Penjelasan:

Urutan fitur saat prediksi harus sama persis dengan urutan fitur saat training. Jika urutan salah, model bisa membaca `age` sebagai `education`, `systolic_bp` sebagai fitur lain, dan prediksi menjadi salah.

### 7.3 Endpoint Predict

```python
@app.post("/predict")
def predict_stroke(data: StrokeInput):
```

Endpoint ini menerima input user, lalu menjalankan langkah:

1. Ubah input Pydantic menjadi dictionary.
2. Susun vektor input sesuai `FEATURE_ORDER`.
3. Scaling dengan scaler training.
4. Prediksi probabilitas stroke.
5. Bandingkan dengan threshold.
6. Buat response risiko dan penjelasan.

Kode inti:

```python
input_dict = data.dict()
input_vector = [input_dict[feat] for feat in FEATURE_ORDER]
input_scaled = scaler.transform([input_vector])
proba = float(model.predict_proba(input_scaled)[0, 1])
```

Penjelasan:

`predict_proba()` menghasilkan probabilitas. Index `[0, 1]` berarti baris pertama dan kelas 1, yaitu kelas stroke.

### 7.4 Threshold Risiko

```python
threshold = 0.2344
prediction_label = "high_risk" if proba >= threshold else "low_risk"
```

Penjelasan:

Threshold tidak memakai 0.5 karena data stroke sangat imbalance. Threshold hasil tuning dipakai agar model lebih sensitif mendeteksi risiko stroke.

---

## 8. Jawaban Singkat Jika Dosen Bertanya

### Pertanyaan: Dari mana data berasal?

Data berasal dari NHANES 2015-2016 dalam format `.xpt`. Ada 10 modul yang digunakan: demografi, riwayat penyakit, hipertensi, diabetes, pengukuran tubuh, tekanan darah, merokok, tidur, stres/depresi, dan aktivitas fisik.

### Pertanyaan: Kenapa digabung dengan `SEQN`?

`SEQN` adalah ID unik responden NHANES. Karena data NHANES dipisah ke banyak modul, semua modul harus digabung menggunakan `SEQN` agar setiap baris merepresentasikan satu responden.

### Pertanyaan: Target stroke dibuat dari kolom apa?

Target dibuat dari `MCQ160F`. Kode `1` diubah menjadi stroke, dan kode `2` diubah menjadi tidak stroke.

### Pertanyaan: Kenapa pakai SMOTE?

Karena data stroke sangat imbalance. Hanya sekitar 3,6 persen responden positif stroke. Tanpa SMOTE, model bisa bias ke kelas mayoritas dan gagal mendeteksi stroke.

### Pertanyaan: Kenapa SMOTE hanya pada train?

Karena test set harus tetap menggambarkan distribusi asli populasi. Kalau SMOTE diterapkan ke test, evaluasi menjadi tidak realistis.

### Pertanyaan: Kenapa pakai StandardScaler?

Karena fitur punya skala berbeda. Contohnya tekanan darah bisa ratusan, sedangkan fitur biner hanya 0 atau 1. Scaling membantu model bekerja dengan input yang konsisten.

### Pertanyaan: Kenapa feature selection pakai Chi-Square dan ANOVA?

Chi-Square cocok untuk fitur kategorikal, biner, atau ordinal. ANOVA F-Test cocok untuk fitur numerik kontinu. Karena fitur proyek terdiri dari dua jenis itu, keduanya digunakan.

### Pertanyaan: Kenapa pakai 5 skenario?

Untuk menguji kontribusi fitur gaya hidup. Skenario A menjadi baseline klinis, lalu skenario B, C, D, dan E menambahkan tidur, stres, aktivitas fisik, atau semuanya.

### Pertanyaan: Model apa saja yang dibandingkan?

Decision Tree, Random Forest, dan XGBoost. Ketiganya cocok untuk data tabular dan mudah dibandingkan dari model sederhana sampai ensemble yang lebih kuat.

### Pertanyaan: Kenapa model final Random Forest Skenario E?

Karena Skenario E memakai semua fitur klinis dan gaya hidup, sehingga sesuai dengan tujuan proyek untuk menganalisis kontribusi gaya hidup. Random Forest juga stabil dan cocok untuk SHAP berbasis tree model. Namun, jika hanya melihat AUC tertinggi, XGBoost Skenario A punya nilai AUC paling tinggi.

### Pertanyaan: Kenapa threshold tidak 0.5?

Karena data imbalance. Threshold 0.5 bisa membuat model terlalu konservatif dan melewatkan pasien berisiko. Threshold tuning dengan Youden's J-Statistic dipakai untuk meningkatkan sensitivitas/recall.

### Pertanyaan: Apa fungsi SHAP?

SHAP menjelaskan kontribusi fitur terhadap prediksi model. Jadi hasil model tidak hanya berupa risiko tinggi atau rendah, tetapi juga bisa dijelaskan faktor apa yang paling memengaruhi prediksi.

---

## 9. Catatan Kritis untuk Presentasi

Ada tiga hal yang sebaiknya dijelaskan dengan hati-hati:

1. Pipeline feature selection final memakai 24 fitur, bukan 31 fitur.
2. Kelompok fitur final tidak semuanya berjumlah 5: tidur 4 fitur, stres 5 fitur, aktivitas fisik 3 fitur.
3. Model final deployment adalah Random Forest Skenario E karena alasan interpretasi dan kelengkapan fitur gaya hidup, bukan karena AUC tertinggi.

Kalimat aman untuk presentasi:

```text
Pada tahap awal kami memiliki kandidat fitur yang lebih banyak. Setelah seleksi statistik menggunakan Chi-Square dan ANOVA F-Test, fitur dipangkas menjadi 24 fitur yang lebih relevan. Model final yang kami deploy adalah Random Forest pada skenario E karena skenario ini mencakup seluruh domain klinis dan gaya hidup, sehingga paling sesuai untuk tujuan explainable machine learning. Namun, kami tetap melaporkan seluruh hasil eksperimen agar performa antar model dan skenario transparan.
```

