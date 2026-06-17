# Analisis Perbandingan: Sebelum vs. Sesudah Feature Selection

Dokumen ini berisi perbandingan mendalam antara performa model prediksi stroke NHANES **Sebelum Feature Selection** (menggunakan 31 fitur manual) dan **Sesudah Feature Selection** (menggunakan 24 fitur hasil uji Chi-Square & ANOVA F-Test).

---

## 1. Perbedaan Komposisi Fitur

Dengan menerapkan seleksi fitur statistik (Chi-Square untuk data kategorikal dan ANOVA F-Test untuk data kontinu), jumlah fitur dipangkas dari **31 fitur menjadi 24 fitur** dengan mengeliminasi fitur-fitur yang tidak memiliki hubungan univariat signifikan ($p \ge 0.05$) terhadap target `stroke`.

### Fitur yang Dieliminasi (Noise):
1.  **Klinis & Demografi**: `gender`, `race`, `bmi`, `current_smoker`.
2.  **Tidur**: `sleep_hours`.
3.  **Aktivitas Fisik**: `vigorous_work`, `sedentary_min`.
*(Seluruh 5 fitur stres PHQ-9 berhasil dipertahankan karena semuanya terbukti sangat signifikan).*

| Kelompok Fitur | Sebelum Feature Selection (Manual) | Sesudah Feature Selection (Statistik) |
| :--- | :---: | :---: |
| **Klinis & Demografi** | 16 Fitur | 12 Fitur |
| **Gaya Hidup - Tidur** | 5 Fitur | 4 Fitur |
| **Gaya Hidup - Stres** | 5 Fitur | 5 Fitur |
| **Gaya Hidup - Aktivitas Fisik** | 5 Fitur | 3 Fitur |
| **TOTAL FITUR** | **31 Fitur** | **24 Fitur** |

---

## 2. Tabel Perbandingan Kinerja Model

Uji perbandingan dilakukan menggunakan data uji (*test set*) yang sama untuk menilai generalisasi model. Evaluasi mencakup **AUC-ROC** (kemampuan pemisahan kelas) dan **Tuned F1-Score** (keseimbangan presisi dan recall setelah optimasi threshold Youden's J-Statistic).

### A. Perbandingan AUC-ROC

| Skenario Eksperimen | Model | Sebelum Feature Selection (31 Fitur) | Sesudah Feature Selection (24 Fitur) | Selisih Performa |
| :--- | :--- | :---: | :---: | :---: |
| **A: Klinis saja** | Decision Tree | 0.6789 | 0.7335 | **+0.0546** (Naik) |
| | Random Forest | 0.7421 | 0.8106 | **+0.0685** (Naik) |
| | XGBoost | 0.6961 | 0.8232 | **+0.1271** (Naik Masif!) |
| **B: Klinis + Tidur** | Decision Tree | 0.7119 | 0.7389 | **+0.0270** (Naik) |
| | Random Forest | 0.7585 | 0.7763 | **+0.0178** (Naik) |
| | XGBoost | 0.7375 | 0.8139 | **+0.0764** (Naik) |
| **C: Klinis + Stres** | Decision Tree | 0.6936 | 0.7271 | **+0.0335** (Naik) |
| | Random Forest | 0.7453 | 0.8026 | **+0.0573** (Naik) |
| | XGBoost | 0.7361 | 0.8064 | **+0.0703** (Naik) |
| **D: Klinis + Aktivitas**| Decision Tree | 0.6692 | 0.7485 | **+0.0793** (Naik) |
| | Random Forest | 0.7734 | 0.7805 | **+0.0071** (Naik) |
| | XGBoost | 0.7007 | 0.7831 | **+0.0824** (Naik) |
| **E: Klinis + Semua Gaya Hidup**| Decision Tree | 0.6751 | 0.6624 | *-0.0127* (Turun tipis) |
| | Random Forest | 0.7819 | 0.7678 | *-0.0141* (Turun tipis) |
| | XGBoost | 0.7591 | 0.7286 | *-0.0305* (Turun tipis) |

### B. Perbandingan Tuned F1-Score (Threshold Optimized)

| Skenario Eksperimen | Model Terbaik | Tuned F1 (Sebelum FS) | Tuned F1 (Sesudah FS) | Peningkatan Relatif |
| :--- | :--- | :---: | :---: | :---: |
| **A: Klinis saja** | Random Forest / XGBoost | 0.1360 | **0.1734** | **+27.5%** |
| **B: Klinis + Tidur** | XGBoost | 0.1449 | **0.1843** | **+27.1%** |
| **C: Klinis + Stres** | XGBoost | 0.1157 | **0.2101** | **+81.5% (Sangat Drastis!)** |
| **D: Klinis + Aktivitas**| Random Forest / Decision Tree | 0.1442 | **0.2060** | **+42.8%** |
| **E: Klinis + Semua** | Random Forest / XGBoost | **0.1534** | 0.1440 | *-6.1% (Turun tipis)* |

---

## 3. Analisis Hasil Eksperimen

1.  **Reduksi Overfitting pada Skenario A, B, C, dan D**:
    Penghapusan fitur-fitur berisik (*noise*) secara konsisten **meningkatkan performa model secara signifikan**. XGBoost memperoleh peningkatan tertinggi (AUC-ROC naik **+12.7%** pada skenario klinis). Model machine learning tidak lagi terdistorsi oleh fitur-fitur yang tidak relevan secara statistik, membuat batas keputusan (*decision boundary*) menjadi lebih kokoh.
2.  **Peningkatan Kualitas Sampling SMOTE**:
    SMOTE mensintesis sampel baru berdasarkan tetangga terdekat (*k-nearest neighbors*). Dengan mengurangi dimensi fitur dari 31 menjadi 24, SMOTE dapat bekerja lebih akurat di ruang fitur yang lebih sempit tanpa terganggu oleh variabel *noise* (seperti `race` atau `vigorous_work`), menghasilkan data training tiruan yang jauh lebih berkualitas.
3.  **Mengapa Skenario E Mengalami Penurunan Tipis?**:
    Saat menggabungkan seluruh gaya hidup (Tidur + Stres + Aktivitas) secara bersamaan, interaksi non-linear yang kompleks dari beberapa fitur yang dibuang (seperti `sleep_hours` dan `sedentary_min`) mungkin memiliki kontribusi kecil secara kolektif terhadap prediksi model. Namun, penurunan ini sangat tipis ($< 2\%$ pada Random Forest) dan sangat sebanding dengan penyederhanaan model yang didapat.

---

## 4. Kesimpulan: Mana yang Lebih Bagus?

Secara keseluruhan, pendekatan **DENGAN FEATURE SELECTION (24 Fitur)** jauh **LEBIH BAGUS** dibandingkan pendekatan tanpa feature selection (31 fitur).

### Alasan Utama:
1.  **Performa Lebih Tinggi**: Nilai AUC-ROC meningkat pada **4 dari 5 skenario** eksperimen. F1-Score hasil optimasi threshold juga meningkat secara konsisten (bahkan melonjak hingga **+81.5%** pada skenario Klinis + Stres).
2.  **Model Lebih Sederhana (*Parsimonious*)**: Menggunakan fitur yang lebih sedikit (24 vs. 31) membuat model lebih efisien dalam komputasi, mengurangi risiko *overfitting*, dan lebih mudah diterapkan di dunia nyata.
3.  **Pembenaran Ilmiah (Medis) Lebih Kuat**: Pemilihan fitur tidak lagi bersifat manual/subjektif, melainkan didasarkan pada uji statistik Chi-Square & ANOVA yang terstandarisasi. Hal ini memberikan argumen ilmiah yang sangat solid untuk presentasi sidang atau publikasi jurnal.
