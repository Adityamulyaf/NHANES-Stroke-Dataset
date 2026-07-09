# 🛠️ Penjelasan Detail: `preprocessing_nhanes.ipynb`

Notebook ini adalah tahap pertama dalam pipeline data science proyek prediksi stroke NHANES. Tugas utamanya adalah mengunduh data mentah, menggabungkannya, melakukan pembersihan data (*data cleaning*), imputasi, pembagian dataset (*data split*), scaling, dan penyeimbangan kelas (*resampling*).

---

## 1. Library yang Diimpor & Rencana Penggunaannya

Berikut adalah penjelasan mengapa library tertentu harus diimpor di awal notebook ini dan apa kegunaannya:

### A. Library Manajemen Sistem & File
*   `import os` & `from pathlib import Path`:
    *   **Mengapa digunakan?** Proyek ini membutuhkan jalur file (*file path*) yang konsisten baik dijalankan di komputer lokal maupun server. Path dinamis mencari folder `NHANES-Stroke-Dataset` (root proyek) dari folder mana pun notebook dijalankan agar script import data `.xpt` dan eksport data `.csv` tidak patah (*error*).
*   `import warnings; warnings.filterwarnings('ignore')`:
    *   **Mengapa digunakan?** Selama preprocessing, Pandas sering menampilkan peringatan seperti *SettingWithCopyWarning* (saat memodifikasi salinan slice DataFrame). Peringatan ini diabaikan agar output notebook bersih dan mudah dibaca oleh pengguna atau penguji.
*   `import joblib`:
    *   **Mengapa digunakan?** Digunakan untuk melakukan *dumping* (menyimpan) dan *loading* (memuat) objek Python. Di notebook ini, `joblib` digunakan untuk mengekspor objek pemisah kelompok fitur (`feature_groups.pkl`) ke dalam format file biner agar bisa digunakan kembali secara identik di notebook pelatihan model.

### B. Library Analisis & Manipulasi Data
*   `import pandas as pd`:
    *   **Mengapa digunakan?** Pandas adalah core library untuk manipulasi data tabular di Python. Ia menyediakan objek `DataFrame` (tabel dua dimensi) yang digunakan untuk menggabungkan 10 file modul `.xpt` dari CDC NHANES berdasarkan ID unik responden (`SEQN`), melakukan filter kolom, mengubah tipe data, dan mengekspor hasil akhir ke `.csv`.
*   `import numpy as np`:
    *   **Mengapa digunakan?** NumPy digunakan untuk komputasi numerik berbasis array. Ia menyediakan konstanta `np.nan` untuk menandai nilai kosong (*missing value*), mengubah tipe data array, dan melakukan operasi matematika cepat pada data numerik.

### C. Library Machine Learning & Preprocessing (`sklearn` & `imblearn`)
*   `from sklearn.model_selection import train_test_split`:
    *   **Mengapa digunakan?** Untuk membagi data secara acak menjadi subset data latih (*training set*) dan data uji (*testing set*) dengan proporsi tertentu (misal 80:20).
*   `from sklearn.preprocessing import StandardScaler`:
    *   **Mengapa digunakan?** Untuk melakukan standardisasi fitur numerik kontinu. Rumusnya mengubah nilai $x$ menjadi $z = \frac{x - \mu}{\sigma}$ (di mana $\mu$ adalah rata-rata dan $\sigma$ adalah standar deviasi), sehingga semua fitur numerik memiliki rata-rata 0 dan variansi 1. Ini penting untuk algoritma yang sensitif terhadap skala fitur.
*   `from sklearn.impute import SimpleImputer`:
    *   **Mengapa digunakan?** Untuk mengisi nilai yang hilang (*missing values*) secara otomatis dengan strategi tertentu (seperti median atau modus) sebelum data dimasukkan ke model machine learning (karena sebagian besar model tidak menerima input bernilai NaN).
*   `from imblearn.over_sampling import SMOTE`:
    *   **Mengapa digunakan?** SMOTE (*Synthetic Minority Over-sampling Technique*) digunakan untuk menangani ketidakseimbangan kelas (*class imbalance*). Kasus stroke pada dataset NHANES sangat sedikit (hanya ~3.6%). SMOTE membuat sampel buatan baru untuk kelas minoritas agar model tidak bias memprediksi semua orang sebagai "sehat/tidak stroke".

---

## 2. Mengenal Pembagian Dataset: Fitur vs. Label

Sebelum melatih model, data harus dibagi menjadi input fitur dan label target, serta dipisah menjadi data training dan data testing:

| Istilah | Nama Variabel | Definisi & Penjelasan untuk Sidang |
| :--- | :---: | :--- |
| **Input Fitur** | `X` | Seluruh kolom variabel independen (seperti usia, tekanan darah, riwayat merokok, tingkat stres) yang digunakan oleh model untuk memprediksi risiko stroke. |
| **Label Target** | `y` | Variabel dependen tunggal yang ingin diprediksi. Bernilai `1` jika responden menderita stroke, dan `0` jika responden tidak stroke. |
| **Fitur Latih** | `X_train` | Bagian dari matriks fitur `X` (80% data) yang digunakan khusus untuk melatih model agar mengenali pola stroke. |
| **Label Latih** | `y_train` | Bagian dari target `y` (80% data) yang berisi label asli pasangan dari `X_train`, digunakan model untuk belajar koreksi kesalahan prediksi (*loss function*). |
| **Fitur Uji** | `X_test` | Bagian dari matriks fitur `X` (20% sisanya) yang disimpan rapat-rapat selama latihan. Digunakan untuk menguji performa model di akhir. |
| **Label Uji** | `y_test` | Label asli dari data uji `X_test`, digunakan sebagai kunci jawaban pembanding untuk menghitung akurasi, precision, recall, dan F1-Score model setelah prediksi dilakukan. |

---

## 3. Pertanyaan Kritis yang Sering Ditanyakan Dosen Penguji

Berikut adalah skenario pertanyaan sulit dan jawaban ilmiah yang siap Anda sampaikan saat presentasi:

### Q1: Mengapa Dataset Split (train_test_split) harus dilakukan di AWAL, sebelum Imputasi, Scaling, dan SMOTE?
> **Jawaban Utama:**
> Hal ini mutlak dilakukan untuk menghindari **Data Leakage (Kebocoran Data)**.
>
> **Penjelasan Teknis:**
> *   **Jika kita scaling dulu sebelum split:** Rata-rata ($\mu$) dan standar deviasi ($\sigma$) dari data uji akan ikut dihitung untuk menstandardisasi data latih. Model secara tidak langsung sudah "mengintip" distribusi data uji sejak tahap preprocessing.
> *   **Jika kita imputasi dulu sebelum split:** Nilai median yang digunakan untuk mengisi nilai kosong pada data latih akan terpengaruh oleh nilai-nilai yang ada di data uji.
> *   **Jika kita SMOTE dulu sebelum split:** SMOTE akan mensintesis data baru berdasarkan tetangga terdekat (*k-nearest neighbors*). Jika data uji ikut di-SMOTE, data sintetis di data latih bisa dibuat berdasarkan kemiripan dengan data uji. Hal ini membuat performa model di data uji menjadi sangat tinggi secara palsu (*overfitting* terselubung), tetapi akan hancur saat diuji pada data riil baru di luar dataset.
>
> *Oleh karena itu, split dilakukan di awal. Objek `SimpleImputer` dan `StandardScaler` hanya melakukan `.fit()` (mempelajari parameter rata-rata/median) pada `X_train` saja, lalu melakukan `.transform()` ke `X_train` dan `X_test`.*

---

### Q2: Mengapa SMOTE hanya boleh diterapkan pada Data Latih (Train Set) saja?
> **Jawaban Utama:**
> Karena data uji (*test set*) harus mencerminkan **distribusi populasi di dunia nyata** yang sesungguhnya.
>
> **Penjelasan Teknis:**
> Di dunia nyata, penguji atau dokter ingin mendeteksi stroke pada populasi asli di mana penderita stroke adalah kasus langka (~3.6%). Jika kita menerapkan SMOTE pada data uji, kita menyeimbangkan proporsi stroke menjadi 50:50 secara sintetis.
> Evaluasi metrik (seperti Akurasi, F1-score, dan AUC-ROC) pada data uji yang di-SMOTE akan menghasilkan angka yang sangat bagus tapi **palsu**, karena model dievaluasi pada data buatan komputer, bukan data klinis riil manusia. Data uji harus dibiarkan tidak seimbang dan murni tanpa SMOTE.

---

### Q3: Mengapa memilih menggunakan MEDIAN untuk imputasi data numerik, bukan MEAN (Rata-rata)?
> **Jawaban Utama:**
> Karena median bersifat **robust** (kebal) terhadap data pencilan (*outliers*), sedangkan mean sangat sensitif terhadap pencilan.
>
> **Penjelasan Teknis:**
> Pada dataset kesehatan NHANES, variabel seperti pendapatan (`income_ratio`), BMI (`bmi`), atau tekanan darah sering kali memiliki nilai pencilan yang ekstrem (misalnya, beberapa orang dengan tekanan darah sangat tinggi). Jika kita menggunakan Mean (rata-rata), nilai pencilan ekstrem ini akan menarik rata-rata menjauhi pusat distribusi data yang sebenarnya.
> Sebagai contoh, jika ada data: `[10, 12, 14, 15, 120]`, maka:
> *   `Mean = 34.2` (tidak merepresentasikan mayoritas data karena ditarik angka 120).
> *   `Median = 14` (merepresentasikan pusat data yang sebenarnya).
> Oleh karena itu, Median adalah pilihan paling aman dalam statistika kesehatan untuk menjaga integritas data sebelum pemodelan.
