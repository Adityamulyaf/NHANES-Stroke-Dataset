# 📊 Penjelasan Detail: `feature_selection.ipynb`

Notebook ini berfokus pada **Seleksi Fitur secara Objektif** menggunakan metode pengujian statistik. Tujuannya adalah memangkas fitur-fitur yang tidak relevan (*noise*) terhadap target `stroke` sehingga model yang dihasilkan lebih sederhana (*parsimonious*), lebih cepat, dan tidak mudah *overfitting*.

---

## 1. Library yang Diimpor & Rencana Penggunaannya

*   `from sklearn.feature_selection import SelectKBest`:
    *   **Mengapa digunakan?** Ini adalah kelas pembungkus (*wrapper*) scikit-learn yang digunakan untuk memilih sejumlah $k$ fitur terbaik berdasarkan skor fungsi statistik yang kita tentukan (misalnya memilih semua fitur yang signifikan).
*   `from sklearn.feature_selection import chi2` (Uji Chi-Square):
    *   **Mengapa digunakan?** Ini adalah fungsi penilaian statistik yang digunakan bersama `SelectKBest` untuk mengevaluasi fitur-fitur **kategorikal** (biner/ordinal).
*   `from sklearn.feature_selection import f_classif` (ANOVA F-Test):
    *   **Mengapa digunakan?** Ini adalah fungsi penilaian statistik yang digunakan bersama `SelectKBest` untuk mengevaluasi fitur-fitur **kontinu** (numerik).

---

## 2. Mekanisme & Matematika Algoritma Seleksi Fitur

Dosen penguji AI sangat menyukai penjelasan matematis di balik pengujian ini:

### A. Uji Chi-Square ($\chi^2$) — Untuk Fitur Kategorikal
Uji Chi-Square digunakan untuk menentukan apakah ada hubungan asosiasi yang signifikan antara dua variabel kategorikal (dalam hal ini, variabel fitur kategorikal seperti `hypertension` terhadap target `stroke`).

*   **Rumus Matematika**:
    $$\chi^2 = \sum \frac{(O_{i} - E_{i})^2}{E_{i}}$$
    *   Di mana $O_{i}$ adalah nilai observasi aktual (jumlah riil data pada tabel kontingensi).
    *   $E_{i}$ adalah nilai ekspektasi (jumlah teoretis jika kedua variabel saling bebas/tidak berhubungan), dihitung dengan:
        $$E = \frac{\text{Total Baris} \times \text{Total Kolom}}{\text{Total Sampel}}$$

*   **Derajat Kebebasan (df)**:
    $$df = (r - 1) \times (c - 1)$$
    *   Di mana $r$ adalah jumlah baris kategori (2 untuk ya/tidak) dan $c$ adalah jumlah kolom kategori (2 untuk stroke/tidak stroke). Sehingga $df = (2-1) \times (2-1) = 1$.

*   **Interpretasi Hasil (*p-value*)**:
    *   **Hipotesis Nol ($H_0$)**: Fitur dan target saling independen (tidak berhubungan).
    *   **Hipotesis Alternatif ($H_1$)**: Fitur dan target saling dependen (berhubungan signifikan).
    *   Jika nilai **$p\text{-value} < 0.05$** (pada tingkat kepercayaan 95%), kita menolak $H_0$. Artinya, fitur tersebut **signifikan secara statistik** memengaruhi risiko stroke dan layak dipertahankan.

---

### B. ANOVA F-Test (Analysis of Variance) — Untuk Fitur Kontinu
Uji ANOVA (F-Test) digunakan untuk membandingkan rata-rata variabel numerik kontinu (seperti `age` atau `systolic_bp`) di antara kelompok independen yang berbeda (kelompok stroke = 1 vs. kelompok non-stroke = 0).

*   **Rumus Matematika**:
    $$F = \frac{\text{Variansi Antar-Grup (Between-group variance)}}{\text{Variansi Dalam-Grup (Within-group variance)}} = \frac{MSB}{MSW}$$
    *   Di mana $MSB$ (Mean Squared Between) adalah estimasi variansi antar rata-rata kelompok.
    *   $MSW$ (Mean Squared Within) adalah estimasi rata-rata variansi di dalam masing-masing kelompok.
    $$MSB = \frac{SSB}{df_{\text{grup}}} \quad \text{dan} \quad MSW = \frac{SSW}{df_{\text{kesalahan}}}$$

*   **Interpretasi Hasil (*p-value*)**:
    *   Jika rata-rata suatu fitur (misalnya variabel `age`) sangat berbeda secara signifikan di antara kelompok stroke dan non-stroke, maka nilai variansi antar-grup ($MSB$) akan jauh lebih besar daripada variansi dalam-grup ($MSW$).
    *   Ini menghasilkan nilai **$F$ yang besar** dan **$p\text{-value} < 0.05$**. Fitur dengan perbedaan rata-rata yang signifikan ini akan **dipertahankan**.

---

## 3. Pertanyaan Kritis yang Sering Ditanyakan Dosen Penguji

### Q1: Kenapa Uji Chi-Square dan ANOVA F-Test harus dijalankan SEBELUM Standard Scaling dilakukan?
> **Jawaban Utama:**
> Karena uji Chi-Square memiliki **batasan matematis** di mana ia hanya menerima input nilai **non-negatif** ($\ge 0$).
>
> **Penjelasan Teknis:**
> Standardisasi (`StandardScaler`) memusatkan data di sekitar rata-rata 0 dengan membagi standar deviasi. Akibatnya, nilai di bawah rata-rata akan diubah menjadi **nilai negatif** (misal -1.25).
> Jika kita memasukkan nilai negatif ke dalam fungsi Chi-Square (`chi2`), kalkulasi matematika kuadrat dan pembagian akan menghasilkan nilai bias atau menyebabkan error program/kompilasi di Python. Oleh karena itu, seleksi fitur dilakukan segera setelah imputasi missing values, baru kemudian data latih terpilih di-scale.

---

### Q2: Apakah Chi-Square dan ANOVA F-Test mampu mendeteksi dan menangani Multikolinearitas (Korelasi antar-fitur)?
> **Jawaban Utama:**
> **Tidak**. Chi-Square dan ANOVA F-Test adalah metode **univariat**.
>
> **Penjelasan Teknis:**
> Kedua uji ini hanya melihat hubungan independen antara satu fitur tunggal dengan target (`stroke`), tanpa peduli apakah fitur tersebut memiliki korelasi tinggi dengan fitur lainnya (misal korelasi antara `systolic_bp` dan `diastolic_bp`).
> Namun, multikolinearitas ini bukan masalah besar karena model akhir yang kita gunakan adalah **Decision Tree, Random Forest, dan XGBoost** (algoritma berbasis pohon). Algoritma ensemble pohon membagi node secara terpisah dan secara alami dapat menangani multikolinearitas dengan baik selama pembagian split fitur.

---

### Q3: Berapa banyak fitur yang dieliminasi dan mengapa?
> **Jawaban Utama:**
> Dari **31 fitur awal**, sebanyak **7 fitur dieliminasi** karena memiliki nilai $p\text{-value} \ge 0.05$, menyisakan **24 fitur signifikan** yang lolos.
>
> **Daftar Fitur yang Dieliminasi (Noise):**
> 1.  `gender` ($p = 0.308$) - Secara univariat, gender tidak signifikan terhadap risiko stroke di dataset ini.
> 2.  `race` ($p = 0.729$) - Ras responden tidak menunjukkan korelasi univariat yang signifikan.
> 3.  `current_smoker` ($p = 0.287$) - Di luar dugaan, status perokok aktif saat ini tidak signifikan, melainkan variabel `ever_smoked` (pernah merokok seumur hidup) yang terbukti signifikan ($p = 0.002$).
> 4.  `vigorous_work` ($p = 0.763$) - Aktivitas kerja berat tidak signifikan.
> 5.  `bmi` ($p = 0.289$) - Indeks Massa Tubuh tidak signifikan secara univariat, sedangkan `waist_circ` (lingkar pinggang) signifikan ($p = 0.006$) menunjukkan bahwa obesitas sentral lebih relevan secara klinis.
> 6.  `sleep_hours` ($p = 0.697$) - Durasi jam tidur tidak signifikan secara langsung, namun gejala klinis seperti `sleep_apnea` (apnea tidur) sangat signifikan ($p = 1.4 \times 10^{-13}$).
> 7.  `sedentary_min` ($p = 0.738$) - Waktu diam/duduk per hari tidak signifikan.
