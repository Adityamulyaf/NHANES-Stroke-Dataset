"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import StrikethroughTyper from "@/components/StrikethroughTyper";

const steps = [
  {
    num: "01",
    title: "Jawab Pertanyaan",
    desc: "Isi kuis singkat tentang gaya hidup dan riwayat kesehatan Anda.",
  },
  {
    num: "02",
    title: "Analisis AI",
    desc: "Sistem menganalisis jawaban Anda menggunakan model machine learning.",
  },
  {
    num: "03",
    title: "Dapatkan Rekomendasi",
    desc: "Terima analisis risiko dan rekomendasi kesehatan yang dipersonalisasi.",
  },
];

type ModelItem = {
  src: string;
  title: string;
  desc: string;
  explanation: string[];
};

type ModelTab = {
  id: string;
  label: string;
  icon: React.ReactNode;
  items: ModelItem[];
};

const modelTabs: ModelTab[] = [
  {
    id: "auc",
    label: "Perbandingan AUC",
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 13h2l3-8 4 16 3-8h6" />
      </svg>
    ),
    items: [
      {
        src: "/model-results/auc_comparison.png",
        title: "Perbandingan AUC-ROC",
        desc: "Perbandingan skor AUC-ROC antara Decision Tree, Random Forest, dan XGBoost di seluruh skenario eksperimen.",
        explanation: [
          "Grafik ini menampilkan bar chart perbandingan skor AUC-ROC (Area Under the Curve, Receiver Operating Characteristic) dari tiga model, yaitu Decision Tree (biru tua), Random Forest (oranye), dan XGBoost (hijau teal).",
          "AUC-ROC adalah metrik yang mengukur seberapa baik model membedakan antara pasien berisiko stroke dan tidak berisiko. Nilai 1.0 berarti sempurna, sedangkan 0.5 berarti model hanya menebak secara acak.",
          "Bar Decision Tree selalu lebih pendek karena model ini cenderung overfitting, yaitu terlalu menghafal pola data training sehingga performanya turun saat diuji dengan data baru. Random Forest dan XGBoost mengatasi masalah ini dengan menggabungkan banyak pohon keputusan sekaligus (ensemble method).",
          "Skenario A (Klinis saja) justru menunjukkan AUC tertinggi karena fitur klinis seperti usia, hipertensi, dan riwayat jantung memiliki korelasi paling kuat dengan stroke. Penambahan fitur gaya hidup (Skenario B sampai E) tidak selalu meningkatkan performa karena bisa menambah noise atau gangguan pada model.",
          "Perlu diperhatikan bahwa skala sumbu Y dimulai dari 0.50, bukan dari 0. Hal ini bertujuan untuk memperjelas perbedaan antar model, sehingga perbedaan visual antar bar terlihat lebih besar dari selisih angka sebenarnya.",
        ],
      },
      {
        src: "/model-results/heatmap_auc.png",
        title: "Heatmap AUC-ROC",
        desc: "Visualisasi heatmap performa AUC-ROC untuk setiap kombinasi skenario dan model classifier.",
        explanation: [
          "Heatmap ini menampilkan skor AUC-ROC dalam format tabel berwarna. Sumbu Y menunjukkan 5 skenario eksperimen (A sampai E), sedangkan sumbu X menunjukkan 3 model classifier.",
          "Warna biru tua menandakan AUC tinggi (performa bagus), sedangkan warna biru muda menandakan AUC lebih rendah (performa kurang). Angka di dalam setiap sel menunjukkan skor AUC-ROC yang tepat.",
          "Sel paling gelap berada di XGBoost Skenario A (0.8232), yang berarti kombinasi ini memiliki performa tertinggi secara keseluruhan. Namun, platform ini menggunakan Random Forest Skenario E (0.7678) karena memanfaatkan 24 fitur gabungan (klinis + gaya hidup) agar prediksi lebih komprehensif.",
          "Decision Tree secara konsisten berwarna paling terang di setiap skenario. Hal ini menunjukkan bahwa model ini paling lemah dibanding kedua model ensemble lainnya.",
          "Hal menarik lainnya, Skenario E (semua fitur) tidak selalu memberikan AUC terbaik. Ini menunjukkan bahwa lebih banyak fitur belum tentu lebih baik, fenomena ini dikenal sebagai 'curse of dimensionality'.",
        ],
      },
    ],
  },
  {
    id: "roc",
    label: "ROC Curve",
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6m6 0h6m-6 0V9a2 2 0 012-2h2a2 2 0 012 2v10m6 0v-4a2 2 0 00-2-2h-2a2 2 0 00-2 2v4" />
      </svg>
    ),
    items: [
      {
        src: "/model-results/roc_curve_scenario_E.png",
        title: "ROC Curve, Skenario E",
        desc: "Kurva ROC untuk Skenario E (Klinis + Semua Gaya Hidup) yang membandingkan ketiga model klasifikasi.",
        explanation: [
          "Kurva ROC menggambarkan hubungan antara True Positive Rate (TPR/Recall, yaitu berapa banyak kasus stroke yang berhasil terdeteksi) di sumbu Y, dan False Positive Rate (FPR, yaitu berapa banyak yang salah didiagnosis stroke) di sumbu X.",
          "Garis putus-putus abu-abu adalah garis diagonal 'Random Guess' atau tebakan acak. Model yang tidak berguna akan mengikuti garis ini dengan AUC = 0.5. Semakin kurva melengkung ke arah kiri atas, semakin bagus performa modelnya.",
          "Warna biru tua mewakili Decision Tree (AUC 0.6624), oranye mewakili Random Forest (AUC 0.7678), dan hijau teal mewakili XGBoost (AUC 0.7286). Random Forest menunjukkan kurva paling melengkung, yang artinya model ini paling konsisten dalam mendeteksi kasus stroke dengan tingkat kesalahan paling rendah.",
          "Kurva Random Forest (oranye) lebih tinggi karena model ini mengombinasikan voting atau suara dari ratusan pohon keputusan, sehingga prediksinya lebih stabil dan tahan terhadap noise dalam data.",
          "Titik optimal pada kurva (yang paling dekat ke pojok kiri atas) digunakan untuk menentukan threshold prediksi. Platform ini menggunakan threshold 0.2344 berdasarkan analisis Youden's J-Statistic, yang lebih rendah dari default 0.5, agar model lebih sensitif dalam mendeteksi potensi stroke demi memprioritaskan keselamatan pasien.",
        ],
      },
    ],
  },
  {
    id: "shap",
    label: "Analisis SHAP",
    icon: (
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    items: [
      {
        src: "/model-results/shap_summary_bar.png",
        title: "SHAP Feature Importance",
        desc: "Ranking fitur berdasarkan rata-rata nilai SHAP absolut yang menunjukkan fitur mana yang paling berpengaruh dalam prediksi stroke.",
        explanation: [
          "Grafik ini mengurutkan 22 fitur berdasarkan mean(|SHAP value|), yaitu rata-rata kontribusi absolut setiap fitur terhadap output prediksi model. Semakin panjang barnya, semakin besar pengaruh fitur tersebut terhadap hasil prediksi.",
          "Fitur 'age' (usia) mendominasi dengan SHAP value sekitar 0.09, hampir 2 kali lipat dari fitur kedua. Ini berarti usia adalah faktor paling berpengaruh dalam prediksi stroke. Secara medis, risiko stroke memang meningkat secara signifikan setelah usia 55 tahun.",
          "Fitur 'hypertension' dan 'sleep_apnea' di posisi 2 dan 3 menunjukkan bahwa riwayat tekanan darah tinggi dan gangguan tidur (sleep apnea) sangat berkontribusi terhadap risiko stroke.",
          "Fitur di bagian bawah seperti 'diabetes', 'waist_circ', dan 'heart_attack' memiliki bar yang sangat pendek (SHAP sekitar 0.003). Bukan berarti fitur ini tidak penting secara medis, melainkan dalam konteks model ini, kontribusinya relatif kecil setelah memperhitungkan fitur lain yang lebih dominan.",
          "Fitur 'moderate_leisure' (olahraga sedang) memiliki bar lebih tinggi dari 'coronary_disease' karena variasi data aktivitas fisik lebih besar di dalam dataset. Banyak responden yang jarang berolahraga, sehingga perbedaan prediksi antara yang aktif dan tidak aktif menjadi cukup signifikan.",
        ],
      },
      {
        src: "/model-results/shap_beeswarm.png",
        title: "SHAP Beeswarm Plot",
        desc: "Visualisasi pengaruh nilai setiap fitur terhadap output model. Warna merah menunjukkan nilai fitur tinggi, sedangkan biru menunjukkan nilai rendah.",
        explanation: [
          "Beeswarm plot menampilkan setiap titik data (pasien) sebagai satu titik. Posisi horizontal menunjukkan SHAP value atau dampak terhadap prediksi, sedangkan warna menunjukkan nilai fitur: merah/pink berarti nilai tinggi, biru berarti nilai rendah.",
          "Pada fitur 'age', titik merah (usia tua) berada di sisi kanan positif, yang artinya usia tinggi meningkatkan prediksi risiko stroke. Sebaliknya, titik biru (usia muda) berada di sisi kiri negatif, yang menunjukkan usia muda menurunkan risiko.",
          "Pada fitur 'hypertension', titik merah (hipertensi = 1, ada riwayat) berkumpul di kanan sekitar +0.05 sampai +0.08. Ini menunjukkan bahwa riwayat hipertensi secara konsisten meningkatkan prediksi risiko stroke.",
          "Pada fitur 'moderate_leisure' dan 'vigorous_leisure', titik merah (aktif berolahraga) berkumpul di sisi kiri negatif, yang berarti olahraga rutin menurunkan risiko stroke. Sebaliknya, titik biru (tidak berolahraga) berada di kanan, menunjukkan gaya hidup sedentary meningkatkan risiko.",
          "Fitur 'coronary_disease' menunjukkan pola yang menarik, yaitu sebagian besar titik biru berkumpul di tengah (0), tetapi beberapa titik merah menyebar jauh ke kanan. Artinya, untuk sebagian kecil pasien yang memiliki riwayat penyakit koroner, dampaknya terhadap prediksi stroke sangat besar.",
        ],
      },
      {
        src: "/model-results/shap_groups.png",
        title: "Kontribusi Kelompok Fitur",
        desc: "Perbandingan kontribusi kelompok fitur (Klinis, Tidur, Aktivitas Fisik, Stres) berdasarkan analisis SHAP.",
        explanation: [
          "Grafik ini mengelompokkan 24 fitur ke dalam 4 kategori, kemudian menjumlahkan total mean |SHAP value| per kelompok. Hasil ini menunjukkan kontribusi relatif setiap domain kesehatan terhadap prediksi model.",
          "Kelompok Klinis (merah) mendominasi dengan SHAP sebesar 0.3036, sekitar 58% dari total kontribusi. Hal ini wajar karena kelompok ini mencakup fitur-fitur medis inti seperti usia, tekanan darah, hipertensi, diabetes, riwayat jantung, dan kebiasaan merokok.",
          "Kelompok Tidur (hijau teal, 0.0980) berada di posisi kedua, menunjukkan bahwa kualitas tidur seperti sleep apnea, mendengkur, dan masalah tidur cukup berpengaruh. Perlu dicatat bahwa sleep apnea sendiri sudah masuk ke dalam 3 besar fitur terpenting.",
          "Kelompok Aktivitas Fisik (oranye, 0.0844) memiliki kontribusi hampir setara dengan kelompok Tidur. Ini membuktikan bahwa pola hidup aktif ataupun sedentary berperan penting dalam prediksi risiko stroke.",
          "Kelompok Stres/Mental (biru tua, 0.0376) memiliki kontribusi paling kecil. Meskipun stres secara medis berkaitan dengan stroke, data kuesioner PHQ-9 dalam dataset NHANES mungkin kurang sensitif untuk menangkap hubungan langsung. Meski demikian, fitur ini tetap dimasukkan untuk memberikan gambaran kesehatan mental yang lebih lengkap.",
        ],
      },
    ],
  },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState("auc");
  const [lightboxItem, setLightboxItem] = useState<ModelItem | null>(null);

  const currentTab = modelTabs.find((t) => t.id === activeTab)!;

  return (
    <main className="flex flex-1 flex-col items-center">
      {/* ── Hero Section ── */}
      <section className="w-full bg-white border-b border-zinc-100">
        <div className="max-w-3xl mx-auto px-6 py-20 text-center space-y-8">
          {/* Badge */}
          <span className="inline-block rounded-full bg-teal-light text-teal-dark px-4 py-1.5 text-xs font-semibold tracking-wide">
            AI-POWERED HEALTH ASSESSMENT
          </span>

          <h1 className="text-4xl sm:text-5xl font-extrabold leading-tight tracking-tight text-zinc-900">
            Pahami Risiko{" "}
            <span className="text-teal">Stroke</span>{" "}
            Anda
            <br />
            dalam{" "}
            <StrikethroughTyper
              words={["1 Jam", "30 Menit", "5 Menit"]}
              typingSpeed={90}
              strikeDelay={600}
              nextDelay={500}
              className="text-teal"
              finalClassName="border-b-[3px] border-teal pb-0.5"
            />
          </h1>

          <p className="text-base sm:text-lg text-zinc-400 max-w-xl mx-auto leading-relaxed">
            Platform kami menganalisis kesehatan Anda menggunakan machine learning
            yang canggih untuk memberikan wawasan dan rekomendasi yang dipersonalisasi.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link
              href="/quiz"
              className="inline-flex items-center justify-center gap-2 rounded-full bg-teal-dark text-white font-semibold px-8 py-3.5 shadow-lg hover:shadow-xl transition hover:scale-[1.02] hover:bg-teal"
            >
              Mulai Assessment
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </Link>
            <Link
              href="#cara-kerja"
              className="inline-flex items-center justify-center rounded-full border border-zinc-200 text-zinc-600 font-medium px-8 py-3.5 hover:bg-zinc-50 hover:border-zinc-300 transition"
            >
              Pelajari Lebih Lanjut
            </Link>
          </div>

          {/* Trust badges */}
          <div className="flex flex-wrap gap-6 justify-center text-xs text-zinc-400 pt-2">
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-teal inline-block" />
              Tanpa pendaftaran
            </span>
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-teal inline-block" />
              24 fitur kesehatan
            </span>
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-teal inline-block" />
              Hasil instan
            </span>
          </div>

        </div>
      </section>

      {/* ── How It Works ── */}
      <section id="cara-kerja" className="w-full max-w-3xl mx-auto px-6 py-20 space-y-12">
        <div className="text-center space-y-2">
          <h2 className="text-2xl sm:text-3xl font-bold text-zinc-900">
            Cara Kerja
          </h2>
          <p className="text-sm text-zinc-400">
            Tiga langkah sederhana untuk memahami kesehatan Anda
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          {steps.map((s) => (
            <div
              key={s.num}
              className="rounded-2xl bg-white border-l-[3px] border-teal border border-zinc-100 p-6 space-y-3 shadow-sm hover:shadow-md transition"
            >
              <span className="text-2xl font-bold text-teal leading-none">
                {s.num}
              </span>
              <h3 className="text-base font-semibold text-zinc-900">
                {s.title}
              </h3>
              <p className="text-sm text-zinc-400 leading-relaxed">
                {s.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Performa Model ── */}
      <section id="performa-model" className="w-full border-t border-zinc-100 bg-gradient-to-b from-white to-zinc-50/80">
        <div className="max-w-5xl mx-auto px-6 py-20 space-y-10">
          {/* Section Header */}
          <div className="text-center space-y-3">
            <span className="inline-block rounded-full bg-teal-light text-teal-dark px-4 py-1.5 text-xs font-semibold tracking-wide">
              TRANSPARANSI MODEL
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold text-zinc-900">
              Performa Model
            </h2>
            <p className="text-sm text-zinc-400 max-w-lg mx-auto leading-relaxed">
              Hasil evaluasi model machine learning kami dari proses training menggunakan dataset NHANES — membuktikan keandalan prediksi.
            </p>
          </div>

          {/* Tab Navigation */}
          <div className="flex flex-wrap justify-center gap-2">
            {modelTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium transition-all duration-200 cursor-pointer ${
                  activeTab === tab.id
                    ? "bg-teal text-white shadow-md shadow-teal/20"
                    : "bg-white text-zinc-500 border border-zinc-200 hover:border-zinc-300 hover:text-zinc-700 hover:shadow-sm"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content — Image Cards */}
          <div
            key={activeTab}
            className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in"
          >
            {currentTab.items.map((item, idx) => (
              <div
                key={idx}
                onClick={() => setLightboxItem(item)}
                className={`group relative rounded-2xl bg-white border border-zinc-100 overflow-hidden shadow-sm hover:shadow-lg transition-all duration-300 cursor-pointer hover:-translate-y-1 ${
                  currentTab.items.length === 1 ? "md:col-span-2 max-w-2xl mx-auto w-full" : ""
                } ${
                  currentTab.items.length === 3 && idx === 2 ? "md:col-span-2 max-w-lg mx-auto w-full" : ""
                }`}
              >
                {/* Image Container */}
                <div className="relative aspect-[4/3] bg-zinc-50 overflow-hidden">
                  <Image
                    src={item.src}
                    alt={item.title}
                    fill
                    className="object-contain p-4 group-hover:scale-[1.03] transition-transform duration-500"
                    sizes="(max-width: 768px) 100vw, 50vw"
                  />
                  {/* Hover overlay */}
                  <div className="absolute inset-0 bg-teal/0 group-hover:bg-teal/5 transition-colors duration-300 flex items-center justify-center">
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-white/90 backdrop-blur-sm rounded-full p-3 shadow-lg">
                      <svg className="w-5 h-5 text-teal" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
                      </svg>
                    </span>
                  </div>
                </div>
                {/* Card info */}
                <div className="p-5 space-y-1.5">
                  <h3 className="text-sm font-semibold text-zinc-900">
                    {item.title}
                  </h3>
                  <p className="text-xs text-zinc-400 leading-relaxed">
                    {item.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Model Yang Digunakan */}
          <div className="mt-6 rounded-2xl bg-gradient-to-r from-teal/5 via-teal/10 to-teal/5 border border-teal/15 p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-teal/10 flex items-center justify-center">
                <svg className="w-6 h-6 text-teal" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                </svg>
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-bold text-zinc-900">
                  Model Digunakan: Random Forest — Skenario E (Klinis + Semua Gaya Hidup)
                </h3>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  Platform ini menggunakan model <span className="font-semibold text-teal">Random Forest</span> dengan 24 fitur gabungan (klinis, tidur, stres, dan aktivitas fisik) dari Skenario E dengan AUC-ROC sebesar <span className="font-semibold text-teal">0.7678</span>. Dipilih karena memanfaatkan seluruh aspek kesehatan untuk memberikan prediksi yang lebih komprehensif dan menyeluruh, serta memiliki keseimbangan terbaik antara sensitivitas deteksi dan interpretabilitas model.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Lightbox Modal with Explanation ── */}
      {lightboxItem && (
        <div
          className="fixed inset-0 z-[100] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in cursor-pointer"
          onClick={() => setLightboxItem(null)}
        >
          <div className="relative max-w-6xl w-full max-h-[92vh] bg-white rounded-2xl overflow-hidden shadow-2xl flex flex-col lg:flex-row" onClick={(e) => e.stopPropagation()}>
            {/* Close button */}
            <button
              onClick={() => setLightboxItem(null)}
              className="absolute top-4 right-4 z-10 bg-white/90 backdrop-blur-sm rounded-full p-2 shadow-md hover:bg-zinc-100 transition cursor-pointer"
            >
              <svg className="w-5 h-5 text-zinc-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            {/* Image Side */}
            <div className="relative w-full lg:w-3/5 h-[40vh] lg:h-auto bg-zinc-50 flex-shrink-0">
              <Image
                src={lightboxItem.src}
                alt={lightboxItem.title}
                fill
                className="object-contain p-4 lg:p-6"
                sizes="60vw"
              />
            </div>

            {/* Explanation Side */}
            <div className="w-full lg:w-2/5 p-6 lg:p-8 overflow-y-auto max-h-[50vh] lg:max-h-[92vh] border-t lg:border-t-0 lg:border-l border-zinc-100">
              <div className="space-y-4">
                {/* Title & Description */}
                <div className="space-y-2">
                  <span className="inline-block rounded-full bg-teal-light text-teal-dark px-3 py-1 text-[10px] font-semibold tracking-wide">
                    PENJELASAN DETAIL
                  </span>
                  <h3 className="text-lg font-bold text-zinc-900">
                    {lightboxItem.title}
                  </h3>
                  <p className="text-sm text-zinc-400 leading-relaxed">
                    {lightboxItem.desc}
                  </p>
                </div>

                {/* Divider */}
                <div className="border-t border-zinc-100" />

                {/* Explanation Points */}
                <div className="space-y-3">
                  {lightboxItem.explanation.map((point, idx) => (
                    <div key={idx} className="flex gap-3">
                      <div className="flex-shrink-0 w-6 h-6 rounded-full bg-teal/10 flex items-center justify-center mt-0.5">
                        <span className="text-[10px] font-bold text-teal">{idx + 1}</span>
                      </div>
                      <p className="text-xs text-zinc-600 leading-relaxed">
                        {point}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <footer className="w-full border-t border-zinc-100 py-6 text-center text-xs text-zinc-400">
        Data tidak disimpan. Hasil hanya untuk tujuan edukasi dan tidak menggantikan diagnosis medis.
      </footer>
    </main>
  );
}