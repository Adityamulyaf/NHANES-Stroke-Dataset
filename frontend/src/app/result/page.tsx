"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { PredictResponse } from "@/lib/api";

const FEATURE_LABEL_MAP: Record<string, string> = {
  // Clinical
  age: "Usia",
  education: "Tingkat Pendidikan",
  income_ratio: "Rasio Pendapatan",
  waist_circ: "Lingkar Pinggang",
  systolic_bp: "Tekanan Darah Sistolik",
  diastolic_bp: "Tekanan Darah Diastolik",
  hypertension: "Riwayat Hipertensi",
  diabetes: "Riwayat Diabetes",
  heart_failure: "Gagal Jantung",
  coronary_disease: "Penyakit Jantung Koroner",
  heart_attack: "Serangan Jantung",
  ever_smoked: "Riwayat Merokok",
  // Sleep
  snoring_freq: "Frekuensi Mendengkur",
  sleep_apnea: "Henti Napas (Sleep Apnea)",
  sleep_problem_doctor: "Konsultasi Masalah Tidur",
  daytime_sleepy: "Kantuk di Siang Hari",
  // Stress
  stress_anhedonia: "Kehilangan Minat (Anhedonia)",
  stress_depressed: "Perasaan Sedih/Depresi",
  stress_fatigue: "Kelelahan/Kurang Energi",
  stress_concentration: "Sulit Berkonsentrasi",
  stress_self_esteem: "Perasaan Negatif Diri",
  // Physical
  vigorous_leisure: "Olahraga Berat",
  vigorous_leisure_min: "Durasi Olahraga Berat",
  moderate_leisure: "Olahraga Sedang",
};

function normalizeResult(value: unknown): PredictResponse | null {
  if (!value || typeof value !== "object") return null;
  const data = value as Partial<PredictResponse> & {
    prediction?: unknown;
    probability?: unknown;
    explanation?: unknown;
    recommendation?: unknown;
    shap_contributions?: unknown;
  };

  return {
    prediction: typeof data.prediction === "string" ? data.prediction : "low_risk",
    probability: typeof data.probability === "number" ? data.probability : 0,
    explanation: Array.isArray(data.explanation)
      ? data.explanation.filter((item): item is string => typeof item === "string")
      : [],
    recommendation: typeof data.recommendation === "string" ? data.recommendation : "Tidak ada rekomendasi tersedia.",
    shap_contributions: Array.isArray(data.shap_contributions)
      ? data.shap_contributions.map((item: any) => ({
          feature: typeof item?.feature === "string" ? item.feature : "",
          value: typeof item?.value === "number" ? item.value : 0,
        }))
      : [],
  };
}

export default function ResultPage() {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    if (!result) return;
    
    const probability = Math.round(Math.max(0, Math.min(1, result.probability)) * 100);
    const healthScore = 100 - probability;

    let startTimestamp: number;
    const duration = 1500; // 1.5s animation
    
    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      
      // easeOutQuart
      const easeProgress = 1 - Math.pow(1 - progress, 4);
      setAnimatedScore(Math.round(easeProgress * healthScore));
      
      if (progress < 1) {
        window.requestAnimationFrame(step);
      }
    };
    
    const animationId = window.requestAnimationFrame(step);
    return () => window.cancelAnimationFrame(animationId);
  }, [result]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const stored = sessionStorage.getItem("predict_result");
      if (stored) {
        try {
          setResult(normalizeResult(JSON.parse(stored)));
        } catch {
          setResult(null);
        }
      } else {
        setResult(null);
      }
      setLoaded(true);
    }, 0);

    return () => window.clearTimeout(timer);
  }, []);

  if (!loaded) {
    return (
      <main className="flex flex-1 items-center justify-center bg-white">
        <div className="animate-spin w-8 h-8 border-4 border-teal border-t-transparent rounded-full" />
      </main>
    );
  }

  if (!result) {
    return (
      <main className="flex flex-1 items-center justify-center bg-white px-6 py-20">
        <div className="text-center space-y-6 max-w-sm">
          <div className="w-16 h-16 mx-auto rounded-full bg-teal-light flex items-center justify-center">
            <svg className="w-8 h-8 text-teal" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div className="space-y-2">
            <h1 className="text-xl font-semibold text-zinc-900">Belum Ada Hasil</h1>
            <p className="text-sm text-zinc-500">
              Silakan isi kuis terlebih dahulu untuk melihat hasil analisis.
            </p>
          </div>
          <Link href="/quiz" className="inline-flex items-center gap-2 rounded-2xl clinical-button-primary px-6 py-3 text-sm font-semibold transition">
            Mulai Kuis
          </Link>
        </div>
      </main>
    );
  }

  const isHighRisk = result.prediction === "high_risk";
  const rawProbability = Math.max(0, Math.min(1, result.probability)) * 100;
  const probability = Math.round(rawProbability);
  const healthScore = 100 - probability;
  const explanation = result.explanation ?? [];

  // Urutan kontribusi SHAP
  const rawContributions = result.shap_contributions ?? [];
  const sortedContributions = [...rawContributions].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const topShapContributions = sortedContributions.slice(0, 6);


  return (
    <main className="flex flex-1 flex-col items-center bg-white px-4 py-6 sm:py-8">
      <div className="w-full max-w-5xl space-y-6">

        {/* Back link */}
        <div className="flex items-center justify-between rounded-2xl border border-[#dbe4e2] bg-white px-6 py-4">
          <Link href="/" className="text-sm text-zinc-600 hover:text-teal transition flex items-center gap-1">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
          </svg>
            Keluar
          </Link>
          <span className="text-sm font-medium text-zinc-500">Penilaian Selesai</span>
        </div>

        {/* Status Card */}
        <div className="clinical-surface rounded-[28px] px-6 py-10 sm:px-10 sm:py-12 text-center space-y-8">
          <div className="flex justify-center">
            <div className="w-20 h-20 rounded-full flex items-center justify-center flex-shrink-0 bg-teal-light text-teal">
              {isHighRisk ? (
                <svg className="w-9 h-9 text-zinc-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              ) : (
                <svg className="w-9 h-9" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
            </div>
          </div>

          <div className="space-y-3">
            <h1 className="text-4xl sm:text-5xl font-bold text-zinc-900">
              {isHighRisk ? "Risiko Tinggi" : "Risiko Rendah"}
            </h1>
            <p className="text-2xl sm:text-3xl font-semibold text-teal">
              {isHighRisk ? "Konsultasi Dianjurkan" : "Status Sehat Terkonfirmasi"}
            </p>
            <p className="mx-auto max-w-3xl text-base sm:text-lg leading-8 text-zinc-600">
              {isHighRisk
                ? "Berdasarkan jawaban Anda, profil kesehatan menunjukkan beberapa indikator risiko yang perlu diperhatikan lebih lanjut."
                : "Berdasarkan jawaban Anda, profil kesehatan tampak berada dalam rentang yang cukup aman pada saat ini."}
            </p>
          </div>

          {/* Skor Kesehatan */}
          <div className="mx-auto mt-10 max-w-3xl space-y-2 text-left">
            <div className="flex items-center justify-between text-sm font-medium text-zinc-600">
              <span>Skor Kesehatan</span>
              <span className="text-teal font-bold">{animatedScore}/100</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-[#e7ecee]">
              <div className="h-full rounded-full bg-teal transition-none" style={{ width: `${animatedScore}%` }} />
            </div>
          </div>

          <hr className="border-zinc-200/60 my-6 mx-auto max-w-3xl" />

          {/* Probabilitas Risiko Stroke & Decision Threshold (Youden's J-Statistic) */}
          <div className="mx-auto max-w-3xl space-y-3 text-left">
            <div className="flex items-center justify-between text-sm font-medium text-zinc-600">
              <span className="flex items-center gap-1.5 font-semibold text-zinc-800">
                <svg className="w-4 h-4 text-rose-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Probabilitas Risiko Stroke (AI)
              </span>
              <span className={`${isHighRisk ? 'text-rose-600' : 'text-zinc-600'} font-bold`}>
                {rawProbability.toFixed(1)}%
              </span>
            </div>

            {/* Visual Threshold Slider Bar */}
            <div className="relative h-6 bg-zinc-100 rounded-lg border border-zinc-200 flex items-center overflow-visible">
              {/* Safe Range (0% to 23.44%) */}
              <div className="h-full bg-emerald-100/50 border-r border-zinc-300 rounded-l-lg flex items-center pl-2 text-[9px] sm:text-[10px] text-emerald-800 font-semibold" style={{ width: '23.44%' }}>
                Aman
              </div>
              {/* Warning Range (23.44% to 100%) */}
              <div className="h-full bg-rose-50 flex items-center pl-2 text-[9px] sm:text-[10px] text-rose-800 font-semibold rounded-r-lg" style={{ width: '76.56%' }}>
                Risiko Tinggi
              </div>

              {/* Threshold Indicator Line & Label */}
              <div className="absolute left-[23.44%] top-0 bottom-0 w-[2px] bg-red-600 z-20 flex flex-col items-center">
                <span className="absolute top-6 transform -translate-x-1/2 whitespace-nowrap text-[8px] sm:text-[9px] font-bold text-red-600 bg-white px-1.5 py-0.5 border border-red-200 rounded shadow-sm">
                  Batas Threshold AI (23.44%)
                </span>
              </div>

              {/* User Probability Pin */}
              <div 
                className="absolute h-4 w-4 bg-zinc-800 border-2 border-white rounded-full shadow-md z-30 transition-all duration-1000"
                style={{ 
                  left: `calc(${Math.max(0, Math.min(100, rawProbability))}% - 8px)`
                }}
              >
                <span className="absolute -top-7 transform -translate-x-1/2 whitespace-nowrap text-[9px] sm:text-[10px] font-bold bg-zinc-800 text-white px-1.5 py-0.5 rounded shadow-md">
                  Anda: {rawProbability.toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Spacer for threshold label */}
            <div className="h-4" />

            {/* Threshold Explanation Note */}
            <div className="rounded-2xl border border-[#d8e1df] bg-teal-light/20 p-4 mt-6 text-xs text-zinc-600 leading-relaxed space-y-2">
              <p className="font-bold text-teal-dark flex items-center gap-1.5">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Penjelasan Penalaan Model (Youden's J-Statistic)
              </p>
              <p>
                Dalam penapisan medis awal, kegagalan mendeteksi risiko (<em>false negative</em>) jauh lebih fatal daripada salah mendeteksi risiko (<em>false positive</em>). Oleh karena itu, model Random Forest disetel dengan ambang batas keputusan sensitif sebesar <strong>23.44%</strong> (bukan default 50%). Penyesuaian statistik ini melipatgandakan <strong>Recall (Sensitivitas) model hingga 77.14%</strong> demi keselamatan dan kewaspadaan dini kesehatan Anda.
              </p>
            </div>
          </div>
        </div>

        <section className="grid gap-6 lg:grid-cols-[1fr_1fr]">
          {/* Left Column: SHAP Explainable AI Chart */}
          <div className="space-y-4">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-700">
              <span className="h-5 w-1 rounded-full bg-teal" />
              Bagan Kontribusi Fitur (SHAP - Explainable AI)
            </h2>

            <div className="rounded-[24px] border border-[#d8e1df] bg-white p-6 space-y-6">
              <div className="space-y-2">
                <span className="inline-block rounded-full bg-teal-light text-teal px-3 py-1 text-[10px] font-bold uppercase tracking-wider">
                  Explainable AI (XAI)
                </span>
                <h3 className="text-lg font-bold text-zinc-900">Mengapa Risiko Anda Bernilai {rawProbability.toFixed(1)}%?</h3>
                <p className="text-xs text-zinc-500 leading-relaxed">
                  Grafik dua arah di bawah menampilkan 6 faktor terbesar yang paling memengaruhi pergeseran probabilitas risiko stroke Anda berdasarkan interpretasi SHAP model Random Forest.
                </p>
              </div>

              {/* Legend */}
              <div className="flex justify-between items-center text-[10px] text-zinc-500 bg-zinc-50 rounded-lg p-2.5 border border-zinc-100">
                <span className="flex items-center gap-1.5 font-medium">
                  <span className="w-2.5 h-2.5 rounded bg-emerald-400 inline-block" />
                  Menurunkan Risiko
                </span>
                <span className="w-0.5 h-4 border-l border-dashed border-zinc-300" />
                <span className="flex items-center gap-1.5 font-medium">
                  <span className="w-2.5 h-2.5 rounded bg-rose-400 inline-block" />
                  Meningkatkan Risiko
                </span>
              </div>

              {/* SHAP Bar Chart List */}
              <div className="space-y-3.5">
                {topShapContributions.length > 0 ? (
                  topShapContributions.map((item) => {
                    const label = FEATURE_LABEL_MAP[item.feature] || item.feature;
                    const val = item.value;
                    const maxAbsVal = Math.max(...topShapContributions.map(c => Math.abs(c.value)), 0.01);
                    const barWidth = `${(Math.abs(val) / maxAbsVal) * 100}%`;
                    const percentStr = `${val > 0 ? '+' : ''}${(val * 100).toFixed(1)}%`;
                    
                    return (
                      <div key={item.feature} className="grid grid-cols-[100px_1fr_60px] sm:grid-cols-[130px_1fr_60px] items-center gap-3 py-2 border-b border-zinc-100 last:border-b-0">
                        <span className="text-xs font-semibold text-zinc-700 truncate" title={label}>
                          {label}
                        </span>
                        
                        <div className="relative h-6 bg-zinc-50 rounded border border-zinc-200/60 flex overflow-hidden">
                          {/* Neutral center marker */}
                          <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-zinc-300 z-10" />
                          <div className="w-1/2 flex justify-end">
                            {val < 0 && (
                              <div className="h-full bg-emerald-400 rounded-l" style={{ width: barWidth }} />
                            )}
                          </div>
                          <div className="w-1/2 flex justify-start">
                            {val > 0 && (
                              <div className="h-full bg-rose-400 rounded-r" style={{ width: barWidth }} />
                            )}
                          </div>
                        </div>
                        
                        <span className={`text-xs font-bold text-right ${val > 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                          {percentStr}
                        </span>
                      </div>
                    );
                  })
                ) : (
                  <div className="text-center py-8 text-xs text-zinc-400">
                    Tidak ada kontribusi fitur yang terekam dari backend.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column: Recommendations & Clinical Explanations */}
          <div className="space-y-6">
            {/* Rekomendasi Personal */}
            <div className="space-y-4">
              <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-700">
                <span className="h-5 w-1 rounded-full bg-teal" />
                Rekomendasi Personal
              </h2>

              <div className="grid gap-4">
                <article className="rounded-[24px] border border-[#d8e1df] bg-white p-5 sm:p-6">
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-light text-teal">
                    <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3" />
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 22a10 10 0 100-20 10 10 0 000 20z" />
                    </svg>
                  </div>
                  <h3 className="mb-2 text-lg font-semibold text-zinc-900">Step 1</h3>
                  <p className="text-sm leading-7 text-zinc-600 text-justify">
                    {isHighRisk
                      ? "Pertimbangkan konsultasi ke tenaga medis untuk evaluasi lebih lanjut."
                      : "Pertahankan pola hidup sehat dan lakukan monitoring berkala."}
                  </p>
                </article>

                <article className="rounded-[24px] border border-[#d8e1df] bg-white p-5 sm:p-6">
                  <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-light text-blue">
                    <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                  </div>
                  <h3 className="mb-2 text-lg font-semibold text-zinc-900">Step 2</h3>
                  <p className="text-sm leading-7 text-zinc-600 text-justify">{result.recommendation}</p>
                </article>
              </div>
            </div>

            {/* Penjelasan */}
            <div className="space-y-4">
              <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-700">
                <span className="h-5 w-1 rounded-full bg-teal" />
                Penjelasan Gejala Klinis
              </h2>

              <div className="clinical-surface overflow-hidden rounded-[24px]">
                {explanation.length > 0 ? (
                  explanation.map((item, i) => (
                    <div key={i} className="flex items-start gap-4 border-b border-[#edf1f1] px-5 py-4 last:border-b-0">
                      <span className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-2xl bg-teal-light text-sm font-bold text-teal">
                        {i + 1}
                      </span>
                      <span className="text-sm leading-7 text-zinc-700 text-justify">{item}</span>
                    </div>
                  ))
                ) : (
                  <div className="px-5 py-6 text-sm text-zinc-500">
                    Tidak ada penjelasan tambahan dari backend.
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>



        <section className="rounded-[24px] border border-[#d8e1df] bg-[linear-gradient(135deg,#f4fbfa_0%,#ffffff_55%,#eaf3f7_100%)] p-6 sm:p-8">
          <div className="mx-auto max-w-3xl text-center space-y-3">
            <p className="text-sm font-semibold tracking-[0.16em] text-teal uppercase">MediTrust</p>
            <p className="text-sm leading-7 text-zinc-600">
              Hasil ini hanya untuk tujuan informasi dan tidak menggantikan saran medis profesional.
            </p>
          </div>
        </section>
      </div>
    </main>
  );
}
