"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { PredictResponse } from "@/lib/api";

function normalizeResult(value: unknown): PredictResponse | null {
  if (!value || typeof value !== "object") return null;
  const data = value as Partial<PredictResponse> & {
    prediction?: unknown;
    probability?: unknown;
    explanation?: unknown;
    recommendation?: unknown;
  };

  return {
    prediction: typeof data.prediction === "string" ? data.prediction : "low_risk",
    probability: typeof data.probability === "number" ? data.probability : 0,
    explanation: Array.isArray(data.explanation)
      ? data.explanation.filter((item): item is string => typeof item === "string")
      : [],
    recommendation: typeof data.recommendation === "string" ? data.recommendation : "Tidak ada rekomendasi tersedia.",
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
  const probability = Math.round(Math.max(0, Math.min(1, result.probability)) * 100);
  const healthScore = 100 - probability;
  const explanation = result.explanation ?? [];


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

          <div className="mx-auto mt-10 max-w-3xl space-y-2 text-left">
            <div className="flex items-center justify-between text-sm font-medium text-zinc-600">
              <span>Skor Kesehatan</span>
              <span className="text-teal">{animatedScore}/100</span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-[#e7ecee]">
              <div className="h-full rounded-full bg-blue transition-none" style={{ width: `${animatedScore}%` }} />
            </div>
          </div>
        </div>

        <section className="grid gap-6 lg:grid-cols-[1fr_1.15fr]">
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

          <div className="space-y-4">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-700">
              <span className="h-5 w-1 rounded-full bg-teal" />
              Penjelasan
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
