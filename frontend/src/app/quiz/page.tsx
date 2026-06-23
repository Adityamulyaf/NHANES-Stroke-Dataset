"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { quizSections, QuizQuestion } from "@/lib/quiz-config";
import { predict } from "@/lib/api";

export default function QuizPage() {
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [answers, setAnswers] = useState<Record<string, number | undefined>>(() => {
    const init: Record<string, number | undefined> = {};
    quizSections.forEach((section) =>
      section.questions.forEach((q) => {
        init[q.key] = undefined;
      })
    );
    return init;
  });

  const totalSteps = quizSections.length;
  const currentSection = quizSections[step];
  const isLastStep = step === totalSteps - 1;
  const progress = Math.round(((step + 1) / totalSteps) * 100);

  const updateAnswer = (key: string, value: number | undefined) =>
    setAnswers((prev) => ({ ...prev, [key]: value }));

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    
    // Gunakan nilai default jika ada jawaban yang masih kosong
    const finalAnswers: Record<string, number> = {};
    quizSections.forEach((section) =>
      section.questions.forEach((q) => {
        finalAnswers[q.key] = answers[q.key] !== undefined ? answers[q.key]! : (q.defaultValue ?? 0);
      })
    );

    try {
      const result = await predict(finalAnswers);
      sessionStorage.setItem("predict_result", JSON.stringify(result));
      router.push("/result");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Terjadi kesalahan. Coba lagi.");
      setLoading(false);
    }
  };

  const isSectionComplete = currentSection.questions.every((q) => answers[q.key] !== undefined);

  return (
    <main className="flex flex-1 flex-col items-center bg-[#f6f8fa] px-4 py-6 sm:py-8">
      <div className="w-full max-w-2xl space-y-4">

        {/* Top bar */}
        <div className="rounded-2xl border border-zinc-200 bg-white overflow-hidden">
          <div className="h-1.5 w-full bg-zinc-100">
            <div
              className="h-1.5 bg-teal rounded-r-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="flex items-center justify-between px-6 py-4">
            <Link
              href="/"
              className="text-sm text-zinc-500 hover:text-teal transition flex items-center gap-1.5"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
              Keluar
            </Link>
            <div className="flex items-center gap-1.5">
              {quizSections.map((_, i) => (
                <div
                  key={i}
                  className={`rounded-full transition-all duration-300 ${
                    i < step ? "w-2 h-2 bg-teal" :
                    i === step ? "w-4 h-2 bg-teal" :
                    "w-2 h-2 bg-zinc-200"
                  }`}
                />
              ))}
            </div>
            <span className="text-sm font-medium text-zinc-400">
              {step + 1} / {totalSteps}
            </span>
          </div>
        </div>

        {/* Card */}
        <div className="rounded-2xl border border-zinc-100 bg-white shadow-sm p-6 sm:p-8 overflow-hidden min-h-[300px]">
          <AnimatePresence mode="wait">
            <motion.div
              key={step}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3, ease: "easeInOut" }}
              className="space-y-6"
            >
              <div className="space-y-1">
                <h2 className="text-xl font-bold text-zinc-900">{currentSection.title}</h2>
                <p className="text-sm text-zinc-400 leading-relaxed">{currentSection.description}</p>
              </div>

              <div className="space-y-4">
                {currentSection.questions.map((q) => (
                  <QuestionInput
                    key={q.key}
                    question={q}
                    value={answers[q.key]}
                    onChange={(val) => updateAnswer(q.key, val)}
                  />
                ))}
              </div>
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-xl bg-red-50 border border-red-100 px-4 py-3 text-sm text-red-500 flex items-center gap-2">
            <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            {error}
          </div>
        )}

        {/* Privacy notice */}
        <div className="flex items-center gap-3 text-xs text-zinc-400 bg-zinc-50 rounded-xl px-4 py-3 border border-zinc-100">
          <svg className="w-4 h-4 text-teal flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
          Data Anda bersifat rahasia dan hanya digunakan untuk analisis.
        </div>

        {/* Navigation */}
        <div className="flex gap-3">
          {step > 0 && (
            <button
              type="button"
              onClick={() => setStep((s) => s - 1)}
              disabled={loading}
              className="flex-1 rounded-2xl border border-zinc-200 bg-white py-4 text-sm font-medium text-zinc-500 hover:bg-zinc-50 transition disabled:opacity-50"
            >
              ← Kembali
            </button>
          )}
          {isLastStep ? (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={loading || !isSectionComplete}
              className="flex-1 rounded-2xl bg-teal-dark text-white py-4 text-sm font-semibold hover:bg-teal transition disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Menganalisis...
                </span>
              ) : "Lihat Hasil →"}
            </button>
          ) : (
            <button
              type="button"
              onClick={() => setStep((s) => s + 1)}
              disabled={!isSectionComplete}
              className="flex-1 rounded-2xl bg-teal-dark text-white py-4 text-sm font-semibold hover:bg-teal transition disabled:opacity-60 disabled:cursor-not-allowed"
            >
              Selanjutnya →
            </button>
          )}
        </div>

      </div>
    </main>
  );
}

// ─── Question Input Component ─────────────────────────────────────────────────

interface QuestionInputProps {
  question: QuizQuestion;
  value: number | undefined;
  onChange: (val: number | undefined) => void;
}

function QuestionInput({ question, value, onChange }: QuestionInputProps) {
  const { key, label, description, type } = question;
  const [numText, setNumText] = useState(value !== undefined ? String(value) : "");

  // Sync when parent value changes (e.g. navigating back to a section)
  if (type === "number") {
    const expected = value !== undefined ? String(value) : "";
    if (numText !== expected && parseInt(numText, 10) !== value) {
      setNumText(expected);
    }
  }

  return (
    <div className="rounded-xl border border-zinc-100 bg-zinc-50 p-4 sm:p-5 space-y-3">
      <div>
        <p className="text-sm font-semibold text-zinc-800">{label}</p>
        <p className="text-xs text-zinc-400 mt-0.5 leading-relaxed">{description}</p>
      </div>

      {type === "number" && (
        <div className="flex items-center rounded-xl border border-zinc-200 bg-white overflow-hidden focus-within:border-teal focus-within:ring-2 focus-within:ring-teal/10 transition">
          <input
            id={key}
            type="number"
            min={question.min}
            max={question.max}
            step={question.step ?? 1}
            value={numText}
            onKeyDown={(e) => { if (["e", "E", "+", "-", "."].includes(e.key)) e.preventDefault(); }}
            onFocus={(e) => e.target.select()}
            onChange={(e) => {
              const raw = e.target.value;
              setNumText(raw);
              if (raw === "") {
                onChange(undefined);
              } else {
                onChange(parseInt(raw, 10) || 0);
              }
            }}
            onBlur={() => {
              if (numText === "") {
                onChange(undefined);
                return;
              }
              let num = parseInt(numText, 10) || 0;
              // Clamp ke batas min/max yang didefinisikan di quiz-config
              if (question.min !== undefined) num = Math.max(question.min, num);
              if (question.max !== undefined) num = Math.min(question.max, num);
              setNumText(String(num));
              onChange(num);
            }}
            className="flex-1 bg-transparent px-4 py-3 text-sm text-zinc-900 outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          {question.unit && (
            <span className="px-3 text-xs font-medium text-zinc-400 border-l border-zinc-100 bg-zinc-50 self-stretch flex items-center select-none">
              {question.unit}
            </span>
          )}
        </div>
      )}

      {type === "slider" && (
        <div className="space-y-4 pt-2 px-1">
          <div className="flex items-center justify-between text-sm">
            <span className="text-zinc-400">{question.min} {question.unit}</span>
            <span className="font-bold text-teal text-lg">
              {value !== undefined ? value : "-"} <span className="text-sm font-medium text-zinc-500">{question.unit}</span>
            </span>
            <span className="text-zinc-400">{question.max} {question.unit}</span>
          </div>
          <input
            type="range"
            min={question.min}
            max={question.max}
            step={question.step ?? 1}
            value={value !== undefined ? value : ((question.min ?? 0) + (question.max ?? 100)) / 2}
            onChange={(e) => onChange(parseInt(e.target.value, 10))}
            className={`w-full h-2 bg-zinc-200 rounded-lg appearance-none cursor-pointer ${question.sliderColor ?? "accent-teal"}`}
          />
        </div>
      )}

      {type === "yesno" && (
        <div className="grid grid-cols-2 gap-2">
          {[{ label: "Tidak", val: 0 }, { label: "Ya", val: 1 }].map((opt) => (
            <button
              key={opt.val}
              type="button"
              onClick={() => onChange(opt.val)}
              className={`rounded-xl py-3 text-sm font-medium transition border ${
                value === opt.val
                  ? "bg-teal-light border-teal text-teal-dark"
                  : "bg-white border-zinc-200 text-zinc-500 hover:border-zinc-300"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}

      {type === "scale" && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {(question.scaleLabels ?? []).map((lbl, i) => (
            <button
              key={i}
              type="button"
              onClick={() => onChange(i)}
              className={`rounded-xl py-3 px-2 text-xs font-medium transition border text-center leading-tight min-h-[3rem] ${
                value === i
                  ? "bg-teal-light border-teal text-teal-dark"
                  : "bg-white border-zinc-200 text-zinc-500 hover:border-zinc-300"
              }`}
            >
              {lbl}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}