const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface PredictResponse {
  prediction: string;
  probability: number;
  explanation: string[];
  recommendation: string;
  shap_contributions?: Array<{ feature: string; value: number }>;
}

// Map index pilihan quiz → nilai asli NHANES sebelum dikirim ke model
const EDUCATION_MAP: Record<number, number> = {
  0: 1, // Tidak tamat SD
  1: 2, // SD
  2: 3, // SMP
  3: 4, // SMA / SMK
  4: 5, // Kuliah / Lebih tinggi
};

const INCOME_MAP: Record<number, number> = {
  0: 0.5, // < Rp 1 juta
  1: 1.0, // Rp 1–3 juta
  2: 2.0, // Rp 3–6 juta
  3: 3.5, // Rp 6–12 juta
  4: 5.0, // > Rp 12 juta
};

function mapAnswers(raw: Record<string, number>): Record<string, number> {
  const out = { ...raw };

  // education: index 0–4 → nilai NHANES DMDEDUC2 (1–5)
  if (out.education !== undefined) out.education = EDUCATION_MAP[out.education] ?? 3;

  // income_ratio: index 0–4 → PIR float (0.5–5.0)
  if (out.income_ratio !== undefined) out.income_ratio = INCOME_MAP[out.income_ratio] ?? 2.0;

  // snoring_freq: index 0–4 → nilai NHANES SLQ030 (1–5)
  if (out.snoring_freq !== undefined) out.snoring_freq = out.snoring_freq + 1;

  // sleep_apnea: index 0–4 → nilai NHANES SLQ040 (1–5)
  if (out.sleep_apnea !== undefined) out.sleep_apnea = out.sleep_apnea + 1;

  // daytime_sleepy: index 0–3 → nilai NHANES SLQ120 (1–4)
  if (out.daytime_sleepy !== undefined) out.daytime_sleepy = out.daytime_sleepy + 1;

  // gender: index 0/1 sudah sesuai (0=Perempuan, 1=Laki-laki)
  // stress_*: index 0–3 sudah sesuai PHQ (0–3)
  // semua yesno: 0/1 sudah sesuai

  return out;
}

export async function predict(
  answers: Record<string, number>
): Promise<PredictResponse> {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(mapAnswers(answers)),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `Server error: ${res.status}` }));
    throw new Error(err.detail);
  }

  return res.json();
}