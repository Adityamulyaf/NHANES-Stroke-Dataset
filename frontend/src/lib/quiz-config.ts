export type QuestionType = "number" | "yesno" | "scale" | "slider";

export interface QuizQuestion {
  key: string;
  label: string;
  description: string;
  type: QuestionType;
  unit?: string;
  min?: number;
  max?: number;
  step?: number;
  defaultValue?: number;
  scaleLabels?: string[];
  sliderColor?: string;
}

export interface QuizSection {
  title: string;
  description: string;
  questions: QuizQuestion[];
}

// 24 fitur exact sesuai model RF Skenario E (feature_groups.pkl + rf_best_model.pkl)
export const quizSections: QuizSection[] = [
  // ─── CLINICAL (12 fitur) ────────────────────────────────────────────────────
  {
    title: "Klinis & Demografi",
    description: "Informasi dasar tentang kesehatan dan riwayat medis Anda.",
    questions: [
      {
        key: "age",
        label: "Usia",
        description: "Berapa usia Anda saat ini?",
        type: "number",
        unit: "tahun",
        min: 18,
        max: 80,
        step: 1,
        defaultValue: 40,
      },
      {
        key: "education",
        label: "Tingkat Pendidikan",
        description: "Pendidikan tertinggi yang pernah atau sedang Anda tempuh.",
        // DMDEDUC2: index 0→1, 1→2, 2→3, 3→4, 4→5 (di-map di api.ts)
        type: "scale",
        scaleLabels: ["Tidak tamat SD", "SD", "SMP", "SMA / SMK", "Kuliah / Lebih tinggi"],
        defaultValue: 2,
      },
      {
        key: "income_ratio",
        label: "Pendapatan Rumah Tangga per Bulan",
        description: "Perkiraan total pendapatan semua anggota keluarga.",
        // PIR: index 0→0.5, 1→1.0, 2→2.0, 3→3.5, 4→5.0 (di-map di api.ts)
        type: "scale",
        scaleLabels: [
          "< Rp 1 juta",
          "Rp 1–3 juta",
          "Rp 3–6 juta",
          "Rp 6–12 juta",
          "> Rp 12 juta",
        ],
        defaultValue: 2,
      },
      {
        key: "waist_circ",
        label: "Lingkar Pinggang",
        description: "Ukur di titik terkecil pinggang Anda.",
        type: "slider",
        unit: "cm",
        min: 40,
        max: 180,
        step: 1,
        defaultValue: 80,
      },
      {
        key: "systolic_bp",
        label: "Tekanan Darah Sistolik",
        description: "Angka atas pada pengukuran tensi. Jika tidak tahu, isi 120.",
        type: "number",
        unit: "mmHg",
        min: 70,
        max: 200,
        step: 1,
        defaultValue: 120,
      },
      {
        key: "diastolic_bp",
        label: "Tekanan Darah Diastolik",
        description: "Angka bawah pada pengukuran tensi. Jika tidak tahu, isi 80.",
        type: "number",
        unit: "mmHg",
        min: 40,
        max: 130,
        step: 1,
        defaultValue: 80,
      },
      {
        key: "hypertension",
        label: "Hipertensi (Darah Tinggi)",
        description: "Pernah didiagnosis dokter mengidap tekanan darah tinggi?",
        type: "yesno",
        defaultValue: 0,
      },
      {
        key: "diabetes",
        label: "Diabetes",
        description: "Pernah didiagnosis dokter mengidap diabetes atau prediabetes?",
        type: "yesno",
        defaultValue: 0,
      },
      {
        key: "heart_failure",
        label: "Gagal Jantung",
        description: "Pernah didiagnosis dokter mengidap gagal jantung?",
        type: "yesno",
        defaultValue: 0,
      },
      {
        key: "coronary_disease",
        label: "Penyakit Jantung Koroner",
        description: "Pernah didiagnosis dokter mengidap penyakit jantung koroner atau angina?",
        type: "yesno",
        defaultValue: 0,
      },
      {
        key: "heart_attack",
        label: "Serangan Jantung",
        description: "Pernah mengalami serangan jantung sebelumnya?",
        type: "yesno",
        defaultValue: 0,
      },
      {
        key: "ever_smoked",
        label: "Riwayat Merokok",
        description: "Pernah merokok minimal 100 batang sepanjang hidup?",
        type: "yesno",
        defaultValue: 0,
      },
    ],
  },

  // ─── SLEEP (4 fitur) ────────────────────────────────────────────────────────
  {
    title: "Kualitas Tidur",
    description: "Pola dan kualitas tidur Anda dalam 1 bulan terakhir.",
    questions: [
      {
        key: "snoring_freq",
        label: "Frekuensi Mendengkur",
        description: "Seberapa sering Anda mendengkur saat tidur?",
        // SLQ030: index 0→1, 1→2, 2→3, 3→4, 4→5 (di-map di api.ts)
        type: "scale",
        scaleLabels: [
          "Tidak pernah",
          "Jarang (1–2 malam/minggu)",
          "Kadang (3–4 malam/minggu)",
          "Sering (5–6 malam/minggu)",
          "Hampir setiap malam",
        ],
        defaultValue: 0,
      },
      {
        key: "sleep_apnea",
        label: "Henti Napas saat Tidur",
        description: "Seberapa sering Anda berhenti bernapas sejenak saat tidur?",
        // SLQ040: index 0→1, 1→2, 2→3, 3→4, 4→5 (di-map di api.ts)
        type: "scale",
        scaleLabels: [
          "Tidak pernah",
          "Jarang (1–2 malam/minggu)",
          "Kadang (3–4 malam/minggu)",
          "Sering (5–6 malam/minggu)",
          "Hampir setiap malam",
        ],
        defaultValue: 0,
      },
      {
        key: "sleep_problem_doctor",
        label: "Konsultasi Masalah Tidur ke Dokter",
        description: "Pernah menemui dokter atau tenaga medis karena masalah tidur?",
        type: "yesno",
        defaultValue: 0,
      },
      {
        key: "daytime_sleepy",
        label: "Kantuk di Siang Hari",
        description: "Seberapa sering mengantuk berat atau tertidur di siang hari tanpa disengaja?",
        // SLQ120: index 0→1, 1→2, 2→3, 3→4 (di-map di api.ts)
        type: "scale",
        scaleLabels: [
          "Tidak pernah",
          "Jarang (1–2 hari/minggu)",
          "Kadang (3–4 hari/minggu)",
          "Sering (5–6 hari/minggu)",
        ],
        defaultValue: 0,
      },
    ],
  },

  // ─── STRESS (5 fitur) ───────────────────────────────────────────────────────
  {
    title: "Kesehatan Mental",
    description: "Dalam 2 minggu terakhir, seberapa sering Anda mengalami hal berikut?",
    questions: [
      {
        key: "stress_anhedonia",
        label: "Kehilangan Minat atau Kesenangan",
        description: "Kurang bergairah melakukan hal yang biasanya Anda nikmati.",
        // PHQ: 0–3, tidak perlu mapping
        type: "scale",
        scaleLabels: ["Tidak sama sekali", "Beberapa hari", "Lebih dari separuh hari", "Hampir setiap hari"],
        defaultValue: 0,
      },
      {
        key: "stress_depressed",
        label: "Perasaan Sedih atau Putus Asa",
        description: "Merasa murung, sedih, atau tidak punya harapan.",
        type: "scale",
        scaleLabels: ["Tidak sama sekali", "Beberapa hari", "Lebih dari separuh hari", "Hampir setiap hari"],
        defaultValue: 0,
      },
      {
        key: "stress_fatigue",
        label: "Kelelahan atau Kurang Energi",
        description: "Merasa lelah atau tidak bertenaga meski sudah cukup istirahat.",
        type: "scale",
        scaleLabels: ["Tidak sama sekali", "Beberapa hari", "Lebih dari separuh hari", "Hampir setiap hari"],
        defaultValue: 0,
      },
      {
        key: "stress_concentration",
        label: "Sulit Berkonsentrasi",
        description: "Sulit fokus, misalnya saat membaca atau menonton TV.",
        type: "scale",
        scaleLabels: ["Tidak sama sekali", "Beberapa hari", "Lebih dari separuh hari", "Hampir setiap hari"],
        defaultValue: 0,
      },
      {
        key: "stress_self_esteem",
        label: "Perasaan Negatif tentang Diri Sendiri",
        description: "Merasa buruk, gagal, atau mengecewakan diri sendiri / keluarga.",
        type: "scale",
        scaleLabels: ["Tidak sama sekali", "Beberapa hari", "Lebih dari separuh hari", "Hampir setiap hari"],
        defaultValue: 0,
      },
    ],
  },

  // ─── PHYSICAL (3 fitur) ─────────────────────────────────────────────────────
  {
    title: "Aktivitas Fisik",
    description: "Kebiasaan olahraga Anda dalam 1 minggu terakhir.",
    questions: [
      {
        key: "vigorous_leisure",
        label: "Olahraga Berat di Waktu Luang",
        description: "Melakukan olahraga berat di waktu luang? (Lari, bersepeda cepat, aerobik, renang kencang)",
        type: "yesno",
        defaultValue: 0,
      },
      {
        key: "vigorous_leisure_min",
        label: "Total Durasi Olahraga Berat Mingguan",
        description: "Total menit olahraga berat dalam satu minggu terakhir. (Contoh: jika olahraga 30 menit per sesi sebanyak 3 kali seminggu, isi 90. Dianjurkan minimal 75 menit per minggu. Isi 0 jika tidak olahraga)",
        type: "number",
        unit: "menit/minggu",
        min: 0,
        max: 1000,
        step: 5,
        defaultValue: 0,
      },
      {
        key: "moderate_leisure",
        label: "Olahraga Sedang di Waktu Luang",
        description: "Melakukan olahraga sedang di waktu luang? (Jalan cepat, bersepeda santai, yoga)",
        type: "yesno",
        defaultValue: 0,
      },
    ],
  },
];