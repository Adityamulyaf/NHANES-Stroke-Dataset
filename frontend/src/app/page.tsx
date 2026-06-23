import Link from "next/link";

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

export default function Home() {
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
            <span className="text-teal border-b-[3px] border-teal pb-0.5">
              5 Menit
            </span>
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

      {/* ── Footer ── */}
      <footer className="w-full border-t border-zinc-100 py-6 text-center text-xs text-zinc-400">
        Data tidak disimpan. Hasil hanya untuk tujuan edukasi dan tidak menggantikan diagnosis medis.
      </footer>
    </main>
  );
}