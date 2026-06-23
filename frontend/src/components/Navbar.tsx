"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const pathname = usePathname();

  // Jika kita berada di halaman kuis atau hasil, kita bisa menampilkan versi navbar yang lebih bersih
  // Namun untuk saat ini kita tampilkan navbar lengkap di semua halaman
  const isQuizMode = pathname === "/quiz" || pathname === "/result";

  return (
    <header className="sticky top-0 z-50 w-full border-b border-zinc-100 bg-white/80 backdrop-blur-md shadow-sm">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        
        {/* Logo Kiri Atas */}
        <Link href="/" className="flex items-center gap-2.5 transition hover:opacity-80">
          <Image 
            src="/favicon.png" 
            alt="Stroke Assessment Logo" 
            width={36} 
            height={36} 
            className="object-contain"
          />
          <span className="text-lg font-bold tracking-tight text-teal-dark hidden sm:inline-block">
            StrokeRisk AI
          </span>
        </Link>
        
        {/* Menu Navigasi Tengah (Disembunyikan di mode kuis agar user fokus) */}
        {!isQuizMode && (
          <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-zinc-500">
            <Link href="/" className={`hover:text-teal transition ${pathname === "/" ? "text-teal font-semibold" : ""}`}>
              Beranda
            </Link>
            <Link href="/#cara-kerja" className="hover:text-teal transition">
              Cara Kerja
            </Link>
          </nav>
        )}

        {/* Tombol Aksi Kanan */}
        <div className="flex items-center gap-3">
          {!isQuizMode ? (
            <Link
              href="/quiz"
              className="rounded-full bg-teal px-5 py-2 text-sm font-semibold text-white transition hover:bg-teal-dark shadow-sm hover:shadow"
            >
              Mulai Assessment
            </Link>
          ) : (
            <Link
              href="/"
              className="text-xs font-medium text-zinc-400 hover:text-zinc-600 transition flex items-center gap-1"
            >
              Batalkan
            </Link>
          )}
        </div>

      </div>
    </header>
  );
}
