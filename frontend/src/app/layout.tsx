import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Stroke Risk Assessment - NHANES",
  description: "Prediksi risiko stroke berbasis machine learning dari dataset NHANES",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id" className="h-full antialiased" suppressHydrationWarning>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
