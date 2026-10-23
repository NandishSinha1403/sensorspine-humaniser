import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ScholarAI — Intelligent Manuscript Refinement",
  description: "Elite AI-to-Human text humanizer for academic manuscripts. A product by Nandish Sinha.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <nav className="bg-white/70 backdrop-blur-md border-b border-slate-200 sticky top-0 z-50">
          <div className="container mx-auto px-4 md:px-6 h-16 flex items-center justify-between">
            <div className="flex items-center gap-2 group">
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold group-hover:rotate-12 transition-transform shrink-0">S</div>
              <Link href="/" className="text-lg md:text-xl font-bold tracking-tight text-slate-900 truncate">Scholar<span className="text-indigo-600">AI</span></Link>
            </div>
            <div className="flex space-x-4 md:space-x-8 items-center">
              <Link href="/dashboard" className="text-sm font-semibold text-slate-600 hover:text-indigo-600 transition-colors">
                <span className="md:hidden">App</span>
                <span className="hidden md:inline">Humanizer</span>
              </Link>
              <div className="h-4 w-[1px] bg-slate-200 mx-1 md:mx-2 hidden sm:block"></div>
              <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 bg-slate-50 px-2 py-1 rounded hidden sm:block whitespace-nowrap">Beta 3.0</span>
            </div>
          </div>
        </nav>
        <main className="min-h-screen bg-white text-slate-900">
          {children}
        </main>
        <footer className="bg-slate-50 border-t border-slate-200 py-4 text-center">
          <p className="text-[10px] text-slate-400 font-medium tracking-wide">
            ScholarAI — a product by <span className="font-bold text-slate-600">Nandish Sinha</span>
          </p>
        </footer>
      </body>
    </html>
  );
}
