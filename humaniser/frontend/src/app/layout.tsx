import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Humaniser",
  description: "AI-to-Human Text Humanizer",
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
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold group-hover:rotate-12 transition-transform shrink-0">H</div>
              <Link href="/" className="text-lg md:text-xl font-bold tracking-tight text-slate-900 truncate">Humaniser<span className="text-indigo-600">.</span></Link>
            </div>
            <div className="flex space-x-4 md:space-x-8 items-center">
              <Link href="/dashboard" className="text-sm font-semibold text-slate-600 hover:text-indigo-600 transition-colors">
                <span className="md:hidden">App</span>
                <span className="hidden md:inline">Humanizer</span>
              </Link>
              <a href="http://localhost:8001" target="_blank" rel="noopener noreferrer" className="text-sm font-semibold text-slate-600 hover:text-indigo-600 transition-colors">
                <span className="md:hidden">Train</span>
                <span className="hidden md:inline">Trainer</span>
              </a>
              <div className="h-4 w-[1px] bg-slate-200 mx-1 md:mx-2 hidden sm:block"></div>
              <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 bg-slate-50 px-2 py-1 rounded hidden sm:block whitespace-nowrap">Beta 2.0</span>
            </div>
          </div>
        </nav>
        <main className="min-h-screen bg-white text-slate-900">
          {children}
        </main>
      </body>
    </html>
  );
}
