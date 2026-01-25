import type { Metadata } from "next";
import "./globals.css";
import { ToastProvider } from "@/contexts/ToastContext";

export const metadata: Metadata = {
  title: "gradscii-art",
  description: "Generate ASCII art from images",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen">
        <header className="fixed top-0 z-50 w-full bg-[var(--bg)] px-[4ch] pt-4">
          <nav className="flex gap-[4ch] text-xs">
            <a href="/" className="underline decoration-[0.24ch] hover:no-underline">gradscii-art</a>
            <a href="https://github.com/microcon/gradscii-art" target="_blank" className="underline decoration-[0.24ch] hover:no-underline">github</a>
          </nav>
        </header>
        <main style={{ padding: '4rem 4ch 0 4ch' }}>
          <ToastProvider>{children}</ToastProvider>
        </main>
      </body>
    </html>
  );
}
