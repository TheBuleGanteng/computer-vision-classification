import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Navigation } from "@/components/navigation";
import { Footer } from "@/components/footer";
import { QueryProvider } from "@/providers/query-client-provider";
import { Toaster } from "sonner";
import { PerformanceMonitorInit } from "@/components/performance-monitor-init";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Onegin:Neural Architecture Explorer",
  description: "Interactive 3D visualization of neural network architectures from hyperparameter optimization",
  icons: {
    icon: "/logo_keytech_favicon.ico",
    shortcut: "/logo_keytech_favicon.ico",
    apple: "/logo_keytech_favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen bg-background font-sans`}
      >
        <QueryProvider>
          <PerformanceMonitorInit />
          <div className="min-h-screen flex flex-col">
            <Navigation />
            <main className="flex-1">
              {children}
            </main>
            <Footer />
          </div>
          <Toaster richColors position="top-right" />
        </QueryProvider>
      </body>
    </html>
  );
}
