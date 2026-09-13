import type { Metadata } from "next";
import { Instrument_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const sans = Instrument_Sans({ variable: "--font-sans", subsets: ["latin"], weight: ["400", "500", "600", "700"] });
const mono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"], weight: ["400", "500", "700"] });

export const metadata: Metadata = {
  title: "ZeroCloud — What can this laptop actually run, and how fast?",
  description:
    "zc measures your machine in about two seconds, then predicts decode speed, time to first token and maximum usable context for local LLMs. No account, no upload, no telemetry.",
  openGraph: {
    title: "ZeroCloud — What can this laptop actually run, and how fast?",
    description: "A 5 MB binary that measures your machine and predicts which local LLMs will run, at what speed. Every number is checkable.",
    type: "website",
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
