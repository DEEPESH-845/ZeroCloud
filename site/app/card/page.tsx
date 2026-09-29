import type { Metadata } from "next";
import CardView from "@/components/CardView";

// Static on purpose: the result lives in the URL fragment, which only the
// browser sees. See lib/card.ts and crates/zc-report/src/card.rs.
export const metadata: Metadata = {
  title: "Result card — ZeroCloud",
  description: "What one machine can run, measured by zc. The result is carried in the link itself; nothing was uploaded.",
};

export default function Page() {
  return <CardView />;
}
