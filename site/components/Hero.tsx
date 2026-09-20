import CopyButton from "./CopyButton";
import Terminal from "./Terminal";
import HeroIntro from "./HeroIntro";
import MagneticButton from "./MagneticButton";

const H1 = "What can this laptop actually run, and how fast?";

export const INSTALL =
  "curl -fsSL https://raw.githubusercontent.com/DEEPESH-845/ZeroCloud/main/install.sh | sh";

// Soft break opportunities after each "/" so the URL wraps at path boundaries.
export function Wrappable({ text }: { text: string }) {
  return text.split(/(?<=\/)/).map((part, i) => <span key={i}>{part}<wbr /></span>);
}

export default function Hero() {
  return (
    <header className="hero wrap" id="top">
      <p className="eyebrow">zc · local LLM hardware predictor · macOS, Linux, Windows</p>
      <h1 aria-label={H1}>
        {H1.split(" ").map((w, i, a) => (
          <span className="w" key={i} aria-hidden="true"><span className="wi">{w}</span></span>
        )).flatMap((el, i, a) => (i < a.length - 1 ? [el, " "] : [el]))}
      </h1>
      <div className="hero-grid">
        <p className="lede">
          <code>zc</code> measures your machine in about two seconds, then predicts decode speed,
          time to first token and maximum usable context for local LLMs. No account, no upload, no
          telemetry.
        </p>
        <div className="ctas">
          <div className="install">
            <code><Wrappable text={INSTALL} /></code>
            <CopyButton text={INSTALL} />
          </div>
          <p className="alt">
            <MagneticButton className="pill" href="https://github.com/DEEPESH-845/ZeroCloud/releases/latest">Windows .exe</MagneticButton>
            <MagneticButton className="pill" href="https://github.com/DEEPESH-845/ZeroCloud">Read the source</MagneticButton>
          </p>
          <p className="facts">
            <span>under 5 MB</span>
            <span>no runtime deps</span>
            <span>Apache-2.0</span>
            <span>opens no connection</span>
          </p>
        </div>
      </div>
      <Terminal />
      <HeroIntro />
    </header>
  );
}
