import CopyButton from "./CopyButton";
import Terminal from "./Terminal";

export const INSTALL =
  "curl -fsSL https://raw.githubusercontent.com/DEEPESH-845/ZeroCloud/main/install.sh | sh";

export default function Hero() {
  return (
    <header className="hero wrap" id="top">
      <p className="eyebrow">zc · local LLM hardware predictor · macOS, Linux, Windows</p>
      <h1>What can this laptop actually run, and how fast?</h1>
      <div className="hero-grid">
        <p className="lede">
          <code>zc</code> measures your machine in about two seconds, then predicts decode speed,
          time to first token and maximum usable context for local LLMs. No account, no upload, no
          telemetry.
        </p>
        <div className="ctas">
          <div className="install">
            <code>{INSTALL}</code>
            <CopyButton text={INSTALL} />
          </div>
          <p className="alt">
            <a href="https://github.com/DEEPESH-845/ZeroCloud/releases/latest">Windows .exe</a>
            <a href="https://github.com/DEEPESH-845/ZeroCloud">Read the source</a>
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
    </header>
  );
}
