import CopyButton from "@/components/CopyButton";
import Reveal from "@/components/Reveal";

const INSTALL = "curl -fsSL https://raw.githubusercontent.com/DEEPESH-845/ZeroCloud/main/install.sh | sh";
const GH = "https://github.com/DEEPESH-845/ZeroCloud";

export default function Sections() {
  return (
    <>
      <Reveal />

      <section id="why">
        <div className="wrap two">
          <div className="sticky">
            <p className="eyebrow">why measure</p>
            <h2>Spec sheets are wrong in exactly the ways that matter on cheap hardware.</h2>
            <p className="lede" style={{ marginTop: 20 }}>
              Every one of them is invisible from a model name and a RAM figure. A two-second benchmark catches all seven at once: RAM bandwidth above cache, random reads on the volume your models actually live on, and f32 and int8 compute.
            </p>
          </div>
          <ol className="seven reveal">
            <li><b>RAM</b><div><strong>Single-channel memory.</strong><small>Half the bandwidth the spec implies. Extremely common on budget laptops.</small></div></li>
            <li><b>RAM</b><div><strong>An iGPU quietly holding 1–2 GB of system memory.</strong><small>Your 8 GB machine has less than it says.</small></div></li>
            <li><b>DISK</b><div><strong>A DRAM-less SSD reading 4× slower than advertised.</strong><small>Weights that spill past RAM stream at this speed.</small></div></li>
            <li><b>OS</b><div><strong>WSL2 defaulting to half your host RAM.</strong><small>The budget is what the guest can see, not the sticker.</small></div></li>
            <li><b>HEAT</b><div><strong>Thermal throttling that is already happening.</strong><small>A benchmark on the machine as it is, not as it was when new.</small></div></li>
            <li><b>DISK</b><div><strong>Windows Defender or FileVault in the read path.</strong><small>Measured on the real volume, through the real filesystem.</small></div></li>
            <li><b>DISK</b><div><strong>Models stored in OneDrive or iCloud.</strong><small>The files are stubs that re-download.</small></div></li>
          </ol>
        </div>
      </section>

      <section id="rule">
        <div className="wrap two">
          <div className="sticky">
            <p className="eyebrow">the house rule</p>
            <p className="rule-box">A number is measured, derived from measured inputs, or printed as <code>-</code>.</p>
            <p className="lede" style={{ marginTop: 22 }}>
              No fallback constants. Predictions are ranges with the confidence tier their evidence earns. Ranges narrow as the dataset grows; they are never narrowed by hand.
            </p>
          </div>
          <div className="stack reveal">
            <div>
              <pre className="formula">{`# decode is memory-bound: every token reads the active weights once
t_token  = resident·bytes/BW_ram + (1−resident)·bytes/BW_disk
tok/s    = `}<i>η</i>{`(backend, quant) / t_token
max_ctx  = (usable − weights − compute_buffers) / kv_bytes_per_token`}</pre>
              <p className="cap-label">η is the one term that cannot be derived. It comes from real measured runs, collected by <code>zc verify</code>.</p>
            </div>
            <div>
              <pre className="cap">{`$ zc check Qwen/Qwen3-32B

`}<span className="k">== Qwen/Qwen3-32B ==</span>{`  (published weights, not a quantisation)

  geometry     64L  n_embd 5120  vocab 151936  GQA 8x128
  parameters   `}<span className="n">32.76B</span>{`  from the repo's safetensors metadata
  weights      `}<span className="n">61.02 GiB</span>{`  as published in BF16
  budget       `}<span className="n">12.80 GiB</span>{`  measured on this machine, idle

  `}<span className="no">WON&apos;T FIT</span>{`    the published weights alone are 48.48 GiB over budget

  decode       `}<span className="n">-</span>{`   speed needs the byte count of the quantised file you
                   would actually load, which lives in a separate GGUF
                   repo. `}<span className="d">{`Memory above is arithmetic over numbers the
                   repo states; a speed here would not be.`}</span></pre>
              <p className="cap-label">The dash is the point. Any model on Hugging Face, and the one number it would have to invent is the one it refuses to print.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="what">
        <div className="wrap">
          <p className="eyebrow">what it does</p>
          <h2>Four questions, one binary.</h2>
          <div className="two" style={{ marginTop: 40 }}>
            <table className="reveal">
              <thead><tr><th>command</th><th>answers</th></tr></thead>
              <tbody>
                <tr><td>zc check</td><td>What can this machine run, and how fast? Ranked by verdict, then speed, then context.</td></tr>
                <tr><td>zc plan &lt;model&gt;</td><td>What would it take to run <em>this</em> well? The answer is a bandwidth figure you can check against any spec sheet, not a GPU name.</td></tr>
                <tr><td>zc check &lt;hf-repo&gt;</td><td>Will a model outside the 26-model catalog fit? Arithmetic over what the repository publishes.</td></tr>
                <tr><td>zc verify &lt;model&gt;</td><td>Run it for real for 30 seconds. Predicted versus actual, appended to a dataset on your disk.</td></tr>
                <tr><td>zc share</td><td>Turn that measurement into a pull request. Shows the whole record, then opens your browser. Never a token.</td></tr>
                <tr><td>zc</td><td>With no arguments and a human watching: a table where every row explains itself. Piped or <code>--json</code>, plain text as always.</td></tr>
              </tbody>
            </table>
            <div className="reveal">
              <pre className="cap">{`$ zc plan qwen3-8b --context 32K

`}<span className="k">== plan ==</span>{`  qwen3-8b at 32K context, KV F16, target 10 tok/s

  this machine   12.80 GiB budget, 131 GB/s measured, Metal

  quant     weights      KV    total         needs   on this machine
  Q4_K_M       4.68    4.50     9.40       `}<span className="n">57 GB/s</span>{`   `}<span className="ok">fits</span>{`, 17-28 t/s
  Q5_K_M       5.45    4.50    10.17       `}<span className="n">67 GB/s</span>{`   `}<span className="ok">fits</span>{`, 15-24 t/s
  Q6_K         6.26    4.50    10.98       `}<span className="n">77 GB/s</span>{`   `}<span className="ok">fits</span>{`, 13-21 t/s
  Q8_0         8.11    4.50    12.83       `}<span className="n">92 GB/s</span>{`   over by 0.03 GiB

  `}<span className="d">{`Bandwidth is checkable against a spec sheet. A GPU model name
  would be a lookup, and this tool puts no lookup under a number.`}</span></pre>
              <p className="cap-label">Built for the machines that get told &quot;you need a better GPU&quot;: 8 GB Windows laptops, old Intel Macs, WSL2, Raspberry Pis. It works on a 4090 too.</p>
            </div>
          </div>
        </div>
      </section>

      <section id="accuracy">
        <div className="wrap">
          <p className="eyebrow">accuracy · recompute it with <code>zc gate</code></p>
          <h2>How wrong it has been, out of sample.</h2>
          <div className="stats reveal" style={{ marginTop: 36 }}>
            <div className="stat"><b>9.4<small>%</small></b><span>median error per machine</span></div>
            <div className="stat"><b>8<small>machines</small></b><span>7 hypervisor, 1 bare metal, 15 runs</span></div>
            <div className="stat"><b>66.7<small>%</small></b><span>of measurements landed inside the published range</span></div>
          </div>
          <div className="two" style={{ marginTop: 40 }}>
            <div className="note reveal">
              <b>Pre-1.0. The Phase 0 gate is open.</b> The gate is median error under 25% across at least 5 machines including 2 on bare metal. The median clears it. The bare-metal count does not: there is one, and it is the laptop this was written on. Cloud runners are hypervisors and run 10–30% below real hardware, so a gate passed in CI would prove nothing.
              <p className="gate">after one <code>zc verify</code> on your machine, your own <code>zc gate</code> prints:<br /><span className="ok">PASS</span>  median 11.6% &lt; 25% across 9 machines.</p>
            </div>
            <div className="reveal">
              <p className="eyebrow">the one thing this project needs</p>
              <h3 style={{ fontSize: 22 }}>A real machine that is not this one.</h3>
              <p className="lede" style={{ fontSize: 16, marginTop: 10 }}>
                A Windows laptop, a Linux desktop, an old Intel Mac. Three commands, about twenty minutes, and the record is written to your disk only until you say otherwise.
              </p>
              <ol className="steps" style={{ marginTop: 18 }}>
                <li><div><code>ollama pull qwen3:1.7b</code><small>or measure whatever you already have</small></div></li>
                <li><div><code>zc verify qwen3:1.7b</code><small>30s: predicted vs actual, written to your disk only</small></div></li>
                <li><div><code>zc share</code><small>shows you the record and what is not in it, then offers to open a browser</small></div></li>
              </ol>
            </div>
          </div>
        </div>
      </section>

      <section className="final">
        <div className="wrap">
          <p className="eyebrow" style={{ color: "#8C95A1" }}>install</p>
          <h2>Two seconds to find out.</h2>
          <p className="lede" style={{ marginTop: 16 }}>
            One binary, under 5 MB, statically linked on Linux. It opens no connection unless you hand it a Hugging Face repo, and it prints the URL first when it does.
          </p>
          <div className="ctas" style={{ marginTop: 32, maxWidth: 720 }}>
            <div className="install">
              <code>{INSTALL}</code>
              <CopyButton text={INSTALL} />
            </div>
            <p className="alt"><a href={`${GH}/releases/latest`}>Windows .exe from Releases</a><a href={`${GH}#install`}>Build from source</a></p>
            <p className="facts"><span>cargo install --git {GH} zc-cli</span></p>
          </div>
        </div>
      </section>

      <footer>
        <div className="wrap">
          <span>ZeroCloud · Apache-2.0 · no telemetry, no account, no token</span>
          <span><a href={GH}>GitHub</a> · <a href={`${GH}/blob/main/CONTRIBUTING.md`}>Contributing</a> · <a href={`${GH}/blob/main/SECURITY.md`}>Security</a></span>
        </div>
      </footer>
    </>
  );
}
