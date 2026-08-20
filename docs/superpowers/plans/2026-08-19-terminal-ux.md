# Terminal UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **STATUS: complete, 2026-08-20.** All eight tasks landed, plus four items
> the plan did not call for: the crates.io packaging fix, `zc check
> <hf-repo-id>` (parity §2.14), `scripts/tui_smoke.py` and
> `scripts/contract_smoke.py`. 218 tests, clippy clean, CI green on ubuntu,
> macOS and Windows. `main` is at a393a00.
>
> Twelve bugs were found and fixed along the way; the four that mattered most
> were invisible to the unit suite because nothing automated had pressed a key,
> and two more were invisible on this laptop because its CPU brand is short and
> its RAM is large. Both classes now have guards. See VERIFICATION.md.
>
> **Not done, and not doable here:** the v0.1.0 tag. Creating and pushing it is
> blocked by the harness permission classifier, so it needs a human hand. The
> release is otherwise prepared and `docs/publishing.md` carries the command.

**Goal:** Give `zc` a default-on interactive TUI for `check`, live benchmark progress, and a table that fits an 80-column terminal — without changing a single byte of any non-TTY output.

**Architecture:** A new `crates/zc-tui` owns the only crossterm dependency; every other crate stays dependency-free. Row ranking and collapsing move from `zc-cli` into `zc-report` so the static and interactive surfaces sort identically. TUI state and rendering are pure functions over a `State` struct, unit-tested with no terminal; only the event loop touches crossterm.

**Tech Stack:** Rust 2024 edition, workspace crates, `crossterm` (the sole new dependency), `libc` (existing, Unix-only).

**Spec:** `docs/superpowers/specs/2026-08-19-terminal-ux-design.md`

## Global Constraints

- **Non-TTY output must stay byte-identical.** `zc check | head`, `zc check --json`, CI and agent usage all depend on it. Task 6 tests this explicitly against a saved golden file.
- **crossterm may only be named in `crates/zc-tui`.** `zc-report`, `zc-model`, `zc-probe`, `zc-bench` stay dependency-free.
- **No line of terminal output may carry trailing whitespace.** Existing rule, enforced by `push()` in `zc-report/src/text.rs`.
- **No line of `zc check` output may exceed 80 columns.** `zc doctor` is exempt — it is Markdown for a GitHub issue where soft wrap is correct.
- **A number is measured, derived from measured inputs, or printed as `-`.** Never substitute a plausible constant. This applies to every new field the detail pane shows.
- **Colour never changes a column's width.** Pad first, paint second — see the existing test `colour_never_changes_a_column_width`.
- **No emoji.** Double-width glyphs misalign tables and render as boxes on legacy Windows consoles.
- **Edition 2024, resolver 3.** Dependencies are declared in the workspace `Cargo.toml` and inherited with `crossterm.workspace = true`.
- Every task ends green on `cargo test --workspace` and `cargo clippy --workspace --all-targets -- -D warnings`.

---

### Task 1: Shared ranking, and the 80-column fix

`rank` and `best_per_model` currently live private in `crates/zc-cli/src/check.rs`. The TUI must sort and collapse rows identically or the two surfaces disagree about what a machine can run — the exact drift `zc-report`'s module doc exists to prevent. Move them, then fix the column overflow that motivated this phase.

**Files:**
- Modify: `crates/zc-report/src/lib.rs` (add `pub fn rank`, `pub fn best_per_model`)
- Modify: `crates/zc-cli/src/check.rs` (delete both, call through)
- Modify: `crates/zc-report/src/text.rs:203-262` (dynamic model column)
- Test: inline `#[cfg(test)]` in `crates/zc-report/src/lib.rs` and `text.rs`

**Interfaces:**
- Consumes: `Row<'a>`, `Verdict`, `Prediction` — all already in `zc-report`/`zc-model`.
- Produces:
  - `pub fn rank(row: &Row) -> (u8, i64, i64)`
  - `pub fn best_per_model<'a>(rows: Vec<Row<'a>>) -> Vec<Row<'a>>`
  - `pub fn model_col_width(rows: &[Row]) -> usize` — clamped to `[12, 28]`

- [ ] **Step 1: Write the failing test for the column width**

Add to `crates/zc-report/src/lib.rs`:

```rust
#[cfg(test)]
mod width_tests {
    use super::*;

    /// A constrained machine lists short model ids, so the table should narrow
    /// to fit rather than wrapping. The 93-column row that motivated this was
    /// a fixed 28-character column carrying "qwen3-1.7b".
    #[test]
    fn model_column_shrinks_to_the_widest_visible_row() {
        assert_eq!(super::clamp_model_width(10), 12); // floor
        assert_eq!(super::clamp_model_width(20), 20);
        assert_eq!(super::clamp_model_width(40), 28); // ceiling, never truncates
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo test -p zc-report clamp_model_width`
Expected: FAIL — `cannot find function 'clamp_model_width'`

- [ ] **Step 3: Implement the width helpers**

Add to `crates/zc-report/src/lib.rs`:

```rust
/// Width of the model column, given the widest id actually being shown.
///
/// Floor of 12 keeps the header readable when every row is short. Ceiling of
/// 28 is the widest catalog id today; a longer one overflows its column rather
/// than being truncated, because a truncated model id is not a model id.
pub fn clamp_model_width(widest: usize) -> usize {
    widest.clamp(12, 28)
}

/// Width the model column needs for these rows.
pub fn model_col_width(rows: &[Row]) -> usize {
    clamp_model_width(rows.iter().map(|r| r.model_id.len()).max().unwrap_or(12))
}
```

- [ ] **Step 4: Run and watch it pass**

Run: `cargo test -p zc-report clamp_model_width`
Expected: PASS

- [ ] **Step 5: Move `rank` and `best_per_model` into `zc-report`**

Cut both functions verbatim from `crates/zc-cli/src/check.rs` — including their doc comments, which carry the reasoning about why this is not a composite score — and paste them into `crates/zc-report/src/lib.rs` with `pub` added. In `check.rs`, delete the definitions and change the two call sites:

```rust
models = zc_report::best_per_model(models);
models.sort_by_key(zc_report::rank);
```

`check.rs` no longer needs `use zc_model::Verdict;` — remove it if clippy flags it as unused.

- [ ] **Step 6: Apply the dynamic width in the text renderer**

In `crates/zc-report/src/text.rs`, immediately before the header `push!` at line ~203:

```rust
let mw = crate::model_col_width(&r.models);
```

Change the header format string from the fixed `{:<28}` to a runtime width, and the same for the row format:

```rust
// header
&format!(
    "  {:<4} {:<mw$} {:<7} {:>12} {:>7} {:>6}  {:<6}",
    "", "model", "quant", "decode tok/s", "max ctx", "TTFT", "conf"
),
// row
&format!(
    "  {} {:<mw$} {:<7} {:>12} {:>7} {:>6}  {:<6}{}",
    mark, row.model_id, row.quant.name, speed, ctx, ttft,
    pr.confidence.label(),
    if pr.resident_fraction < 0.999 { ... } else { String::new() }
),
```

`mw$` reads the `mw` binding by name — no extra positional argument needed.

- [ ] **Step 7: Add the 80-column regression test**

Add to `crates/zc-report/src/text.rs`'s existing `mod tests`:

```rust
/// The row that motivated this phase was 93 columns: a fixed 28-wide model
/// column plus a "% resident" suffix. A row that spills is exactly the row a
/// low-end machine shows, so it is the one that must fit.
#[test]
fn a_spilling_row_fits_eighty_columns() {
    let mw = super::super::clamp_model_width("qwen3-30b-a3b".len());
    let row = format!(
        "  {} {:<mw$} {:<7} {:>12} {:>7} {:>6}  {:<6}{}",
        "OK  ", "qwen3-30b-a3b", "IQ4_XS", "10.7-17.8", "2K", "1.2s", "low",
        "  89% resident"
    );
    assert!(row.len() <= 80, "row was {} columns: {row}", row.len());
}
```

- [ ] **Step 8: Run the full suite and clippy**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings`
Expected: all PASS, no warnings

- [ ] **Step 9: Verify against the real binary**

```bash
cargo build --release
./target/release/zc check --all | awk 'length>80' | head
```
Expected: no output. Every row fits.

- [ ] **Step 10: Commit**

```bash
git add crates/zc-report/src/lib.rs crates/zc-report/src/text.rs crates/zc-cli/src/check.rs
git commit -m "fix: a spilling row ran to 93 columns on the machines it describes"
```

---

### Task 2: Charset with ASCII fallback

**Files:**
- Create: `crates/zc-report/src/charset.rs`
- Modify: `crates/zc-report/src/lib.rs` (add `pub mod charset;`)

**Interfaces:**
- Produces:
  - `pub enum Charset { Unicode, Ascii }`
  - `pub fn detect() -> Charset`
  - `impl Charset` methods: `resident(&self) -> &'static str`, `partial`, `wont_fit`, `rule`, `vrule`, `sep`, `spinner(&self, tick: usize) -> &'static str`

- [ ] **Step 1: Write the failing tests**

Create `crates/zc-report/src/charset.rs`:

```rust
//! Which glyphs this terminal can render.
//!
//! Resolved once from the environment and threaded through every renderer, so
//! no literal box-drawing character appears anywhere else in the codebase.
//!
//! Detection reads environment variables and makes no system calls. Probing
//! `GetConsoleOutputCP` would mean an `extern "system"` block in a crate that
//! has none, on a platform `VERIFICATION.md` records as never executed here.
//! Guessing Ascii on an unknown console renders everywhere; guessing Unicode
//! wrong prints replacement boxes.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Charset {
    Unicode,
    Ascii,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_locales_get_unicode() {
        assert_eq!(from_env(Some("xterm"), Some("en_US.UTF-8"), false), Charset::Unicode);
        assert_eq!(from_env(Some("xterm"), Some("C.utf8"), false), Charset::Unicode);
    }

    #[test]
    fn a_non_utf8_or_dumb_terminal_gets_ascii() {
        assert_eq!(from_env(Some("xterm"), Some("C"), false), Charset::Ascii);
        assert_eq!(from_env(Some("dumb"), Some("en_US.UTF-8"), false), Charset::Ascii);
        assert_eq!(from_env(None, Some("en_US.UTF-8"), false), Charset::Ascii);
    }

    /// The escape hatch, and what every other test in the workspace sets.
    #[test]
    fn zc_ascii_overrides_everything() {
        assert_eq!(from_env(Some("xterm"), Some("en_US.UTF-8"), true), Charset::Ascii);
    }

    /// Both charsets must produce the same column count, or a table built for
    /// one wraps in the other.
    #[test]
    fn every_glyph_is_one_column_wide_in_both() {
        for c in [Charset::Unicode, Charset::Ascii] {
            for g in [c.resident(), c.partial(), c.wont_fit(), c.rule(), c.vrule(), c.sep()] {
                assert_eq!(g.chars().count(), 1, "{g:?} in {c:?} is not one column");
            }
            for tick in 0..12 {
                assert_eq!(c.spinner(tick).chars().count(), 1);
            }
        }
    }
}
```

- [ ] **Step 2: Run and watch it fail**

Run: `cargo test -p zc-report charset`
Expected: FAIL — `from_env` and the glyph methods are not defined

- [ ] **Step 3: Implement**

Append to `crates/zc-report/src/charset.rs`:

```rust
/// Pure so it can be tested without touching the process environment.
fn from_env(term: Option<&str>, locale: Option<&str>, force_ascii: bool) -> Charset {
    if force_ascii {
        return Charset::Ascii;
    }
    match term {
        None | Some("dumb") | Some("") => return Charset::Ascii,
        Some(_) => {}
    }
    // On Windows, `TERM` being set at all means Git Bash, WSL or MSYS, each of
    // which is UTF-8. Windows Terminal sets `WT_SESSION` instead and is also
    // UTF-8; legacy conhost sets neither and gets Ascii, which is correct.
    if cfg!(windows) {
        return Charset::Unicode;
    }
    match locale {
        Some(l) if l.to_ascii_lowercase().replace('-', "").contains("utf8") => Charset::Unicode,
        _ => Charset::Ascii,
    }
}

pub fn detect() -> Charset {
    let term = std::env::var("TERM").ok().or_else(|| {
        // Windows Terminal sets no TERM but renders UTF-8 fine.
        std::env::var("WT_SESSION").ok().map(|_| "wt".to_string())
    });
    let locale = ["LC_ALL", "LC_CTYPE", "LANG"]
        .iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()));
    let force = std::env::var("ZC_ASCII").is_ok_and(|v| v == "1");
    from_env(term.as_deref(), locale.as_deref(), force)
}

impl Charset {
    pub fn resident(&self) -> &'static str {
        match self { Charset::Unicode => "●", Charset::Ascii => "*" }
    }
    pub fn partial(&self) -> &'static str {
        match self { Charset::Unicode => "◐", Charset::Ascii => "o" }
    }
    pub fn wont_fit(&self) -> &'static str {
        match self { Charset::Unicode => "○", Charset::Ascii => "." }
    }
    pub fn rule(&self) -> &'static str {
        match self { Charset::Unicode => "─", Charset::Ascii => "-" }
    }
    pub fn vrule(&self) -> &'static str {
        match self { Charset::Unicode => "│", Charset::Ascii => "|" }
    }
    pub fn sep(&self) -> &'static str {
        match self { Charset::Unicode => "·", Charset::Ascii => "-" }
    }
    /// Frame `tick` of the spinner. Wraps, so a caller may increment forever.
    pub fn spinner(&self, tick: usize) -> &'static str {
        const BRAILLE: [&str; 10] = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"];
        const ASCII: [&str; 4] = ["-", "\\", "|", "/"];
        match self {
            Charset::Unicode => BRAILLE[tick % BRAILLE.len()],
            Charset::Ascii => ASCII[tick % ASCII.len()],
        }
    }
}
```

Add `pub mod charset;` to `crates/zc-report/src/lib.rs` beside the other module declarations.

- [ ] **Step 4: Run and watch it pass**

Run: `cargo test -p zc-report charset`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add crates/zc-report/src/charset.rs crates/zc-report/src/lib.rs
git commit -m "feat: one place that knows which glyphs a terminal can draw"
```

---

### Task 3: The `zc-tui` crate and its state machine

Pure state and key handling. No terminal, no crossterm calls yet — every test in this task runs headless.

**Files:**
- Create: `crates/zc-tui/Cargo.toml`
- Create: `crates/zc-tui/src/lib.rs`
- Create: `crates/zc-tui/src/state.rs`
- Modify: `Cargo.toml` (workspace: add `crossterm` to `[workspace.dependencies]`)

**Interfaces:**
- Consumes: `zc_report::{Row, Report, rank, best_per_model}`, `zc_model::Verdict`
- Produces:
  - `pub enum Key { Up, Down, PageUp, PageDown, Home, End, Enter, Slash, Esc, Char(char), Backspace, Quit }`
  - `pub enum Action { Redraw, Quit }`
  - `pub enum Sort { Verdict, Decode, Context }`
  - `pub struct State` with `pub fn new(total_rows: usize) -> State`
  - `pub fn on_key(&mut State, Key) -> Action`
  - `State::visible(&self, all: &[Row]) -> Vec<usize>` — indices into the full row set

- [ ] **Step 1: Create the crate manifest**

`crates/zc-tui/Cargo.toml`:

```toml
[package]
name = "zc-tui"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true

[dependencies]
zc-report = { path = "../zc-report" }
zc-model = { path = "../zc-model" }

# The only crate in this workspace permitted to name crossterm. It buys raw
# mode, key-event parsing and resize handling on Windows -- the one layer
# VERIFICATION.md records as never executed here, and therefore the one layer
# not worth writing blind.
crossterm.workspace = true
```

Add to the workspace `Cargo.toml` under `[workspace.dependencies]`:

```toml
crossterm = "0.29"
```

Add `zc-tui = { path = "../zc-tui" }` to `crates/zc-cli/Cargo.toml`'s `[dependencies]`.

- [ ] **Step 2: Write the failing state tests**

Create `crates/zc-tui/src/state.rs` with only the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn st(n: usize) -> State { State::new(n) }

    #[test]
    fn the_cursor_cannot_leave_the_row_set() {
        let mut s = st(3);
        on_key(&mut s, Key::Up);
        assert_eq!(s.cursor, 0, "up at the top must stay at the top");
        for _ in 0..10 { on_key(&mut s, Key::Down); }
        assert_eq!(s.cursor, 2, "down at the bottom must stay at the bottom");
    }

    #[test]
    fn an_empty_row_set_is_navigable_without_panicking() {
        let mut s = st(0);
        on_key(&mut s, Key::Down);
        on_key(&mut s, Key::End);
        on_key(&mut s, Key::PageDown);
        assert_eq!(s.cursor, 0);
    }

    #[test]
    fn home_and_end_jump_to_the_ends() {
        let mut s = st(50);
        on_key(&mut s, Key::End);
        assert_eq!(s.cursor, 49);
        on_key(&mut s, Key::Home);
        assert_eq!(s.cursor, 0);
    }

    #[test]
    fn sort_cycles_through_three_and_returns() {
        let mut s = st(5);
        assert_eq!(s.sort, Sort::Verdict);
        on_key(&mut s, Key::Char('s'));
        assert_eq!(s.sort, Sort::Decode);
        on_key(&mut s, Key::Char('s'));
        assert_eq!(s.sort, Sort::Context);
        on_key(&mut s, Key::Char('s'));
        assert_eq!(s.sort, Sort::Verdict);
    }

    /// Typing into the filter must not be interpreted as a command, or a user
    /// searching for "smollm2" would toggle sort on the 's'.
    #[test]
    fn keys_typed_into_the_filter_are_text_not_commands() {
        let mut s = st(5);
        on_key(&mut s, Key::Slash);
        assert!(s.filtering);
        on_key(&mut s, Key::Char('s'));
        on_key(&mut s, Key::Char('q'));
        assert_eq!(s.filter, "sq");
        assert_eq!(s.sort, Sort::Verdict, "'s' was text, not a sort command");
        on_key(&mut s, Key::Backspace);
        assert_eq!(s.filter, "s");
    }

    #[test]
    fn esc_leaves_the_filter_and_then_quits() {
        let mut s = st(5);
        on_key(&mut s, Key::Slash);
        on_key(&mut s, Key::Char('x'));
        assert_eq!(on_key(&mut s, Key::Esc), Action::Redraw);
        assert!(!s.filtering);
        assert_eq!(s.filter, "", "leaving the filter clears it");
        assert_eq!(on_key(&mut s, Key::Esc), Action::Quit);
    }

    #[test]
    fn q_quits_but_only_outside_the_filter() {
        let mut s = st(5);
        on_key(&mut s, Key::Slash);
        assert_eq!(on_key(&mut s, Key::Char('q')), Action::Redraw);
        on_key(&mut s, Key::Esc);
        assert_eq!(on_key(&mut s, Key::Char('q')), Action::Quit);
    }

    #[test]
    fn enter_toggles_the_detail_pane() {
        let mut s = st(5);
        assert!(!s.detail);
        on_key(&mut s, Key::Enter);
        assert!(s.detail);
        on_key(&mut s, Key::Enter);
        assert!(!s.detail);
    }

    /// Narrowing the row set under a cursor that sat past the new end used to
    /// be the crash: the cursor must be pulled back into range.
    #[test]
    fn shrinking_the_row_set_pulls_the_cursor_back() {
        let mut s = st(50);
        on_key(&mut s, Key::End);
        assert_eq!(s.cursor, 49);
        s.set_len(3);
        assert_eq!(s.cursor, 2);
        s.set_len(0);
        assert_eq!(s.cursor, 0);
    }
}
```

- [ ] **Step 3: Run and watch it fail**

Run: `cargo test -p zc-tui`
Expected: FAIL — nothing in `state.rs` is defined

- [ ] **Step 4: Implement the state machine**

Prepend to `crates/zc-tui/src/state.rs`:

```rust
//! What the TUI is showing, and what a keystroke does to it.
//!
//! Deliberately free of crossterm and of any terminal at all: every branch
//! here is reachable from a unit test. Only `run.rs` translates real events
//! into these `Key` values and paints the result.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Up, Down, PageUp, PageDown, Home, End,
    Enter, Slash, Esc, Backspace, Quit,
    Char(char),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action { Redraw, Quit }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sort { Verdict, Decode, Context }

impl Sort {
    fn next(self) -> Sort {
        match self {
            Sort::Verdict => Sort::Decode,
            Sort::Decode => Sort::Context,
            Sort::Context => Sort::Verdict,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Sort::Verdict => "verdict",
            Sort::Decode => "speed",
            Sort::Context => "context",
        }
    }
}

pub struct State {
    pub cursor: usize,
    pub top: usize,
    pub len: usize,
    pub filter: String,
    pub filtering: bool,
    pub sort: Sort,
    pub show_all: bool,
    pub detail: bool,
    pub help: bool,
    /// Body height in rows, set by the last frame. Page moves need it.
    pub page: usize,
}

impl State {
    pub fn new(len: usize) -> State {
        State {
            cursor: 0, top: 0, len,
            filter: String::new(), filtering: false,
            sort: Sort::Verdict, show_all: false,
            detail: false, help: false, page: 10,
        }
    }

    /// Tell the state the row set changed size — after a filter or an `a`
    /// toggle. Pulls the cursor back into range rather than letting a later
    /// index panic.
    pub fn set_len(&mut self, len: usize) {
        self.len = len;
        self.cursor = self.cursor.min(len.saturating_sub(1));
        self.top = self.top.min(self.cursor);
    }

    fn move_by(&mut self, delta: isize) {
        if self.len == 0 {
            self.cursor = 0;
            return;
        }
        let last = self.len - 1;
        self.cursor = (self.cursor as isize + delta).clamp(0, last as isize) as usize;
    }

    /// Scroll so the cursor is visible in a body `height` rows tall.
    pub fn scroll_into_view(&mut self, height: usize) {
        self.page = height.max(1);
        if self.cursor < self.top {
            self.top = self.cursor;
        } else if height > 0 && self.cursor >= self.top + height {
            self.top = self.cursor + 1 - height;
        }
    }
}

pub fn on_key(s: &mut State, k: Key) -> Action {
    // The help overlay swallows everything except the keys that dismiss it.
    if s.help {
        s.help = !matches!(k, Key::Esc | Key::Enter | Key::Char('?') | Key::Char('q'));
        return Action::Redraw;
    }
    // Inside the filter, printable keys are text. Otherwise a user searching
    // for "smollm2" would toggle sort on the 's'.
    if s.filtering {
        match k {
            Key::Char(c) => s.filter.push(c),
            Key::Backspace => { s.filter.pop(); }
            Key::Esc => { s.filtering = false; s.filter.clear(); }
            Key::Enter => s.filtering = false,
            Key::Quit => return Action::Quit,
            _ => {}
        }
        return Action::Redraw;
    }
    match k {
        Key::Up => s.move_by(-1),
        Key::Down => s.move_by(1),
        Key::PageUp => s.move_by(-(s.page as isize)),
        Key::PageDown => s.move_by(s.page as isize),
        Key::Home => s.cursor = 0,
        Key::End => s.cursor = s.len.saturating_sub(1),
        Key::Enter => s.detail = !s.detail,
        Key::Slash => s.filtering = true,
        Key::Char('s') => s.sort = s.sort.next(),
        Key::Char('a') => s.show_all = !s.show_all,
        Key::Char('?') => s.help = true,
        Key::Char('q') | Key::Esc | Key::Quit => return Action::Quit,
        _ => {}
    }
    Action::Redraw
}
```

Create `crates/zc-tui/src/lib.rs`:

```rust
//! The interactive surface of `zc check`.
//!
//! Split so that everything except the event loop is testable without a
//! terminal: `state` owns what is shown and what a key does to it, `frame`
//! turns that into lines, and `run` is the only module that talks to
//! crossterm.

pub mod state;
```

- [ ] **Step 5: Run and watch it pass**

Run: `cargo test -p zc-tui`
Expected: PASS (9 tests)

- [ ] **Step 6: Clippy**

Run: `cargo clippy -p zc-tui --all-targets -- -D warnings`
Expected: no warnings

- [ ] **Step 7: Commit**

```bash
git add crates/zc-tui Cargo.toml crates/zc-cli/Cargo.toml
git commit -m "feat: the TUI's state machine, with no terminal in sight"
```

---

### Task 4: Frame rendering

Turn `State` plus rows into lines. Still pure — `frame` takes a width and height and returns `Vec<String>`.

**Files:**
- Create: `crates/zc-tui/src/frame.rs`
- Modify: `crates/zc-tui/src/lib.rs` (add `pub mod frame;`)

**Interfaces:**
- Consumes: `State`, `Sort` (Task 3); `zc_report::{Row, Report, model_col_width}`, `zc_report::charset::Charset` (Tasks 1–2)
- Produces: `pub fn frame(r: &Report, rows: &[usize], s: &State, cs: Charset, w: usize, h: usize) -> Vec<String>`
  where `rows` are indices into `r.models`.

- [ ] **Step 1: Write the failing tests**

Create `crates/zc-tui/src/frame.rs` with the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use zc_report::charset::Charset;

    /// A frame that overruns its width corrupts the whole screen: crossterm
    /// wraps, every later line lands one row low, and the footer scrolls off.
    #[test]
    fn no_line_ever_exceeds_the_width_it_was_given() {
        for w in [40usize, 60, 80, 120, 200] {
            for lines in fixture_frame(w, 24) {
                assert!(lines.chars().count() <= w, "{} cols at w={w}: {lines}", lines.chars().count());
            }
        }
    }

    /// The existing rule for every surface in this project.
    #[test]
    fn no_line_carries_trailing_whitespace() {
        for l in fixture_frame(80, 24) {
            assert_eq!(l.trim_end(), l, "trailing whitespace: {l:?}");
        }
    }

    /// A frame must fill exactly the height it was given, or the previous
    /// screen shows through underneath.
    #[test]
    fn a_frame_is_exactly_as_tall_as_requested() {
        for h in [8usize, 24, 50] {
            assert_eq!(fixture_frame(80, h).len(), h);
        }
    }

    /// Both charsets must lay out identically, or a table built for one wraps
    /// in the other.
    #[test]
    fn ascii_and_unicode_frames_have_the_same_shape() {
        let u = fixture_frame_cs(80, 24, Charset::Unicode);
        let a = fixture_frame_cs(80, 24, Charset::Ascii);
        assert_eq!(u.len(), a.len());
        for (x, y) in u.iter().zip(a.iter()) {
            assert_eq!(x.chars().count(), y.chars().count(), "\n{x}\n{y}");
        }
    }

    /// A terminal too small to be useful must say so rather than render
    /// something unreadable.
    #[test]
    fn a_tiny_terminal_gets_a_message_not_a_broken_table() {
        let f = fixture_frame(20, 4);
        assert_eq!(f.len(), 4);
        assert!(f.iter().any(|l| l.contains("too small")), "{f:?}");
    }
}
```

The fixture helpers build a `Report` — see Step 3.

- [ ] **Step 2: Run and watch it fail**

Run: `cargo test -p zc-tui frame`
Expected: FAIL — `frame` and the fixtures are not defined

- [ ] **Step 3: Write the test fixture**

`Report` borrows, so the fixture must own its parts and hand out references. Add to the `tests` module in `frame.rs`:

```rust
    use zc_bench::{compute::ComputeResult, disk::DiskResult, ram::RamResult};
    use zc_model::{Backend, Confidence, Prediction, Quant, QuantFamily, Verdict};
    use zc_probe::{cpu::Cpu, env::Env, memory::Memory, storage::Storage};
    use zc_report::{Assumptions, Report, Row};

    fn fixture_frame(w: usize, h: usize) -> Vec<String> {
        fixture_frame_cs(w, h, Charset::Unicode)
    }

    fn fixture_frame_cs(w: usize, h: usize, cs: Charset) -> Vec<String> {
        let quant = Quant { name: "Q4_K_M".into(), bytes: 4_800_000_000, family: QuantFamily::KQuant };
        let ids = ["qwen3-1.7b", "llama-3.1-8b", "deepseek-r1-distill-llama-8b"];
        let cpu = Cpu::default();
        let mem = Memory::default();
        let env = Env::default();
        let storage = Storage::default();
        let ram = RamResult::default();
        let compute = ComputeResult::default();
        let disk = DiskResult::default();
        let models: Vec<Row> = ids.iter().enumerate().map(|(i, id)| Row {
            model_id: id,
            quant: &quant,
            prediction: Prediction {
                resident_fraction: if i == 0 { 1.0 } else { 0.84 },
                decode_tok_s: (10.3, 17.2),
                prefill_tok_s: None,
                ttft_s: Some(3.4),
                prefill_confidence: Confidence::Low,
                max_context: 37_000,
                kv_bytes_per_token: 524_288,
                verdict: Verdict::Good,
                raw_seconds_per_token: 0.05,
                assumed_eta: 0.875,
                confidence: Confidence::Low,
            },
        }).collect();
        let report = Report {
            cpu: &cpu, mem: &mem, env: &env, storage: &storage, gpus: &[],
            ram: &ram, compute: &compute, disk: Some(&disk),
            backend: Backend::Cpu,
            ram_bw_gbs: 125.0, vram_bw_gbs: 0.0, vram_bytes: 0, disk_gbs: 5.0,
            budget_idle: 13_743_895_347, budget_now: 1_320_702_444,
            assumptions: Assumptions {
                prompt_tokens: 2048, ubatch: 512, kv_precision: "F16",
                idle_machine: true, uncalibrated: false,
                prefill_unmeasured: false, total_rows: 3,
            },
            models,
        };
        let idx: Vec<usize> = (0..report.models.len()).collect();
        let s = State::new(idx.len());
        super::frame(&report, &idx, &s, cs, w, h)
    }
```

If any of `Cpu`, `Memory`, `Env`, `Storage`, `RamResult`, `ComputeResult`, `DiskResult` lacks `Default`, add `#[derive(Default)]` to it in its own crate — these are plain data structs and a default is only used by tests. Run `cargo test -p zc-tui` after each addition to find the next one.

- [ ] **Step 4: Implement `frame`**

Prepend to `crates/zc-tui/src/frame.rs`:

```rust
//! `State` + rows -> the exact lines to paint.
//!
//! Pure, and returns exactly `h` lines each at most `w` columns. Both are
//! load-bearing: a frame that overruns its width makes crossterm wrap, which
//! pushes every later line one row down and scrolls the footer off screen.

use crate::state::State;
use zc_report::charset::Charset;
use zc_report::{model_col_width, Report};

/// Pad or cut a line to exactly the given width, trimming the trailing
/// whitespace that padding would otherwise leave.
fn fit(line: &str, w: usize) -> String {
    let n = line.chars().count();
    if n > w {
        line.chars().take(w).collect()
    } else {
        line.trim_end().to_string()
    }
}

const MIN_W: usize = 40;
const MIN_H: usize = 8;

pub fn frame(
    r: &Report,
    rows: &[usize],
    s: &State,
    cs: Charset,
    w: usize,
    h: usize,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(h);
    if w < MIN_W || h < MIN_H {
        out.push(fit("terminal too small", w));
        out.push(fit(&format!("need {MIN_W}x{MIN_H}, have {w}x{h}"), w));
        while out.len() < h { out.push(String::new()); }
        out.truncate(h);
        return out;
    }

    // -- header: the trust anchor. What was measured, on what machine.
    out.push(fit(&format!(
        "  zc {}  {} {} {} {} {}",
        env!("CARGO_PKG_VERSION"),
        r.cpu.brand, cs.sep(),
        zc_report::backend_tag(r.backend), cs.sep(),
        human_bytes(r.mem.total),
    ), w));
    out.push(fit(&format!(
        "  {:.0} GB/s ram {} {:.1} GB/s disk {} {:.0} GFLOPS",
        r.ram_bw_gbs, cs.sep(), r.disk_gbs, cs.sep(), r.compute.gflops_f32,
    ), w));
    out.push(fit(&cs.rule().repeat(w.saturating_sub(2)), w));

    // -- body
    let mw = model_col_width(&r.models);
    out.push(fit(&format!(
        "    {:<mw$} {:<7} {:>12} {:>7} {:>6}  {:<6}",
        "MODEL", "QUANT", "DECODE tok/s", "MAX CTX", "TTFT", "CONF"
    ), w));

    let footer_lines = 2;
    let body_h = h.saturating_sub(out.len() + footer_lines);
    let detail = if s.detail { detail_lines(r, rows, s, cs, w) } else { Vec::new() };
    let list_h = body_h.saturating_sub(detail.len());

    for (screen_i, row_i) in rows.iter().skip(s.top).take(list_h).enumerate() {
        let row = &r.models[*row_i];
        let p = &row.prediction;
        let dot = if p.verdict == zc_model::Verdict::WontFit {
            cs.wont_fit()
        } else if p.resident_fraction < 0.999 {
            cs.partial()
        } else {
            cs.resident()
        };
        let selected = s.top + screen_i == s.cursor;
        let speed = if p.verdict == zc_model::Verdict::WontFit {
            "-".to_string()
        } else {
            format!("{:.1}-{:.1}", p.decode_tok_s.0, p.decode_tok_s.1)
        };
        let ctx = if p.max_context >= 1024 {
            format!("{}K", p.max_context / 1024)
        } else if p.max_context == 0 {
            "-".to_string()
        } else {
            p.max_context.to_string()
        };
        let ttft = match p.ttft_s {
            Some(t) if t < 100.0 => format!("{t:.1}s"),
            Some(t) => format!("{t:.0}s"),
            None => "-".to_string(),
        };
        out.push(fit(&format!(
            "{} {} {:<mw$} {:<7} {:>12} {:>7} {:>6}  {:<6}",
            if selected { ">" } else { " " },
            dot, row.model_id, row.quant.name, speed, ctx, ttft,
            p.confidence.label(),
        ), w));
    }
    out.extend(detail);

    // -- footer
    while out.len() < h.saturating_sub(footer_lines) { out.push(String::new()); }
    out.push(fit(&cs.rule().repeat(w.saturating_sub(2)), w));
    out.push(fit(&if s.filtering {
        format!("  /{}_", s.filter)
    } else if s.help {
        "  press ? or esc to close".to_string()
    } else {
        format!(
            "  {} of {} {} sort: {} {} enter why {} / filter {} ? help {} q quit",
            rows.len(), r.assumptions.total_rows, cs.sep(),
            s.sort.label(), cs.sep(), cs.sep(), cs.sep(), cs.sep(),
        )
    }, w));
    out.truncate(h);
    while out.len() < h { out.push(String::new()); }
    out
}

fn human_bytes(b: u64) -> String {
    format!("{:.0} GiB", b as f64 / (1u64 << 30) as f64)
}
```

`detail_lines` is Task 5 — for now stub it so this task compiles and its tests pass:

```rust
fn detail_lines(_r: &Report, _rows: &[usize], _s: &State, _cs: Charset, _w: usize) -> Vec<String> {
    Vec::new()
}
```

Add `pub mod frame;` to `crates/zc-tui/src/lib.rs`. If `zc_report::backend_tag` is private, make it `pub`.

- [ ] **Step 5: Run and watch it pass**

Run: `cargo test -p zc-tui`
Expected: PASS. Fix any `Default` derives the fixture needs, one at a time.

- [ ] **Step 6: Commit**

```bash
git add crates/zc-tui/src/frame.rs crates/zc-tui/src/lib.rs crates/zc-report/src/lib.rs
git commit -m "feat: the frame, exact to the column and the row"
```

---

### Task 5: The detail pane

The reason this TUI exists. Every field is already computed; none of it is new maths.

**Files:**
- Modify: `crates/zc-tui/src/frame.rs` (replace the `detail_lines` stub)

**Interfaces:**
- Consumes: `Report`, `State`, `Charset`
- Produces: `fn detail_lines(r: &Report, rows: &[usize], s: &State, cs: Charset, w: usize) -> Vec<String>`

- [ ] **Step 1: Write the failing tests**

Add to `frame.rs`'s `mod tests`:

```rust
    /// The pane is the product thesis made interactive: it must show where the
    /// number came from, not just the number.
    #[test]
    fn the_detail_pane_shows_the_derivation() {
        let f = fixture_frame_detail(100, 30);
        let joined = f.join("\n");
        for want in ["resident", "bandwidth", "eta", "confidence", "context"] {
            assert!(joined.contains(want), "detail pane missing {want:?}:\n{joined}");
        }
    }

    /// An unmeasured value is a dash, never a substituted constant. This is the
    /// standing rule for every number this project prints.
    #[test]
    fn an_unmeasured_field_is_a_dash_in_the_pane() {
        let f = fixture_frame_detail_unmeasured(100, 30);
        let ttft = f.iter().find(|l| l.contains("TTFT")).expect("no TTFT line");
        assert!(ttft.contains('-'), "unmeasured TTFT must print a dash: {ttft}");
    }

    /// The pane still obeys the frame contract.
    #[test]
    fn the_detail_pane_respects_width_and_height() {
        let f = fixture_frame_detail(80, 24);
        assert_eq!(f.len(), 24);
        for l in &f {
            assert!(l.chars().count() <= 80);
            assert_eq!(l.trim_end(), l);
        }
    }
```

Add the two fixture variants beside `fixture_frame_cs`: `fixture_frame_detail` sets `s.detail = true` before calling `frame`; `fixture_frame_detail_unmeasured` additionally sets every row's `prediction.ttft_s = None` and `prefill_tok_s = None`.

- [ ] **Step 2: Run and watch it fail**

Run: `cargo test -p zc-tui detail`
Expected: FAIL — the stub returns an empty vec, so no field is found

- [ ] **Step 3: Implement the pane**

Replace the stub in `crates/zc-tui/src/frame.rs`:

```rust
/// Where the number came from.
///
/// `llmfit` prints a score; this prints the derivation, which is the whole
/// reason to have measured the machine. Every value here is already computed
/// by `zc-model` — nothing is recalculated, and anything unmeasured prints a
/// dash exactly as it does in the table.
fn detail_lines(r: &Report, rows: &[usize], s: &State, cs: Charset, w: usize) -> Vec<String> {
    let Some(&i) = rows.get(s.cursor) else { return Vec::new() };
    let row = &r.models[i];
    let p = &row.prediction;
    let weights = row.quant.bytes as f64 / (1u64 << 30) as f64;
    let spill = weights * (1.0 - p.resident_fraction);
    let kv_mib = p.kv_bytes_per_token as f64 / (1u64 << 20) as f64;

    let mut o = vec![
        String::new(),
        fit(&cs.rule().repeat(w.saturating_sub(2)), w),
        fit(&format!("  {} {} {}", row.model_id, cs.sep(), row.quant.name), w),
        String::new(),
        fit(&format!(
            "  weights      {weights:>8.2} GiB   {:.0}% resident in RAM",
            p.resident_fraction * 100.0
        ), w),
    ];
    if spill > 0.005 {
        o.push(fit(&format!(
            "  spill        {spill:>8.2} GiB   at {:.1} GB/s disk", r.disk_gbs
        ), w));
    }
    o.push(fit(&format!(
        "  bandwidth    {:>8.0} GB/s  measured", r.ram_bw_gbs
    ), w));
    o.push(fit(&format!(
        "  eta          {:>8.3}       assumed, confidence {}",
        p.assumed_eta, p.confidence.label()
    ), w));
    o.push(fit(&format!(
        "  context      {:>8}       KV {} at {kv_mib:.2} MiB/token",
        if p.max_context >= 1024 {
            format!("{}K", p.max_context / 1024)
        } else {
            p.max_context.to_string()
        },
        r.assumptions.kv_precision,
    ), w));
    o.push(fit(&format!(
        "  TTFT         {:>8}       {}-token prompt",
        match p.ttft_s {
            Some(t) if t < 100.0 => format!("{t:.1}s"),
            Some(t) => format!("{t:.0}s"),
            None => "-".to_string(),
        },
        r.assumptions.prompt_tokens,
    ), w));
    o
}
```

- [ ] **Step 4: Run and watch it pass**

Run: `cargo test -p zc-tui`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/zc-tui/src/frame.rs
git commit -m "feat: press enter and the prediction shows its working"
```

---

### Task 6: The event loop, and the contract that must not break

**Files:**
- Create: `crates/zc-tui/src/run.rs`
- Modify: `crates/zc-tui/src/lib.rs`
- Modify: `crates/zc-cli/src/main.rs` (flags, `accepts`, mode selection)
- Modify: `crates/zc-cli/src/check.rs` (`run` gains a `tui: bool`)

**Interfaces:**
- Consumes: `state::{State, on_key, Key, Action}`, `frame::frame`
- Produces: `pub fn run(r: &Report, cs: Charset) -> std::io::Result<()>` in `zc-tui`

- [ ] **Step 1: Save the golden non-TTY output first**

Before touching anything, capture what must not change:

```bash
cargo build --release
./target/release/zc check --top 20 > /tmp/golden-check.txt
./target/release/zc check --json > /tmp/golden-check.json
```

- [ ] **Step 2: Implement the event loop**

Create `crates/zc-tui/src/run.rs`:

```rust
//! The only module that talks to a terminal.
//!
//! Everything decidable lives in `state` and `frame` and is unit-tested
//! headless. This file translates crossterm events into `Key` values, paints
//! the returned lines, and guarantees the terminal is restored on every exit
//! path -- including a panic, which is why the guard is a Drop type.

use crate::frame::frame;
use crate::state::{on_key, Action, Key, State};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::{cursor, execute, terminal};
use std::io::Write;
use zc_report::charset::Charset;
use zc_report::{best_per_model_indices, rank_index, Report};

/// Restores the terminal however we leave. A panic inside the loop would
/// otherwise strand the user in raw mode on the alternate screen, with no echo
/// and no prompt -- the single worst way for a tool like this to fail.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(std::io::stdout(), terminal::LeaveAlternateScreen, cursor::Show);
    }
}

fn to_key(e: KeyEvent) -> Option<Key> {
    if e.modifiers.contains(KeyModifiers::CONTROL) && e.code == KeyCode::Char('c') {
        return Some(Key::Quit);
    }
    Some(match e.code {
        KeyCode::Up => Key::Up,
        KeyCode::Down => Key::Down,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::Enter => Key::Enter,
        KeyCode::Esc => Key::Esc,
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Char('/') => Key::Slash,
        KeyCode::Char('j') => Key::Down,
        KeyCode::Char('k') => Key::Up,
        KeyCode::Char(c) => Key::Char(c),
        _ => return None,
    })
}

pub fn run(r: &Report, cs: Charset) -> std::io::Result<()> {
    let mut out = std::io::stdout();
    terminal::enable_raw_mode()?;
    execute!(out, terminal::EnterAlternateScreen, cursor::Hide)?;
    let _guard = Restore;

    let mut s = State::new(r.models.len());
    loop {
        let (w, h) = terminal::size().unwrap_or((80, 24));
        let rows = visible_rows(r, &s);
        s.set_len(rows.len());
        s.scroll_into_view((h as usize).saturating_sub(7));

        let lines = frame(r, &rows, &s, cs, w as usize, h as usize);
        execute!(out, cursor::MoveTo(0, 0), terminal::Clear(terminal::ClearType::All))?;
        for (i, l) in lines.iter().enumerate() {
            execute!(out, cursor::MoveTo(0, i as u16))?;
            write!(out, "{l}")?;
        }
        out.flush()?;

        match event::read()? {
            Event::Key(k) if k.is_press() => {
                if let Some(key) = to_key(k) {
                    if on_key(&mut s, key) == Action::Quit {
                        return Ok(());
                    }
                }
            }
            // A resize just redraws: the next loop reads the new size.
            Event::Resize(_, _) => {}
            _ => {}
        }
    }
}

/// Indices into `r.models`, after the `a` toggle, the filter and the sort.
fn visible_rows(r: &Report, s: &State) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..r.models.len()).collect();
    if !s.show_all {
        idx = best_per_model_indices(&r.models, &idx);
    }
    if !s.filter.is_empty() {
        let needle = s.filter.to_ascii_lowercase();
        idx.retain(|&i| r.models[i].model_id.to_ascii_lowercase().contains(&needle));
    }
    idx.sort_by_key(|&i| rank_index(&r.models[i], s.sort));
    idx
}
```

- [ ] **Step 3: Add the index-based ranking helpers to `zc-report`**

`Report` owns its rows, so the TUI sorts indices rather than moving rows. Add to `crates/zc-report/src/lib.rs`:

```rust
/// Which sort the caller wants. Mirrors the TUI's cycle; the static renderer
/// only ever uses `Verdict`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKey { Verdict, Decode, Context }

/// `rank`, but selectable and index-friendly. `Verdict` is byte-identical to
/// what `rank` returns, so the TUI's default order matches the static table.
pub fn rank_index(row: &Row, sort: SortKey) -> (u8, i64, i64) {
    let (v, d, c) = rank(row);
    match sort {
        SortKey::Verdict => (v, d, c),
        SortKey::Decode => (0, d, c),
        SortKey::Context => (0, c, d),
    }
}

/// `best_per_model`, over indices rather than owned rows.
pub fn best_per_model_indices(rows: &[Row], idx: &[usize]) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::new();
    for &i in idx {
        match best.iter().position(|&b| rows[b].model_id == rows[i].model_id) {
            None => best.push(i),
            Some(pos) => {
                let cur = best[pos];
                let better = rank(&rows[i]).0 < rank(&rows[cur]).0
                    || (rank(&rows[i]).0 == rank(&rows[cur]).0
                        && rows[i].quant.bytes > rows[cur].quant.bytes);
                if better { best[pos] = i; }
            }
        }
    }
    best
}
```

Change `crate::state::Sort` to re-export `zc_report::SortKey` instead of defining its own, so there is one sort enum in the workspace. Update `state.rs`'s `Sort::next`/`Sort::label` to be free functions or an extension trait over `SortKey`, and update Task 3's tests to match.

- [ ] **Step 4: Wire the CLI**

In `crates/zc-cli/src/main.rs`, add after `let print_only = take_flag(...)`:

```rust
    let force_tui = take_flag(&mut args, "--tui");
    let no_tui = take_flag(&mut args, "--no-tui");
```

Add both to the `supplied` array and to `accepts()` under `"check"` only:

```rust
        "check" => &["--json", "--kv", "--top", "--all", "--tui", "--no-tui"],
```

Add to the `supplied` array:

```rust
        ("--tui", force_tui),
        ("--no-tui", no_tui),
```

Then decide the mode and pass it to `check::run`:

```rust
    // TUI only when a human is on both ends of the pipe. Every other path --
    // a pipe, a redirect, --json, CI, an agent -- takes the static renderer
    // and its byte-identical output.
    use std::io::IsTerminal;
    let interactive = std::io::stdout().is_terminal()
        && std::io::stdin().is_terminal()
        && !as_json
        && !no_tui
        && !std::env::var("TERM").is_ok_and(|t| t == "dumb");
    if force_tui && !interactive {
        eprintln!("--tui needs an interactive terminal on stdin and stdout");
        std::process::exit(2);
    }
    let tui = cmd == "check" && (interactive || force_tui);
```

Note `cmd` defaults to `"check"` when args are empty, so bare `zc` opens the TUI. Pass `tui` into `check::run(&m, &fit, kv, top, show_all, as_json, tui)`.

In `crates/zc-cli/src/check.rs`, at the end of `run`, before the existing `print!`:

```rust
    if tui {
        let cs = zc_report::charset::detect();
        // A TUI failure must not lose the user their answer: fall through to
        // the static report rather than exiting on a terminal error.
        if zc_tui::run::run(&report, cs).is_ok() {
            // Alternate screen is gone; leave the answer in the scrollback.
            print!("{}", zc_report::text::render(&report));
            return 0;
        }
    }
```

When `tui` is true the row limit should not apply — the user scrolls instead. Skip the `models.truncate(n)` when `tui` is set, and keep `total_rows` as it was.

- [ ] **Step 5: Verify the golden output is byte-identical**

```bash
cargo build --release
./target/release/zc check --top 20 > /tmp/after-check.txt
./target/release/zc check --json > /tmp/after-check.json
diff /tmp/golden-check.txt /tmp/after-check.txt && echo "TEXT IDENTICAL"
diff /tmp/golden-check.json /tmp/after-check.json && echo "JSON IDENTICAL"
```

Expected: both print IDENTICAL. If not, the mode selection is leaking into a non-TTY path — fix before continuing. Note these run piped, so `is_terminal()` is false and the static path is taken, which is exactly what is being proven.

- [ ] **Step 6: Verify the error and flag paths**

```bash
./target/release/zc check --tui < /dev/null > /dev/null; echo "expect 2, got $?"
./target/release/zc doctor --tui; echo "expect 2, got $?"
./target/release/zc check --json --tui < /dev/null > /dev/null; echo "expect 2, got $?"
```

Expected: `2` three times.

- [ ] **Step 7: Verify interactively by hand**

Run `./target/release/zc` in a real terminal. Check: arrows and `j`/`k` move; `enter` opens and closes the detail pane; `/` filters and `s` inside the filter types an `s` rather than changing sort; `esc` clears the filter; `s` cycles sort; `a` toggles quantisations; `?` opens help; `q` quits and leaves the report in scrollback; resizing the window reflows without corruption; `ZC_ASCII=1 ./target/release/zc` renders with `*`/`o`/`-`.

- [ ] **Step 8: Full suite, clippy, cross-target**

Run: `./check.sh`
Expected: OK. This also compiles the Windows and Linux targets, which is where a crossterm cfg mistake would surface.

- [ ] **Step 9: Commit**

```bash
git add crates/zc-tui crates/zc-cli crates/zc-report
git commit -m "feat: zc opens a browsable table when a human is watching"
```

---

### Task 7: Live progress during the benchmark

**Files:**
- Create: `crates/zc-report/src/progress.rs`
- Modify: `crates/zc-report/src/lib.rs`
- Modify: `crates/zc-cli/src/machine.rs` (call the reporter around each measurement)

**Interfaces:**
- Produces:
  - `pub struct Progress` with `pub fn new(cs: Charset) -> Progress`
  - `pub fn start(&mut self, label: &str)`
  - `pub fn tick(&mut self)`
  - `pub fn done(&mut self, result: &str)`

- [ ] **Step 1: Write the failing test**

Create `crates/zc-report/src/progress.rs` test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::charset::Charset;

    /// Progress is decoration. On a non-TTY it must emit absolutely nothing,
    /// or `zc check 2>&1 | grep` starts seeing spinner frames.
    #[test]
    fn a_non_tty_gets_no_bytes_at_all() {
        let mut buf = Vec::new();
        let mut p = Progress::to_writer(&mut buf, Charset::Ascii, false);
        p.start("ram");
        p.tick();
        p.done("125 GB/s");
        assert!(buf.is_empty(), "wrote {:?}", String::from_utf8_lossy(&buf));
    }

    /// On a TTY the line is rewritten in place, never appended, so three
    /// measurements leave three lines and not thirty.
    #[test]
    fn a_tty_rewrites_one_line_per_measurement() {
        let mut buf = Vec::new();
        let mut p = Progress::to_writer(&mut buf, Charset::Ascii, true);
        p.start("ram");
        p.tick();
        p.tick();
        p.done("125 GB/s");
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(s.matches('\n').count(), 1, "one newline per finished line: {s:?}");
        assert!(s.contains("125 GB/s"));
        assert!(s.contains('\r'), "must rewrite in place");
    }
}
```

- [ ] **Step 2: Run and watch it fail**

Run: `cargo test -p zc-report progress`
Expected: FAIL — `Progress` is not defined

- [ ] **Step 3: Implement**

Prepend to `crates/zc-report/src/progress.rs`:

```rust
//! What the user watches while the benchmark runs.
//!
//! `zc check` used to print its first byte and its last byte at the same
//! moment -- 1.88s on a fast machine, far longer on the DRAM-less drives this
//! product targets, and a dead terminal throughout. clig.dev puts the
//! threshold at 100ms.
//!
//! Writes to stderr so `zc check > file` still shows progress while `file`
//! stays clean, and emits nothing at all when stderr is not a terminal.

use crate::charset::Charset;
use std::io::Write;

pub struct Progress<W: Write> {
    w: W,
    cs: Charset,
    live: bool,
    tick: usize,
    label: String,
}

impl Progress<std::io::Stderr> {
    pub fn new(cs: Charset) -> Progress<std::io::Stderr> {
        let live = std::io::IsTerminal::is_terminal(&std::io::stderr());
        Progress::to_writer(std::io::stderr(), cs, live)
    }
}

impl<W: Write> Progress<W> {
    pub fn to_writer(w: W, cs: Charset, live: bool) -> Progress<W> {
        Progress { w, cs, live, tick: 0, label: String::new() }
    }

    pub fn start(&mut self, label: &str) {
        if !self.live { return; }
        self.label = label.to_string();
        self.tick = 0;
        self.paint();
    }

    pub fn tick(&mut self) {
        if !self.live { return; }
        self.tick = self.tick.wrapping_add(1);
        self.paint();
    }

    /// Rewrite the line with the measured value and end it.
    pub fn done(&mut self, result: &str) {
        if !self.live { return; }
        // \r then clear-to-end-of-line: never cursor-up, so a resize
        // mid-benchmark cannot corrupt a line already finished.
        let _ = write!(self.w, "\r\x1b[2K  {:<10} {result}\n", self.label);
        let _ = self.w.flush();
    }

    fn paint(&mut self) {
        let spin = self.cs.spinner(self.tick);
        let _ = write!(self.w, "\r\x1b[2K  {:<10} {spin}", self.label);
        let _ = self.w.flush();
    }
}
```

Add `pub mod progress;` to `crates/zc-report/src/lib.rs`.

- [ ] **Step 4: Run and watch it pass**

Run: `cargo test -p zc-report progress`
Expected: PASS (2 tests)

- [ ] **Step 5: Wire it into the probe**

In `crates/zc-cli/src/machine.rs`, wrap the three measurements. Read the file first — the RAM, compute and disk calls are sequential around line 95-105. Around each:

```rust
let mut pr = zc_report::progress::Progress::new(zc_report::charset::detect());

pr.start("ram");
let ram = /* existing call */;
pr.done(&format!("{:.0} GB/s peak", ram.peak_gbs));

pr.start("compute");
let compute = /* existing call */;
pr.done(&format!("{:.0} GFLOPS f32", compute.gflops_f32));

pr.start("disk");
let disk = /* existing call */;
match &disk {
    Some(d) => pr.done(&format!("{:.2} GB/s random 128K", d.random_gbs)),
    None => pr.done("not measurable"),
}
```

Field names must match the real structs — check `zc-bench/src/ram.rs`, `compute.rs` and `disk.rs` and use whatever they actually expose. The benchmarks are synchronous, so there is no thread to tick from; `start` paints one frame and `done` replaces it. That already satisfies the 100ms rule, and a spinner thread is not worth a `std::thread` and a channel here.

- [ ] **Step 6: Verify by hand and confirm the golden output still holds**

```bash
cargo build --release
./target/release/zc check --top 3            # progress visible
./target/release/zc check --top 3 2>/dev/null | head -3   # still clean
./target/release/zc check --top 20 > /tmp/after2.txt 2>/dev/null
diff /tmp/golden-check.txt /tmp/after2.txt && echo "STDOUT UNCHANGED"
./target/release/zc check --top 3 2>&1 | cat   # piped stderr: no spinner bytes
```

Expected: progress on a terminal, nothing on a pipe, stdout unchanged.

- [ ] **Step 7: Commit**

```bash
git add crates/zc-report/src/progress.rs crates/zc-report/src/lib.rs crates/zc-cli/src/machine.rs
git commit -m "feat: the benchmark says what it is doing while it does it"
```

---

### Task 8: Help that leads with examples, and a typo that suggests

`HELP` is ~90 lines of PRECONDITIONS / SIDE EFFECTS / AGENT USAGE. That is excellent for an agent and poor for a human's first five seconds. clig.dev: lead with examples, show the most common commands first. And `zc chekc` currently says "unknown command" without suggesting the obvious.

**Files:**
- Modify: `crates/zc-cli/src/main.rs` (`HELP`, add `did_you_mean`)
- Modify: `README.md` (document the TUI keys, `--tui`/`--no-tui`, `ZC_ASCII`)
- Modify: `check.sh` (add a non-TTY byte-identity guard)

- [ ] **Step 1: Write the failing test for the suggestion**

Add to `main.rs`'s `mod tests`:

```rust
    /// clig.dev: if the user did something wrong and you can guess what they
    /// meant, suggest it.
    #[test]
    fn a_near_miss_command_is_suggested() {
        assert_eq!(super::did_you_mean("chekc"), Some("check"));
        assert_eq!(super::did_you_mean("verift"), Some("verify"));
        assert_eq!(super::did_you_mean("doctr"), Some("doctor"));
        assert_eq!(super::did_you_mean("gat"), Some("gate"));
        // Nothing close enough is worse than no suggestion at all.
        assert_eq!(super::did_you_mean("xyzzy"), None);
    }
```

- [ ] **Step 2: Run and watch it fail**

Run: `cargo test -p zc-cli did_you_mean`
Expected: FAIL — not defined

- [ ] **Step 3: Implement**

Add to `crates/zc-cli/src/main.rs`:

```rust
/// The closest command name, if one is close enough to be worth suggesting.
///
/// Plain Levenshtein over a six-item list -- a fuzzy-match dependency for this
/// would be absurd. The distance cap matters: suggesting `share` for `xyzzy`
/// is worse than saying nothing.
fn did_you_mean(input: &str) -> Option<&'static str> {
    const CMDS: &[&str] = &["check", "verify", "fit", "gate", "share", "doctor"];
    CMDS.iter()
        .map(|c| (*c, distance(input, c)))
        .filter(|(c, d)| *d <= 2 && *d < c.len())
        .min_by_key(|(_, d)| *d)
        .map(|(c, _)| c)
}

fn distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let sub = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + sub);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}
```

Use it at the unknown-command site:

```rust
    if !matches!(cmd, "check" | "verify" | "doctor") {
        match did_you_mean(cmd) {
            Some(c) => eprintln!("unknown command '{cmd}' -- did you mean `zc {c}`?"),
            None => eprintln!("unknown command '{cmd}' -- run `zc --help`"),
        }
        std::process::exit(2);
    }
```

- [ ] **Step 4: Run and watch it pass**

Run: `cargo test -p zc-cli`
Expected: PASS

- [ ] **Step 5: Put examples at the top of `HELP`**

Insert directly under the one-line description in `HELP`, above `USAGE`:

```
EXAMPLES
    zc                    browse what this machine can run  (arrows, ? for keys)
    zc check --top 5      the five best fits, as plain text
    zc check --json       the same data for a script or an agent
    zc verify qwen3:1.7b  run the model for real and compare
    zc doctor             a paste-ready report for a bug

```

Then add the two new flags to the `zc check` USAGE line:

```
    zc [check] [--json] [--kv f16|q8|q4] [--top N | --all] [--tui | --no-tui]
```

And add to the `zc check` section:

```
    INTERACTIVE    On a terminal, `zc check` opens a browsable table:
                   arrows/jk move, enter shows how a number was derived,
                   / filters, s sorts, a toggles quantisations, ? lists keys,
                   q quits and leaves the report in your scrollback.
                   Piped, redirected, or with --json it prints plain text
                   instead -- byte for byte what it always has.
                   --tui forces it on, --no-tui forces it off, ZC_ASCII=1
                   swaps the box-drawing glyphs for ASCII.
```

- [ ] **Step 6: Correct the `--top | --all` help**

The `|` implies the two are exclusive; they compose — `--all --top 3` lists three rows from every quantisation. Change the AGENT USAGE wording to say so:

```
                   `--all` lists every quantisation; `--top N` (default 20)
                   sets the row limit and applies on top of it.
```

- [ ] **Step 7: Guard the non-TTY contract in `check.sh`**

Add before `echo "OK"`:

```bash
echo "== non-tty output =="
# The TUI is default-on for a human. Everything else -- a pipe, a redirect,
# --json, CI, an agent -- must still get the plain renderer. This runs piped,
# so if a TUI escape sequence ever leaks into stdout it fails here.
if cargo run --release --quiet -- check --top 3 2>/dev/null | grep -q $'\x1b'; then
  echo "escape sequences leaked into piped stdout"; exit 1
fi
cargo run --release --quiet -- check --json 2>/dev/null | python3 -m json.tool >/dev/null \
  || { echo "--json is not valid JSON"; exit 1; }
echo "plain when piped"
```

- [ ] **Step 8: Update the README**

Add a short section after the existing usage block documenting the TUI keys, `--tui`/`--no-tui`, and `ZC_ASCII=1`. Keep it to the table of keys plus two sentences — the README's job is the first 30 seconds.

- [ ] **Step 9: Measure the binary size, before and after**

The project rule applies to its own artifact — record the real number, do not estimate it:

```bash
git stash && cargo build --release && ls -l target/release/zc | awk '{print "before:", $5}'
git stash pop && cargo build --release && ls -l target/release/zc | awk '{print "after:", $5}'
```

Put both numbers in the commit message. If the growth is over 2 MB, say so in the commit body rather than hiding it.

- [ ] **Step 10: Full check and commit**

```bash
./check.sh
git add crates/zc-cli/src/main.rs README.md check.sh
git commit -m "docs: help that starts with what to type"
```

---

## Self-review

**Spec coverage.** Mode selection → Task 6. Progress → Task 7. Charset → Task 2. TUI screen and keys → Tasks 3–5. Shared column widths → Task 1. Testing requirements → distributed, with the byte-identity guard in Task 6 Step 5 and Task 8 Step 7. Out-of-scope items are not implemented anywhere. Risks: the alternate-screen risk is handled by Task 6's `Restore` guard and the on-quit static print; binary size by Task 8 Step 9.

**Known plan-time uncertainties**, called out rather than papered over:

1. Task 4's fixture needs `Default` on several probe/bench structs. The step says to add derives one at a time as the compiler asks. If any struct cannot sensibly default, build it field-by-field instead.
2. Task 7 Step 5 names fields (`ram.peak_gbs`, `compute.gflops_f32`, `disk.random_gbs`) that must be checked against the real structs before use.
3. Task 6 Step 3 collapses `state::Sort` into `zc_report::SortKey`. Task 3's tests reference `Sort` and must be updated in the same commit.
4. `crossterm` 0.29's `KeyEvent::is_press()` exists to filter key-repeat and release events on Windows. If the version resolved differs, match on `k.kind == KeyEventKind::Press` instead.
