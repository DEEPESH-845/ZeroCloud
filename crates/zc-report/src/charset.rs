//! Which glyphs this terminal can render.
//!
//! Resolved once from the environment and threaded through every renderer, so
//! no literal box-drawing character appears anywhere else in the codebase.
//!
//! Detection reads environment variables and makes no system calls. Probing
//! `GetConsoleOutputCP` would mean an `extern "system"` block in a crate that
//! has none, on a platform `VERIFICATION.md` records as never executed here.
//! Guessing Ascii on an unknown console renders everywhere; guessing Unicode
//! wrong prints replacement boxes on the old laptops this tool targets.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Charset {
    Unicode,
    Ascii,
}

/// Pure so every branch is reachable from a test without touching the process
/// environment, which is global and makes concurrent tests flaky.
fn from_env(term: Option<&str>, locale: Option<&str>, force_ascii: bool, windows: bool) -> Charset {
    if force_ascii {
        return Charset::Ascii;
    }
    match term {
        None | Some("dumb") | Some("") => return Charset::Ascii,
        Some(_) => {}
    }
    // On Windows, TERM being set at all means Git Bash, WSL or MSYS, each of
    // which is UTF-8. Windows Terminal sets WT_SESSION instead and is also
    // UTF-8. Legacy conhost sets neither, so it never reaches here and gets
    // Ascii -- which is the right answer for it.
    if windows {
        return Charset::Unicode;
    }
    match locale {
        Some(l) if l.to_ascii_lowercase().replace('-', "").contains("utf8") => Charset::Unicode,
        _ => Charset::Ascii,
    }
}

pub fn detect() -> Charset {
    let term = std::env::var("TERM")
        .ok()
        .filter(|t| !t.is_empty())
        // Windows Terminal sets no TERM but renders UTF-8 fine.
        .or_else(|| std::env::var("WT_SESSION").ok().map(|_| "wt".to_string()));
    let locale = ["LC_ALL", "LC_CTYPE", "LANG"]
        .iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()));
    let force = std::env::var("ZC_ASCII").is_ok_and(|v| v == "1");
    from_env(term.as_deref(), locale.as_deref(), force, cfg!(windows))
}

impl Charset {
    /// Every model byte is in RAM.
    pub fn resident(&self) -> &'static str {
        match self {
            Charset::Unicode => "●",
            Charset::Ascii => "*",
        }
    }

    /// Some of the weights stream from disk on every token.
    pub fn partial(&self) -> &'static str {
        match self {
            Charset::Unicode => "◐",
            Charset::Ascii => "o",
        }
    }

    pub fn wont_fit(&self) -> &'static str {
        match self {
            Charset::Unicode => "○",
            Charset::Ascii => ".",
        }
    }

    pub fn rule(&self) -> &'static str {
        match self {
            Charset::Unicode => "─",
            Charset::Ascii => "-",
        }
    }

    pub fn vrule(&self) -> &'static str {
        match self {
            Charset::Unicode => "│",
            Charset::Ascii => "|",
        }
    }

    pub fn sep(&self) -> &'static str {
        match self {
            Charset::Unicode => "·",
            Charset::Ascii => "-",
        }
    }

    /// Frame `tick` of the spinner. Wraps, so a caller may increment forever.
    pub fn spinner(&self, tick: usize) -> &'static str {
        const BRAILLE: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        const ASCII: [&str; 4] = ["-", "\\", "|", "/"];
        match self {
            Charset::Unicode => BRAILLE[tick % BRAILLE.len()],
            Charset::Ascii => ASCII[tick % ASCII.len()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_locales_get_unicode() {
        assert_eq!(
            from_env(Some("xterm"), Some("en_US.UTF-8"), false, false),
            Charset::Unicode
        );
        assert_eq!(
            from_env(Some("xterm"), Some("C.utf8"), false, false),
            Charset::Unicode
        );
    }

    #[test]
    fn a_non_utf8_or_dumb_terminal_gets_ascii() {
        assert_eq!(
            from_env(Some("xterm"), Some("C"), false, false),
            Charset::Ascii
        );
        assert_eq!(
            from_env(Some("dumb"), Some("en_US.UTF-8"), false, false),
            Charset::Ascii
        );
        assert_eq!(
            from_env(None, Some("en_US.UTF-8"), false, false),
            Charset::Ascii
        );
    }

    /// Legacy conhost sets neither TERM nor WT_SESSION, so it lands on the
    /// None arm and gets Ascii. Anything that does set one is a modern shell.
    #[test]
    fn windows_without_a_terminal_hint_gets_ascii() {
        assert_eq!(from_env(None, None, false, true), Charset::Ascii);
        assert_eq!(from_env(Some("wt"), None, false, true), Charset::Unicode);
    }

    /// The escape hatch, and what a user on a terminal we guessed wrong sets.
    #[test]
    fn zc_ascii_overrides_everything() {
        assert_eq!(
            from_env(Some("xterm"), Some("en_US.UTF-8"), true, false),
            Charset::Ascii
        );
    }

    /// Both charsets must occupy the same number of columns, or a table laid
    /// out for one wraps in the other. This is why there are no emoji here:
    /// they are double-width and would misalign every row.
    #[test]
    fn every_glyph_is_one_column_wide_in_both() {
        for c in [Charset::Unicode, Charset::Ascii] {
            for g in [
                c.resident(),
                c.partial(),
                c.wont_fit(),
                c.rule(),
                c.vrule(),
                c.sep(),
            ] {
                assert_eq!(g.chars().count(), 1, "{g:?} in {c:?} is not one column");
            }
            for tick in 0..12 {
                assert_eq!(c.spinner(tick).chars().count(), 1);
            }
        }
    }
}
