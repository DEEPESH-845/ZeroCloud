//! What the user watches while the benchmark runs.
//!
//! `zc check` used to print its first byte and its last byte at the same
//! moment — 1.88s on a fast machine, far longer on the DRAM-less drives this
//! product targets, and a dead terminal throughout. clig.dev puts the
//! threshold at 100ms, and the machines that miss it worst are the ones the
//! tool exists to serve.
//!
//! Writes to **stderr** so `zc check > file` still shows progress while `file`
//! stays clean, and emits nothing at all when stderr is not a terminal — a
//! spinner frame landing in a CI log or a `2>&1` pipeline is noise nobody
//! asked for.

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
        Progress {
            w,
            cs,
            live,
            tick: 0,
            label: String::new(),
        }
    }

    /// Announce a measurement before starting it, so the terminal is never
    /// silent while work is happening.
    pub fn start(&mut self, label: &str) {
        if !self.live {
            return;
        }
        self.label = label.to_string();
        self.tick = 0;
        self.paint();
    }

    pub fn tick(&mut self) {
        if !self.live {
            return;
        }
        self.tick = self.tick.wrapping_add(1);
        self.paint();
    }

    /// Rewrite the line with the measured value and end it.
    pub fn done(&mut self, result: &str) {
        if !self.live {
            return;
        }
        // `\r` then clear-to-end-of-line, never cursor-up: a resize partway
        // through the benchmark cannot corrupt a line that already finished.
        let _ = write!(self.w, "\r\x1b[2K  {:<9} {result}\n", self.label);
        let _ = self.w.flush();
    }

    fn paint(&mut self) {
        let spin = self.cs.spinner(self.tick);
        let _ = write!(self.w, "\r\x1b[2K  {:<9} {spin}", self.label);
        let _ = self.w.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charset::Charset;

    /// Progress is decoration. On a non-TTY it must emit nothing at all, or
    /// `zc check 2>&1 | grep` starts matching spinner frames.
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
    /// measurements leave three lines rather than thirty.
    #[test]
    fn a_tty_rewrites_one_line_per_measurement() {
        let mut buf = Vec::new();
        let mut p = Progress::to_writer(&mut buf, Charset::Ascii, true);
        p.start("ram");
        p.tick();
        p.tick();
        p.done("125 GB/s");
        let s = String::from_utf8(buf).unwrap();
        assert_eq!(
            s.matches('\n').count(),
            1,
            "one newline per finished line: {s:?}"
        );
        assert!(s.contains("125 GB/s"));
        assert!(s.contains('\r'), "must rewrite in place");
    }

    /// The label stays put while the spinner turns, so the three measurements
    /// line up in a column instead of jittering.
    #[test]
    fn the_result_column_is_aligned_across_labels() {
        let render = |label: &str| {
            let mut buf = Vec::new();
            let mut p = Progress::to_writer(&mut buf, Charset::Ascii, true);
            p.start(label);
            p.done("x");
            String::from_utf8(buf).unwrap()
        };
        let a = render("ram");
        let b = render("compute");
        let col = |s: &str| s.rfind('x').unwrap() - s.rfind('\r').unwrap();
        assert_eq!(col(&a), col(&b), "results must share a column:\n{a}\n{b}");
    }
}
