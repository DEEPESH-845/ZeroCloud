//! The interactive surface of `zc check`.
//!
//! Split so that everything except the event loop is testable without a
//! terminal: `state` owns what is shown and what a key does to it, `frame`
//! turns that into lines, and `run` is the only module that talks to
//! crossterm.
//!
//! The TUI is default-on when a human is at both ends of the pipe, and never
//! otherwise. Every non-TTY path -- a pipe, a redirect, `--json`, CI, an
//! agent -- still gets `zc_report::text`, byte for byte what it always got.

pub mod frame;
pub mod run;
pub mod state;
