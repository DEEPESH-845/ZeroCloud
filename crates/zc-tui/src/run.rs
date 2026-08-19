//! The only module that talks to a terminal.
//!
//! Everything decidable lives in `state` and `frame` and is unit-tested
//! headless. This file translates crossterm events into `Key` values, paints
//! the lines it gets back, and guarantees the terminal is restored on every
//! exit path — including a panic, which is why the guard is a `Drop` type.

use crate::frame::{body_height, frame, View};
use crate::state::{on_key, Action, Key, State};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::{cursor, execute, queue, terminal};
use std::io::Write;
use zc_report::charset::Charset;
use zc_report::{best_per_model_indices, rank_by, Report};

/// Restores the terminal however we leave.
///
/// A panic inside the loop would otherwise strand the user in raw mode on the
/// alternate screen, with no echo and no prompt — the single worst way for a
/// tool like this to fail, and the reason a default-on TUI needs this to be
/// unconditional rather than a tidy-up at the end of `run`.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(
            std::io::stdout(),
            terminal::LeaveAlternateScreen,
            cursor::Show
        );
    }
}

fn to_key(e: KeyEvent) -> Option<Key> {
    // Windows reports key releases as well as presses; acting on both would
    // move the cursor two rows per keystroke.
    if e.kind == KeyEventKind::Release {
        return None;
    }
    if e.modifiers.contains(KeyModifiers::CONTROL) && matches!(e.code, KeyCode::Char('c')) {
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
        KeyCode::Char(c) => Key::Char(c),
        _ => return None,
    })
}

/// Vim keys, but only outside the filter — inside it they are text.
fn vim_key(k: Key, filtering: bool) -> Key {
    if filtering {
        return k;
    }
    match k {
        Key::Char('j') => Key::Down,
        Key::Char('k') => Key::Up,
        Key::Char('g') => Key::Home,
        Key::Char('G') => Key::End,
        other => other,
    }
}

/// Indices into `r.models`, after the `a` toggle, the filter and the sort.
///
/// Recomputed on every keystroke rather than cached: the row set is at most a
/// few hundred entries, and a cache invalidated on four different state
/// changes is a bug waiting for the fifth.
/// Returns the rows to show, and how many the current view holds *before* the
/// filter — the footer's denominator. Using the report's own total there was
/// wrong in one view or the other, since `a` changes what "all of them" means.
pub fn visible_rows(r: &Report, s: &State) -> (Vec<usize>, usize) {
    let mut idx: Vec<usize> = (0..r.models.len()).collect();
    if !s.show_all {
        idx = best_per_model_indices(&r.models, &idx);
    }
    let unfiltered = idx.len();
    if !s.filter.is_empty() {
        let needle = s.filter.to_ascii_lowercase();
        idx.retain(|&i| r.models[i].model_id.to_ascii_lowercase().contains(&needle));
    }
    idx.sort_by_key(|&i| rank_by(&r.models[i], s.sort));
    (idx, unfiltered)
}

pub fn run(r: &Report, cs: Charset) -> std::io::Result<()> {
    let mut out = std::io::stdout();
    terminal::enable_raw_mode()?;
    execute!(out, terminal::EnterAlternateScreen, cursor::Hide)?;
    let _guard = Restore;

    let view = View::from_report(r);
    let mut s = State::new(0);

    loop {
        let (w, h) = terminal::size().unwrap_or((80, 24));
        let (w, h) = (w as usize, h as usize);
        let (rows, total) = visible_rows(r, &s);
        s.set_len(rows.len());
        s.scroll_into_view(body_height(h));

        let lines = frame(&view, &rows, total, &s, cs, w, h);
        queue!(out, terminal::Clear(terminal::ClearType::All))?;
        for (i, l) in lines.iter().enumerate() {
            queue!(out, cursor::MoveTo(0, i as u16))?;
            write!(out, "{l}")?;
        }
        out.flush()?;

        match event::read()? {
            Event::Key(k) => {
                if let Some(key) = to_key(k) {
                    // Resolve against the filter state *before* the borrow, so
                    // 'j' is a cursor move when navigating and a letter when
                    // typing a search.
                    let key = vim_key(key, s.filtering);
                    if on_key(&mut s, key) == Action::Quit {
                        return Ok(());
                    }
                }
            }
            // A resize just redraws: the next pass reads the new size.
            Event::Resize(_, _) => {}
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inside the filter every printable key is text, so a user typing "gj"
    /// while searching must not jump the cursor.
    #[test]
    fn vim_keys_are_text_inside_the_filter() {
        assert_eq!(vim_key(Key::Char('j'), false), Key::Down);
        assert_eq!(vim_key(Key::Char('k'), false), Key::Up);
        assert_eq!(vim_key(Key::Char('G'), false), Key::End);
        assert_eq!(vim_key(Key::Char('j'), true), Key::Char('j'));
        assert_eq!(vim_key(Key::Char('G'), true), Key::Char('G'));
    }

    /// Windows reports releases as well as presses; acting on both would move
    /// the cursor two rows per keystroke.
    #[test]
    fn a_key_release_is_ignored() {
        let mut e = KeyEvent::new(KeyCode::Down, KeyModifiers::NONE);
        e.kind = KeyEventKind::Release;
        assert_eq!(to_key(e), None);
        let mut e = KeyEvent::new(KeyCode::Down, KeyModifiers::NONE);
        e.kind = KeyEventKind::Press;
        assert_eq!(to_key(e), Some(Key::Down));
    }

    /// Ctrl-C must reach the state machine as a quit from any mode. A user who
    /// cannot find the exit is the worst failure a default-on TUI has.
    #[test]
    fn ctrl_c_maps_to_quit() {
        let e = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(to_key(e), Some(Key::Quit));
        // Plain 'c' is not a quit.
        let e = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE);
        assert_eq!(to_key(e), Some(Key::Char('c')));
    }
}
