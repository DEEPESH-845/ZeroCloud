//! What the TUI is showing, and what a keystroke does to it.
//!
//! Deliberately free of crossterm and of any terminal at all: every branch
//! here is reachable from a unit test. Only `run` translates real events into
//! these `Key` values and paints the result.

pub use zc_report::SortKey as Sort;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
    Enter,
    Slash,
    Esc,
    Backspace,
    /// Ctrl-C, which quits from anywhere including inside the filter.
    Quit,
    Char(char),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Redraw,
    Quit,
}

pub struct State {
    pub cursor: usize,
    /// Index of the first row on screen.
    pub top: usize,
    pub len: usize,
    pub filter: String,
    pub filtering: bool,
    pub sort: Sort,
    pub show_all: bool,
    pub detail: bool,
    pub help: bool,
    /// Body height in rows, set by the last frame. Page moves need it, and it
    /// changes on every resize.
    pub page: usize,
}

impl State {
    pub fn new(len: usize) -> State {
        State {
            cursor: 0,
            top: 0,
            len,
            filter: String::new(),
            filtering: false,
            sort: Sort::Verdict,
            show_all: false,
            detail: false,
            help: false,
            page: 10,
        }
    }

    /// Tell the state the row set changed size — after a filter keystroke or
    /// an `a` toggle. Pulls the cursor back into range rather than leaving an
    /// index that a later lookup would have to guard against.
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
        let last = (self.len - 1) as isize;
        self.cursor = (self.cursor as isize + delta).clamp(0, last) as usize;
    }

    /// Scroll so the cursor is visible in a body `height` rows tall.
    ///
    /// Called before every frame, so a resize that shrinks the window under
    /// the cursor pulls the view back rather than painting a cursor nobody
    /// can see.
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
    // Ctrl-C means quit from anywhere, including mid-filter. A user who cannot
    // find the exit is the worst outcome a default-on TUI can produce.
    if k == Key::Quit {
        return Action::Quit;
    }
    // The help overlay swallows everything except the keys that dismiss it.
    if s.help {
        s.help = !matches!(k, Key::Esc | Key::Enter | Key::Char('?') | Key::Char('q'));
        return Action::Redraw;
    }
    // Inside the filter, printable keys are text. Otherwise a user searching
    // for "smollm2" would toggle the sort on the 's'.
    if s.filtering {
        match k {
            Key::Char(c) => s.filter.push(c),
            Key::Backspace => {
                s.filter.pop();
            }
            Key::Esc => {
                s.filtering = false;
                s.filter.clear();
            }
            // Enter commits the filter and returns to navigating, leaving the
            // narrowed set in place.
            Key::Enter => s.filtering = false,
            Key::Up => s.move_by(-1),
            Key::Down => s.move_by(1),
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
        // Esc dismisses whatever is showing before it quits. A user who
        // committed a filter with enter and then pressed esc to clear it used
        // to have the program exit under them.
        Key::Esc if !s.filter.is_empty() => s.filter.clear(),
        Key::Char('q') | Key::Esc => return Action::Quit,
        _ => {}
    }
    Action::Redraw
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(n: usize) -> State {
        State::new(n)
    }

    #[test]
    fn the_cursor_cannot_leave_the_row_set() {
        let mut s = st(3);
        on_key(&mut s, Key::Up);
        assert_eq!(s.cursor, 0, "up at the top must stay at the top");
        for _ in 0..10 {
            on_key(&mut s, Key::Down);
        }
        assert_eq!(s.cursor, 2, "down at the bottom must stay at the bottom");
    }

    /// A filter that matches nothing is reachable by typing, so every
    /// navigation key must survive an empty row set.
    #[test]
    fn an_empty_row_set_is_navigable_without_panicking() {
        let mut s = st(0);
        on_key(&mut s, Key::Down);
        on_key(&mut s, Key::End);
        on_key(&mut s, Key::PageDown);
        on_key(&mut s, Key::Home);
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
    fn page_moves_use_the_height_of_the_last_frame() {
        let mut s = st(100);
        s.scroll_into_view(20);
        on_key(&mut s, Key::PageDown);
        assert_eq!(s.cursor, 20);
        on_key(&mut s, Key::PageUp);
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
    /// searching for "smollm2" would toggle sort on the 's' and quit on the
    /// second character of "qwen".
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

    /// Committing a filter with enter leaves it active while no longer
    /// editing. Esc then has something to dismiss, and dismissing it must not
    /// mean exiting.
    #[test]
    fn esc_clears_a_committed_filter_before_it_quits() {
        let mut s = st(5);
        on_key(&mut s, Key::Slash);
        on_key(&mut s, Key::Char('q'));
        on_key(&mut s, Key::Char('w'));
        assert_eq!(on_key(&mut s, Key::Enter), Action::Redraw);
        assert!(!s.filtering, "enter commits the filter");
        assert_eq!(s.filter, "qw", "and leaves it active");
        assert_eq!(on_key(&mut s, Key::Esc), Action::Redraw, "esc clears it");
        assert_eq!(s.filter, "");
        assert_eq!(on_key(&mut s, Key::Esc), Action::Quit, "then esc quits");
    }

    #[test]
    fn q_quits_but_only_outside_the_filter() {
        let mut s = st(5);
        on_key(&mut s, Key::Slash);
        assert_eq!(on_key(&mut s, Key::Char('q')), Action::Redraw);
        on_key(&mut s, Key::Esc);
        assert_eq!(on_key(&mut s, Key::Char('q')), Action::Quit);
    }

    /// A user who cannot find the exit is the worst failure a default-on TUI
    /// has, so ctrl-c works from every mode.
    #[test]
    fn ctrl_c_quits_from_anywhere() {
        let mut s = st(5);
        on_key(&mut s, Key::Slash);
        assert_eq!(on_key(&mut s, Key::Quit), Action::Quit);
        let mut s = st(5);
        on_key(&mut s, Key::Char('?'));
        assert_eq!(on_key(&mut s, Key::Quit), Action::Quit);
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

    #[test]
    fn the_help_overlay_swallows_navigation_until_dismissed() {
        let mut s = st(5);
        on_key(&mut s, Key::Char('?'));
        assert!(s.help);
        on_key(&mut s, Key::Down);
        assert_eq!(s.cursor, 0, "help must swallow navigation");
        on_key(&mut s, Key::Esc);
        assert!(!s.help);
        on_key(&mut s, Key::Down);
        assert_eq!(s.cursor, 1);
    }

    /// Narrowing the row set under a cursor that sat past the new end is the
    /// crash this guards: the cursor must be pulled back into range.
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

    /// A window shrunk under the cursor must scroll rather than paint a
    /// cursor off screen.
    #[test]
    fn scrolling_keeps_the_cursor_visible_after_a_resize() {
        let mut s = st(100);
        s.scroll_into_view(20);
        on_key(&mut s, Key::End);
        s.scroll_into_view(20);
        assert!(s.cursor >= s.top && s.cursor < s.top + 20);
        // The window shrinks to five rows.
        s.scroll_into_view(5);
        assert!(
            s.cursor >= s.top && s.cursor < s.top + 5,
            "cursor {} not inside [{}, {})",
            s.cursor,
            s.top,
            s.top + 5
        );
    }
}
