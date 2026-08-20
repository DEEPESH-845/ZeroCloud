//! What must happen before this process dies, even when it is killed.
//!
//! The rest of the codebase cleans up with `Drop`, which is right for normal
//! returns and for the `?` paths that motivated it. `Drop` does not run when a
//! signal terminates the process, because the process never unwinds — and the
//! disk benchmark holds a 512 MiB scratch file open through the slowest phase
//! of a run, which is exactly when an impatient user presses Ctrl-C. That leak
//! was reproduced before this module existed: the file stayed in the model
//! directory, hidden by its leading dot, on the volume a low-end laptop has
//! least room on.
//!
//! # What a signal handler may do
//!
//! Only async-signal-safe calls. `unlink(2)`, `write(2)`, `tcsetattr(3)` and
//! `raise(3)` are on the list; allocating, locking, formatting and `println!`
//! are not — calling them from a handler can deadlock against the very
//! allocator or lock the interrupted code was holding. So everything this
//! handler needs is prepared at *registration* time, while it is still
//! ordinary code, and the handler only reads it.
//!
//! # Windows
//!
//! There are no POSIX signals. Console-close handling belongs to crossterm,
//! and writing a second unvalidated cross-platform path blind is the mistake
//! this project has a standing rule against. Registration compiles to a no-op
//! there and the `Drop` guards remain the whole story.

#[cfg(unix)]
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

/// The scratch file to unlink, as a NUL-terminated path built while
/// allocation was still allowed. Null when there is nothing to remove.
#[cfg(unix)]
static SCRATCH: AtomicPtr<libc::c_char> = AtomicPtr::new(std::ptr::null_mut());

/// Whether the terminal is currently in raw mode on the alternate screen.
#[cfg(unix)]
static TERMINAL_RAW: AtomicBool = AtomicBool::new(false);

/// Terminal settings captured before raw mode was entered.
#[cfg(unix)]
static mut SAVED_TERMIOS: Option<libc::termios> = None;

/// Leave the alternate screen and show the cursor. A fixed literal because a
/// handler cannot build a string.
#[cfg(unix)]
const RESTORE: &[u8] = b"\x1b[?1049l\x1b[?25h";

#[cfg(unix)]
extern "C" fn handle(sig: libc::c_int) {
    // Unlink first: it is the confirmed leak, and it is the one thing that
    // outlives the process.
    let p = SCRATCH.swap(std::ptr::null_mut(), Ordering::SeqCst);
    if !p.is_null() {
        unsafe { libc::unlink(p) };
    }
    if TERMINAL_RAW.swap(false, Ordering::SeqCst) {
        unsafe {
            libc::write(
                libc::STDOUT_FILENO,
                RESTORE.as_ptr() as *const libc::c_void,
                RESTORE.len(),
            );
            let saved = &raw const SAVED_TERMIOS;
            if let Some(t) = *saved {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &t);
            }
        }
    }
    // Re-raise with the default disposition so the exit status still reports
    // the signal. A tool that swallows Ctrl-C and exits 0 lies to every script
    // that calls it.
    unsafe {
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

/// Install the handler once, for the signals that end a session.
#[cfg(unix)]
fn install() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        for sig in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
            // Through a fn pointer rather than casting the fn item straight
            // to an integer, which is what `sighandler_t` actually is.
            let f: extern "C" fn(libc::c_int) = handle;
            libc::signal(sig, f as *const () as libc::sighandler_t);
        }
    });
}

/// Remove `path` if a signal kills us before [`forget_file`] is called.
///
/// The path is converted and leaked here, while allocation is still safe. One
/// leaked path per process is not a leak worth caring about; a 512 MiB file on
/// the user's disk is.
#[cfg(unix)]
pub fn remove_on_signal(path: &std::path::Path) {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return; // A path containing NUL cannot be unlinked by a handler.
    };
    install();
    let raw = Box::into_raw(c.into_boxed_c_str()) as *mut libc::c_char;
    let old = SCRATCH.swap(raw, Ordering::SeqCst);
    if !old.is_null() {
        unsafe { drop(Box::from_raw(old as *mut libc::c_char)) };
    }
}

/// The file is gone by ordinary means; stop tracking it.
#[cfg(unix)]
pub fn forget_file() {
    let old = SCRATCH.swap(std::ptr::null_mut(), Ordering::SeqCst);
    if !old.is_null() {
        unsafe { drop(Box::from_raw(old as *mut libc::c_char)) };
    }
}

/// Tell the handler the terminal is in raw mode on the alternate screen, so a
/// signal hands it back rather than stranding the user's shell.
///
/// Ctrl-C is already safe inside the TUI — raw mode disables `ISIG`, so
/// crossterm delivers it as a key event — but `kill` and a closed terminal
/// are not.
#[cfg(unix)]
pub fn terminal_raw(raw: bool) {
    if raw {
        install();
        unsafe {
            let mut t: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut t) == 0 {
                let saved = &raw mut SAVED_TERMIOS;
                *saved = Some(t);
            }
        }
    }
    TERMINAL_RAW.store(raw, Ordering::SeqCst);
}

#[cfg(not(unix))]
pub fn remove_on_signal(_path: &std::path::Path) {}
#[cfg(not(unix))]
pub fn forget_file() {}
#[cfg(not(unix))]
pub fn terminal_raw(_raw: bool) {}

#[cfg(all(test, unix))]
mod tests {
    /// Registering, re-registering and forgetting must not leave a dangling
    /// pointer for the handler to read. Exercised in-process because the
    /// handler itself can only be observed by actually being signalled, which
    /// `scripts/signal_smoke.py` does against the real binary.
    #[test]
    fn registration_is_idempotent_and_reversible() {
        use std::sync::atomic::Ordering;
        super::remove_on_signal(std::path::Path::new("/tmp/zc-cleanup-test-a"));
        assert!(!super::SCRATCH.load(Ordering::SeqCst).is_null());
        super::remove_on_signal(std::path::Path::new("/tmp/zc-cleanup-test-b"));
        assert!(!super::SCRATCH.load(Ordering::SeqCst).is_null());
        super::forget_file();
        assert!(super::SCRATCH.load(Ordering::SeqCst).is_null());
        super::forget_file();
        assert!(super::SCRATCH.load(Ordering::SeqCst).is_null());
    }

    /// A path with an interior NUL cannot be handed to `unlink`, so it is
    /// declined rather than truncated into a path that means something else.
    #[test]
    fn a_path_containing_nul_is_declined() {
        use std::sync::atomic::Ordering;
        super::forget_file();
        super::remove_on_signal(std::path::Path::new("/tmp/a\0b"));
        assert!(super::SCRATCH.load(Ordering::SeqCst).is_null());
    }
}
