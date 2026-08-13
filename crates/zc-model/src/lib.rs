//! Prediction: given measured hardware and a model spec, what will happen?
//!
//! Two outputs matter and they come from completely different physics:
//!
//!   decode  (tokens/sec)  memory-bandwidth bound. Every token re-reads every
//!                         active weight. Bytes moved is the whole story.
//!   prefill (TTFT)        compute bound. The prompt is processed in one pass
//!                         at high arithmetic intensity, so FLOPS is the story.
//!
//! Confusing the two is the most common error in local-LLM capacity planning:
//! it is why long prompts feel cheap and long answers feel expensive.

pub mod catalog;
pub mod fit;
pub mod gate;
pub mod json;
pub mod kv;
pub mod predict;
pub mod spec;

pub use fit::{Confidence, Fit};
pub use gate::Gate;
pub use kv::*;
pub use predict::*;
pub use spec::*;
