use std::collections::{HashMap, HashSet};

use protocol::LargeField;
use types::Replica;

/// Number of weight-combination stages for a network of the given layer widths.
/// Widths `[d0, d1, ..., dL]` describe `L` all-to-all weight matrices, stage `l`
/// being `d_{l-1} x d_l`.
pub fn num_stages(widths: &[usize]) -> usize {
    widths.len() - 1
}

/// ACSS instance-id base for weight blocks. Each dealer's block is split into
/// chunks at ids `WEIGHT_ACSS_ID_OFFSET + chunk_index`.
pub const WEIGHT_ACSS_ID_OFFSET: usize = 1000;

/// ACSS instance-id base for input-activation blocks.
pub const ACTIVATION_ACSS_ID_OFFSET: usize = 2000;

pub struct NnState {
    // ---- input-phase bookkeeping ----
    /// Flat global weight-share array, preallocated and written in place as each
    /// dealer's chunk arrives. Holding the chunks in a map first and splicing
    /// afterwards cost an extra full copy of the model.
    pub weights_flat: Vec<LargeField>,
    /// Flat global input-activation share array, same scheme.
    pub activations_flat: Vec<LargeField>,
    /// dealer -> set of chunk indices already written, for the all-n barrier.
    pub weight_chunks_seen: HashMap<Replica, HashSet<usize>>,
    pub activation_chunks_seen: HashMap<Replica, HashSet<usize>>,
    /// Set once every dealer has delivered every chunk and the model is assembled.
    pub input_phase_done: bool,

    // ---- assembled model ----
    /// `weights[l][j]` is column `j` of stage `l`'s weight matrix, length `d_in(l)`.
    /// Column-major because that is exactly the operand shape the DN inner-product
    /// path consumes — no transpose at layer time.
    pub weights: Vec<Vec<Vec<LargeField>>>,
    /// `activations[l]` holds the `b` input vectors of stage `l`, each of length
    /// `d_in(l)`. Stage 1's entry is the network input; stage `l+1`'s entry is
    /// produced by stage `l`'s DN reduction.
    pub activations: HashMap<usize, Vec<Vec<LargeField>>>,
    /// Guards layer dispatch against the re-entrant termination checks.
    pub layer_started: HashSet<usize>,
}

impl NnState {
    pub fn new() -> Self {
        NnState {
            weights_flat: Vec::new(),
            activations_flat: Vec::new(),
            weight_chunks_seen: HashMap::default(),
            activation_chunks_seen: HashMap::default(),
            input_phase_done: false,
            weights: Vec::new(),
            activations: HashMap::default(),
            layer_started: HashSet::default(),
        }
    }
}

/// `(d_in, d_out)` for a 1-indexed stage.
pub fn layer_dims(layer: usize, widths: &[usize]) -> (usize, usize) {
    assert!(
        layer >= 1 && layer < widths.len(),
        "layer_dims: stage {} out of range 1..={}",
        layer,
        num_stages(widths)
    );
    (widths[layer - 1], widths[layer])
}

/// Total weight count of the network: `sum_l d_{l-1} * d_l`.
pub fn total_weights(widths: &[usize]) -> usize {
    (1..widths.len()).map(|l| widths[l - 1] * widths[l]).sum()
}

/// Offset of stage `layer`'s weight matrix in the flat global weight array
/// `[W1 | W2 | ... | WL]`, each stage stored row-major over `(in_index, out_index)`.
pub fn layer_weight_offset(layer: usize, widths: &[usize]) -> usize {
    (1..layer).map(|l| widths[l - 1] * widths[l]).sum()
}

/// Inner products per forward pass: one per (example, output neuron) over every
/// stage. Note this depends only on the *output* widths — a wider input layer
/// costs more local GEMM but not one extra double sharing.
pub fn total_inner_products(widths: &[usize], batch: usize) -> usize {
    batch * widths[1..].iter().sum::<usize>()
}

/// Ceiling division, used everywhere the flat arrays are cut into equal blocks.
pub fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Resident set size in MiB, read from /proc/self/statm. Logged at phase
/// boundaries so memory growth is attributable to a phase rather than guessed at.
pub fn rss_mib() -> usize {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).map(|v| v.to_string()))
        .and_then(|v| v.parse::<usize>().ok())
        .map(|pages| pages * 4096 / (1024 * 1024))
        .unwrap_or(0)
}
