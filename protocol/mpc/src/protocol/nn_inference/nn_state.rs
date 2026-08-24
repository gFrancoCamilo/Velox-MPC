use std::collections::{HashMap, HashSet};

use protocol::LargeField;
use types::Replica;

/// Number of weight-combination stages in the network. The model is
/// `[x, x, x, y]`, so there are three all-to-all weight matrices:
/// `W1: x*x`, `W2: x*x`, `W3: x*y`.
pub const NUM_NN_LAYERS: usize = 3;

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

/// `(d_in, d_out)` for a 1-indexed stage of the `[x, x, x, y]` network.
pub fn layer_dims(layer: usize, nn_x: usize, nn_y: usize) -> (usize, usize) {
    match layer {
        1 => (nn_x, nn_x),
        2 => (nn_x, nn_x),
        3 => (nn_x, nn_y),
        _ => panic!("layer_dims: stage {} out of range 1..={}", layer, NUM_NN_LAYERS),
    }
}

/// Total weight count of the network: `2x^2 + xy`.
pub fn total_weights(nn_x: usize, nn_y: usize) -> usize {
    2 * nn_x * nn_x + nn_x * nn_y
}

/// Offset of stage `layer`'s weight matrix in the flat global weight array
/// `[W1 | W2 | W3]`, each stage stored row-major over `(in_index, out_index)`.
pub fn layer_weight_offset(layer: usize, nn_x: usize) -> usize {
    match layer {
        1 => 0,
        2 => nn_x * nn_x,
        3 => 2 * nn_x * nn_x,
        _ => panic!("layer_weight_offset: stage {} out of range", layer),
    }
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
