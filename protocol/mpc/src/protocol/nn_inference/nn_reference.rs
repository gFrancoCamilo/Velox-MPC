//! Deterministic-model mode, used to validate that the distributed evaluation
//! actually computes the network.
//!
//! When `NN_DETERMINISTIC=1` is set, every weight and input activation is a fixed
//! function of its index in the flat global array instead of a random field
//! element. Every party can then recompute the expected output in the clear and
//! compare it against the value the protocol reconstructed. Enable it only for
//! small parameters — the reference pass is a plaintext GEMM of the whole model.

use protocol::LargeField;

use super::nn_state::{layer_dims, layer_weight_offset, NUM_NN_LAYERS};

/// Offset separating the activation index space from the weight index space, so
/// a weight and an activation never share a value by construction.
const ACTIVATION_INDEX_BASE: usize = 1 << 40;

pub fn deterministic_mode() -> bool {
    std::env::var("NN_DETERMINISTIC").map_or(false, |v| v == "1")
}

/// Value assigned to global index `g`. Kept small and non-zero so that a
/// mis-indexed read shows up as a wrong product rather than a silent zero.
pub fn deterministic_value(g: usize) -> LargeField {
    LargeField::from(((g % 1021) + 1) as u64)
}

pub fn weight_value(flat_index: usize) -> LargeField {
    deterministic_value(flat_index)
}

pub fn activation_value(flat_index: usize) -> LargeField {
    deterministic_value(ACTIVATION_INDEX_BASE + flat_index)
}

/// Plaintext forward pass over the deterministic model, in example-major order:
/// `out[i * nn_y + j]`.
pub fn reference_output(nn_x: usize, nn_y: usize, nn_batch: usize) -> Vec<LargeField> {
    // Stage input: the network's activations, one row per example.
    let mut acts: Vec<Vec<LargeField>> = (0..nn_batch)
        .map(|i| (0..nn_x).map(|k| activation_value(i * nn_x + k)).collect())
        .collect();

    for layer in 1..=NUM_NN_LAYERS {
        let (d_in, d_out) = layer_dims(layer, nn_x, nn_y);
        let base = layer_weight_offset(layer, nn_x);
        let next: Vec<Vec<LargeField>> = acts
            .iter()
            .map(|row| {
                (0..d_out)
                    .map(|j| {
                        let mut sum = LargeField::zero();
                        for k in 0..d_in {
                            sum += &row[k] * &weight_value(base + k * d_out + j);
                        }
                        sum
                    })
                    .collect()
            })
            .collect();
        acts = next;
    }

    acts.into_iter().flatten().collect()
}
