use protocol::LargeField;

use crate::Context;

use super::nn_state::{NUM_NN_LAYERS, layer_dims};

impl Context {
    /// Dispatch one weight-combination stage.
    ///
    /// Stage `l` computes `Z[i][j] = <x_i, W[:,j]>` for every example `i` in the
    /// batch and every output neuron `j`. Each of those `b * d_out` inner products
    /// is a single DN reduction consuming one `([r]_t, [o]_2t)` pair regardless of
    /// its length `d_in` — the whole point of routing this through the inner-product
    /// path rather than `d_in` separate multiplications.
    pub async fn init_nn_layer(&mut self, layer: usize) {
        if self.nn_state.layer_started.contains(&layer) {
            return;
        }
        if !self.nn_state.activations.contains_key(&layer) {
            log::error!("init_nn_layer: activations for stage {} are missing", layer);
            return;
        }
        self.nn_state.layer_started.insert(layer);

        let (d_in, d_out) = layer_dims(layer, self.nn_x, self.nn_y);
        // Both operands are moved out, not cloned: at benchmark scale a stage's
        // weight matrix is hundreds of MB and is never needed again once its
        // reduction is in flight.
        let activations = self.nn_state.activations.remove(&layer).unwrap();
        let weight_cols = std::mem::take(&mut self.nn_state.weights[layer - 1]);

        if activations.len() != self.nn_batch || weight_cols.len() != d_out {
            log::error!(
                "init_nn_layer: stage {} shape mismatch — {} activation vectors (expected {}), \
                 {} weight columns (expected {})",
                layer,
                activations.len(),
                self.nn_batch,
                weight_cols.len(),
                d_out
            );
            return;
        }

        log::info!(
            "Starting NN stage {}: {} x {} weights, batch {} -> {} inner products of dimension {}",
            layer,
            d_in,
            d_out,
            self.nn_batch,
            self.nn_batch * d_out,
            d_in
        );

        Box::pin(self.init_layer_multiplication(activations, weight_cols, layer)).await;
    }

    /// Consume a stage's DN output and either feed the next stage or finish.
    ///
    /// `mult_result` is `b * d_out` values in example-major order, matching the
    /// operand order built by `init_layer_multiplication`.
    pub async fn verify_nn_layer_termination(&mut self, layer: usize, mult_result: Vec<LargeField>) {
        let (_, d_out) = layer_dims(layer, self.nn_x, self.nn_y);
        let expected = self.nn_batch * d_out;
        if mult_result.len() != expected {
            log::error!(
                "verify_nn_layer_termination: stage {} produced {} values, expected {}",
                layer,
                mult_result.len(),
                expected
            );
            return;
        }
        log::info!("NN stage {} complete with {} output wires", layer, mult_result.len());

        if layer < NUM_NN_LAYERS {
            let next: Vec<Vec<LargeField>> =
                mult_result.chunks(d_out).map(|c| c.to_vec()).collect();
            self.nn_state.activations.insert(layer + 1, next);
            Box::pin(self.init_nn_layer(layer + 1)).await;
        } else {
            log::info!(
                "Inference complete: {} output wires ({} examples x {} classes)",
                mult_result.len(),
                self.nn_batch,
                self.nn_y
            );
            self.mult_state.output_layer.output_shares = Some((
                Self::get_share_evaluation_point(self.myid, self.use_fft, self.roots_of_unity.clone()),
                mult_result,
            ));
            self.terminate("Online".to_string(), vec![]).await;
            if self.verification_enabled {
                self.delinearize_mult_tuples().await;
            } else {
                // Tuple verification is disabled (see Context::verification_enabled);
                // go straight to output reconstruction.
                self.reconstruct_output().await;
            }
        }
    }
}
