use std::collections::HashSet;

use protocol::ByteConversion;
use protocol::{LargeField, LargeFieldSer, rand_field_element};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use types::Replica;

use crate::Context;

use super::nn_reference::{activation_value, deterministic_mode, weight_value};
use super::nn_state::{
    ACTIVATION_ACSS_ID_OFFSET, NUM_NN_LAYERS, WEIGHT_ACSS_ID_OFFSET, div_ceil, layer_dims,
    layer_weight_offset, rss_mib, total_weights,
};

impl Context {
    /// Weights per party. The flat global weight array is padded up to
    /// `n * weights_per_party` so every party deals *exactly* the same count —
    /// no party carries a short final block.
    pub fn weights_per_party(&self) -> usize {
        div_ceil(total_weights(self.nn_x, self.nn_y), self.num_nodes)
    }

    /// Input activations per party, padded the same way. The network input is
    /// `nn_batch` vectors of length `nn_x`.
    pub fn activations_per_party(&self) -> usize {
        div_ceil(self.nn_batch * self.nn_x, self.num_nodes)
    }

    pub fn num_weight_chunks(&self) -> usize {
        div_ceil(self.weights_per_party(), self.weight_chunk_size)
    }

    pub fn num_activation_chunks(&self) -> usize {
        div_ceil(self.activations_per_party(), self.weight_chunk_size)
    }

    /// Input phase. Every party secret-shares an equal-sized block of the model
    /// weights and of the input activations via ACSS-Ab.
    ///
    /// Unlike the preprocessing phase, this does **not** run an ACS: the online
    /// phase starts only once *all* `n` dealers have delivered every chunk, so
    /// the assembled weight matrix has no holes. That costs asynchronous
    /// liveness — a single crashed party stalls the run — which is the intended
    /// trade for a benchmark that must evaluate the real model.
    pub async fn init_input_phase(&mut self) {
        let per_party = self.weights_per_party();
        let chunk = self.weight_chunk_size;
        let num_chunks = self.num_weight_chunks();

        log::info!(
            "Input phase: sharing {} weights ({} chunks of <= {}) and {} activations ({} chunks); \
             network [{}, {}, {}, {}], batch {}",
            per_party,
            num_chunks,
            chunk,
            self.activations_per_party(),
            self.num_activation_chunks(),
            self.nn_x,
            self.nn_x,
            self.nn_x,
            self.nn_y,
            self.nn_batch
        );

        let deterministic = deterministic_mode();
        if deterministic {
            log::warn!("NN_DETERMINISTIC=1: weights and activations are fixed functions of their \
                        global index, and the output will be checked against a plaintext forward pass");
        }
        // This party owns the contiguous block starting here in the flat global
        // weight array; the equitable split means every party owns exactly
        // `per_party` entries (the array is zero-padded up to n * per_party).
        let weight_block_base = self.myid * per_party;

        for chunk_idx in 0..num_chunks {
            let this_chunk = chunk.min(per_party - chunk_idx * chunk);
            let chunk_base = weight_block_base + chunk_idx * chunk;
            let values: Vec<LargeFieldSer> = (0..this_chunk)
                .into_par_iter()
                .map(|i| {
                    if deterministic {
                        weight_value(chunk_base + i).to_bytes_be()
                    } else {
                        rand_field_element().to_bytes_be()
                    }
                })
                .collect();
            let status = self
                .acss_ab_send
                .send((WEIGHT_ACSS_ID_OFFSET + chunk_idx, values))
                .await;
            if status.is_err() {
                log::error!(
                    "Failed to send weight chunk {} to ACSS: {:?}",
                    chunk_idx,
                    status.err().unwrap()
                );
            }
        }

        let act_per_party = self.activations_per_party();
        let num_act_chunks = self.num_activation_chunks();
        let act_block_base = self.myid * act_per_party;
        for chunk_idx in 0..num_act_chunks {
            let this_chunk = chunk.min(act_per_party - chunk_idx * chunk);
            let chunk_base = act_block_base + chunk_idx * chunk;
            let values: Vec<LargeFieldSer> = (0..this_chunk)
                .into_par_iter()
                .map(|i| {
                    if deterministic {
                        activation_value(chunk_base + i).to_bytes_be()
                    } else {
                        rand_field_element().to_bytes_be()
                    }
                })
                .collect();
            let status = self
                .acss_ab_send
                .send((ACTIVATION_ACSS_ID_OFFSET + chunk_idx, values))
                .await;
            if status.is_err() {
                log::error!(
                    "Failed to send activation chunk {} to ACSS: {:?}",
                    chunk_idx,
                    status.err().unwrap()
                );
            }
        }
    }

    pub async fn handle_weight_acss_termination(
        &mut self,
        instance_id: usize,
        sender: Replica,
        shares: Option<Vec<LargeFieldSer>>,
    ) {
        if self.nn_state.input_phase_done {
            return;
        }
        let shares = match shares {
            Some(s) => s,
            None => {
                log::error!(
                    "Weight ACSS of dealer {} aborted; the all-n input barrier cannot be met",
                    sender
                );
                return;
            }
        };
        let chunk_idx = instance_id - WEIGHT_ACSS_ID_OFFSET;
        let offset = sender * self.weights_per_party() + chunk_idx * self.weight_chunk_size;
        if self.nn_state.weights_flat.is_empty() {
            self.nn_state.weights_flat =
                vec![LargeField::zero(); self.weights_per_party() * self.num_nodes];
        }
        // Deserialize straight into the flat array — no per-chunk Vec is retained.
        self.nn_state.weights_flat[offset..offset + shares.len()]
            .par_iter_mut()
            .zip(shares.into_par_iter())
            .for_each(|(slot, x)| *slot = LargeField::from_bytes_be(&x).unwrap());
        self.nn_state
            .weight_chunks_seen
            .entry(sender)
            .or_insert_with(HashSet::default)
            .insert(chunk_idx);
        self.check_input_phase_barrier().await;
    }

    pub async fn handle_activation_acss_termination(
        &mut self,
        instance_id: usize,
        sender: Replica,
        shares: Option<Vec<LargeFieldSer>>,
    ) {
        if self.nn_state.input_phase_done {
            return;
        }
        let shares = match shares {
            Some(s) => s,
            None => {
                log::error!(
                    "Activation ACSS of dealer {} aborted; the all-n input barrier cannot be met",
                    sender
                );
                return;
            }
        };
        let chunk_idx = instance_id - ACTIVATION_ACSS_ID_OFFSET;
        let offset = sender * self.activations_per_party() + chunk_idx * self.weight_chunk_size;
        if self.nn_state.activations_flat.is_empty() {
            self.nn_state.activations_flat =
                vec![LargeField::zero(); self.activations_per_party() * self.num_nodes];
        }
        self.nn_state.activations_flat[offset..offset + shares.len()]
            .par_iter_mut()
            .zip(shares.into_par_iter())
            .for_each(|(slot, x)| *slot = LargeField::from_bytes_be(&x).unwrap());
        self.nn_state
            .activation_chunks_seen
            .entry(sender)
            .or_insert_with(HashSet::default)
            .insert(chunk_idx);
        self.check_input_phase_barrier().await;
    }

    /// Barrier: every dealer, every chunk. Idempotent.
    pub async fn check_input_phase_barrier(&mut self) {
        if self.nn_state.input_phase_done {
            return;
        }
        let num_w = self.num_weight_chunks();
        let num_a = self.num_activation_chunks();
        for party in 0..self.num_nodes {
            let w_ok = self
                .nn_state
                .weight_chunks_seen
                .get(&party)
                .map_or(false, |m| m.len() == num_w);
            let a_ok = self
                .nn_state
                .activation_chunks_seen
                .get(&party)
                .map_or(false, |m| m.len() == num_a);
            if !w_ok || !a_ok {
                return;
            }
        }
        log::info!("Input phase: all {} dealers delivered every chunk, assembling model", self.num_nodes);
        self.nn_state.input_phase_done = true;
        self.assemble_model();
        self.terminate("Input".to_string(), vec![]).await;
        self.init_nn_layer(1).await;
    }

    /// Splice the per-dealer blocks back into the flat global arrays in dealer
    /// order, then reshape the weights into per-stage column-major matrices.
    fn assemble_model(&mut self) {
        self.nn_state.weight_chunks_seen.clear();
        self.nn_state.activation_chunks_seen.clear();

        let mut flat_weights = std::mem::take(&mut self.nn_state.weights_flat);
        flat_weights.truncate(total_weights(self.nn_x, self.nn_y));

        let mut flat_activations = std::mem::take(&mut self.nn_state.activations_flat);
        flat_activations.truncate(self.nn_batch * self.nn_x);

        // Each stage is stored row-major over (in_index, out_index) in the flat
        // array; transpose into columns here, once, so every layer's DN call can
        // hand its weight columns straight to the inner-product path. Stages are
        // built back-to-front and split off the flat array as they are consumed,
        // so peak footprint is the model plus one stage rather than twice the model.
        let mut weights: Vec<Vec<Vec<LargeField>>> = Vec::with_capacity(NUM_NN_LAYERS);
        for _ in 0..NUM_NN_LAYERS {
            weights.push(Vec::new());
        }
        for layer in (1..=NUM_NN_LAYERS).rev() {
            let (d_in, d_out) = layer_dims(layer, self.nn_x, self.nn_y);
            let base = layer_weight_offset(layer, self.nn_x);
            let stage = flat_weights.split_off(base);
            flat_weights.shrink_to_fit();
            let columns: Vec<Vec<LargeField>> = (0..d_out)
                .into_par_iter()
                .map(|j| (0..d_in).map(|i| stage[i * d_out + j].clone()).collect())
                .collect();
            weights[layer - 1] = columns;
        }
        drop(flat_weights);
        self.nn_state.weights = weights;

        let input_batch: Vec<Vec<LargeField>> = flat_activations
            .chunks(self.nn_x)
            .map(|c| c.to_vec())
            .collect();
        drop(flat_activations);
        self.nn_state.activations.insert(1, input_batch);

        log::info!(
            "Model assembled: {} weights across {} stages, {} input vectors of length {} (RSS {} MiB)",
            total_weights(self.nn_x, self.nn_y),
            NUM_NN_LAYERS,
            self.nn_batch,
            self.nn_x,
            rss_mib()
        );
    }
}
