use std::ops::Mul;

use lambdaworks_math::polynomial::Polynomial;
use protocol::ByteConversion;
use protocol::mersenne_61::Sqrt;
use protocol::{LargeFieldSer, LargeField, vandermonde_matrix, inverse_vandermonde, matrix_vector_multiply};
use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};
use types::Replica;

use crate::{Context, msg::ProtMsg};

impl Context{
    pub async fn init_rand_bit_reconstruction(&mut self){
        if !self.mix_circuit_state.rand_bit_sharings.is_empty(){
            return;
        }
        if !self.mix_circuit_state.rand_bit_recon_shares.contains_key(&self.myid){
            return;
        }
        log::info!("Initializing batched random bit reconstruction");
        // Take ownership of this party's own reconstruction shares and split them
        // into `num_batches` batches, mirroring the random-sharing dealing
        // template. Every party splits an identically-sized share vector with the
        // same batch count, so batch `b` lines up across parties.
        let my_shares = self.mix_circuit_state.rand_bit_recon_shares.remove(&self.myid).unwrap();
        let num_batches = self.tot_batches.min(my_shares.len()).max(1);
        let per_batch = (my_shares.len() + num_batches - 1) / num_batches;

        let mut pending: Vec<Vec<LargeField>> = Vec::with_capacity(num_batches);
        let mut iter = my_shares.into_iter();
        for _ in 0..num_batches{
            let batch: Vec<LargeField> = iter.by_ref().take(per_batch).collect();
            if batch.is_empty(){
                break;
            }
            pending.push(batch);
        }

        self.mix_circuit_state.rand_bit_recon_pending = pending;
        self.mix_circuit_state.rand_bit_recon_cursor = 0;
        self.mix_circuit_state.rand_bit_recon_completed.clear();
        self.mix_circuit_state.rand_bit_recon_results.clear();
        self.mix_circuit_state.rand_bit_recon_broadcast.clear();

        // Broadcast only the first batch. Subsequent batches are broadcast one at
        // a time, each only after the in-flight batch has been reconstructed (see
        // `handle_reconstruct_rand_bits`), keeping at most one batch of shares in
        // flight so the per-batch buffers can be reclaimed.
        self.dispatch_next_rand_bit_batch().await;
    }

    /// Broadcast this party's next pending reconstruction batch, if any remain.
    /// The cursor is fast-forwarded over any batches that have already been
    /// reconstructed (their shares are already out from t+1 parties, so our own
    /// broadcast would be redundant) before broadcasting the next outstanding one.
    pub async fn dispatch_next_rand_bit_batch(&mut self){
        let total = self.mix_circuit_state.rand_bit_recon_pending.len();
        let mut cur = self.mix_circuit_state.rand_bit_recon_cursor;
        // Fast-forward past batches that are already reconstructed; drop our copy
        // of them since we won't broadcast them.
        while cur < total && self.mix_circuit_state.rand_bit_recon_completed.contains(&cur){
            self.mix_circuit_state.rand_bit_recon_pending[cur] = Vec::new();
            cur += 1;
        }
        if cur >= total{
            self.mix_circuit_state.rand_bit_recon_cursor = cur;
            return;
        }
        let batch = std::mem::take(&mut self.mix_circuit_state.rand_bit_recon_pending[cur]);
        self.mix_circuit_state.rand_bit_recon_cursor = cur + 1;
        // Guard against ever broadcasting the same batch twice.
        if !self.mix_circuit_state.rand_bit_recon_broadcast.insert(cur){
            log::debug!("Batch {} already broadcast, skipping", cur);
            return;
        }
        let batch_ser: Vec<LargeFieldSer> = batch.iter().map(|x| x.to_bytes_be()).collect();
        log::info!("Broadcasting random bit reconstruction shares for batch {}", cur);
        let prot_msg = ProtMsg::ReconstructRandBitShares(cur, batch_ser);
        self.broadcast(prot_msg).await;
    }

    pub async fn handle_reconstruct_rand_bits(&mut self, batch: usize, shares: Vec<LargeFieldSer>, share_sender: Replica){
        // Once every batch has been reconstructed and combined into rand bit
        // sharings, ignore all further messages.
        if !self.mix_circuit_state.rand_bit_sharings.is_empty(){
            return;
        }
        // Ignore messages belonging to an already-reconstructed batch. Batches
        // complete independently, so this can be any earlier-finished batch.
        if self.mix_circuit_state.rand_bit_recon_completed.contains(&batch){
            log::debug!("Ignoring rand bit reconstruction shares for already-reconstructed batch {} from sender {}", batch, share_sender);
            return;
        }
        log::info!("Handling reconstruction of random bit shares from sender {} for batch {}", share_sender, batch);
        let shares: Vec<LargeField> = shares.into_iter()
            .map(|x| LargeField::from_bytes_be(&x).unwrap())
            .collect();

        self.mix_circuit_state.rand_bit_recon_batches
            .entry(batch)
            .or_default()
            .insert(share_sender, shares);

        // Reconstruct this batch as soon as it reaches t+1 contributors, regardless
        // of the order batches arrive in. Reconstructing frees the batch's buffered
        // shares and triggers the broadcast of our next outstanding batch (whose
        // cursor fast-forwards over batches already done).
        let ready = self.mix_circuit_state.rand_bit_recon_batches
            .get(&batch)
            .map_or(0, |senders| senders.len()) >= self.num_faults + 1;
        if ready{
            self.reconstruct_rand_bit_batch(batch);
            // The buffered shares for this batch are now consumed; free them.
            self.mix_circuit_state.rand_bit_recon_batches.remove(&batch);
            self.mix_circuit_state.rand_bit_recon_completed.insert(batch);
            self.dispatch_next_rand_bit_batch().await;

            // Once all batches have been reconstructed, combine into rand bit sharings.
            let total = self.mix_circuit_state.rand_bit_recon_pending.len();
            if total > 0 && self.mix_circuit_state.rand_bit_recon_completed.len() >= total{
                self.verify_rand_bit_reconstruction().await;
            }
        }
    }

    /// Reconstruct the secrets of a single batch from its buffered per-sender
    /// shares and store the resulting square-root inverses under `batch` in
    /// `rand_bit_recon_results` (assembled in batch order at the end).
    fn reconstruct_rand_bit_batch(&mut self, batch: usize){
        log::info!("Received threshold number of shares for random bit batch {}, proceeding to reconstruct.", batch);
        let batch_shares = self.mix_circuit_state.rand_bit_recon_batches.get(&batch).unwrap();
        let shares_len = batch_shares.values().next().map_or(0, |v| v.len());

        let mut indices = Vec::new();
        let mut shares_index_wise = vec![vec![]; shares_len];

        for rep in 0..self.num_nodes{
            if let Some(rep_shares) = batch_shares.get(&rep){
                indices.push(Self::get_share_evaluation_point(rep, self.use_fft, self.roots_of_unity.clone()));
                for (index, share) in rep_shares.iter().enumerate(){
                    shares_index_wise[index].push(share.clone());
                }
            }
        }

        // generate inverse vandermonde matrix
        let vdm_matrix = vandermonde_matrix(indices);
        let inv_vdm_matrix = inverse_vandermonde(vdm_matrix);

        // Reconstruct each secret, take its square root in Fp4_61 via the
        // local `Sqrt` trait (Scott's complex method recursing Fp4 → Fp2 → Fp),
        // and invert. Matches async_mpc's pub_rec.rs:78 pattern — discard the
        // sign-choice branch (`let (sqrt, _) = ...`); the randomness comes
        // from upstream `r`, not from which root is picked.
        let reconstructed_square_inverses: Vec<LargeField> = shares_index_wise.into_par_iter()
            .map(|x| {
                let coefficients = matrix_vector_multiply(&inv_vdm_matrix, &x);
                let secret = Polynomial::new(&coefficients).evaluate(&LargeField::from(0 as u64));
                let (sqrt_root, _) = Sqrt::sqrt(&secret).expect("Square root does not exist");
                sqrt_root.inv()
            })
            .filter(|x| x.is_ok())
            .map(|x| x.unwrap())
            .collect();

        self.mix_circuit_state.rand_bit_recon_results.insert(batch, reconstructed_square_inverses);
        log::info!("Reconstructed random bit batch {}, batches completed: {}/{}",
            batch,
            self.mix_circuit_state.rand_bit_recon_completed.len() + 1,
            self.mix_circuit_state.rand_bit_recon_pending.len());
    }
    
    pub async fn handle_reconstruct_rand_bits_verify(&mut self, shares: Vec<LargeFieldSer>, share_sender: Replica){
        log::info!("Handling reconstruction of random bit verify shares from sender {}", share_sender);
        let shares: Vec<LargeField> = shares.into_iter()
            .map(|x| LargeField::from_bytes_be(&x).unwrap())
            .collect();

        let shares_len = shares.len();
        self.mix_circuit_state.rand_bit_reconstruction.insert(share_sender, shares);

        if self.mix_circuit_state.rand_bit_reconstruction.len() == self.num_faults+1{
            log::info!("Received threshold number of shares for random bit reconstruction, proceeding to reconstruct.");
            let mut indices = Vec::new();
            let mut shares_index_wise = vec![vec![];shares_len];
            
            for rep in 0..self.num_nodes{
                if self.mix_circuit_state.rand_bit_reconstruction.contains_key(&rep){
                    indices.push(Self::get_share_evaluation_point(rep, self.use_fft, self.roots_of_unity.clone()));
                    let rep_shares = self.mix_circuit_state.rand_bit_reconstruction.get(&rep).unwrap();
                    for (index, share) in rep_shares.iter().enumerate(){
                        shares_index_wise[index].push(share.clone());
                    }
                }
            }

            // generate inverse vandermonde matrix
            let vdm_matrix = vandermonde_matrix(indices);
            let inv_vdm_matrix = inverse_vandermonde(vdm_matrix);
            
            let one = LargeField::one();
            let mut reconstructed_square_inverses: Vec<LargeField> = shares_index_wise.into_par_iter()
                .map(|x| {
                    let coefficients = matrix_vector_multiply(&inv_vdm_matrix, &x);
                    let secret = Polynomial::new(&coefficients).evaluate(&LargeField::from(0 as u64));
                    secret
                }).collect();
            reconstructed_square_inverses.truncate(100);
            for secret in reconstructed_square_inverses{
                log::info!("Reconstructed random bit: {:?}", secret);
                log::info!("One: {:?}", one);
                log::info!("Minus one: {:?}", one.inv().unwrap());
            }
        }
    }

    pub async fn verify_rand_bit_reconstruction(&mut self){
        let total_batches = self.mix_circuit_state.rand_bit_recon_pending.len();
        // Wait until every batch has been reconstructed.
        if total_batches == 0 || self.mix_circuit_state.rand_bit_recon_completed.len() < total_batches{
            return;
        }
        if self.mix_circuit_state.rand_bit_inp_shares.is_empty(){
            return;
        }
        if !self.mix_circuit_state.rand_bit_sharings.is_empty(){
            return;
        }

        // Assemble per-batch reconstructed values back into the original order.
        // Batches were reconstructed independently / out of order, so we splice
        // them together by ascending batch index.
        let mut reconstructed_shares: Vec<LargeField> = Vec::new();
        for b in 0..total_batches{
            if let Some(vals) = self.mix_circuit_state.rand_bit_recon_results.remove(&b){
                reconstructed_shares.extend(vals);
            }
        }
        let rand_bit_input_shares = self.mix_circuit_state.rand_bit_inp_shares.clone();


        let final_rand_bit_sharings: Vec<LargeField> = rand_bit_input_shares.into_par_iter().zip(reconstructed_shares.into_par_iter()).map(|(r,re)|{
            let mult_share = r.mul(re);
            return mult_share
        }).collect();

        self.mix_circuit_state.rand_bit_sharings.extend(final_rand_bit_sharings.clone());

        // The reconstruction is complete; reclaim all per-batch reconstruction
        // buffers. Any further reconstruction messages are short-circuited at the
        // top of `handle_reconstruct_rand_bits` now that rand_bit_sharings is set.
        self.mix_circuit_state.rand_bit_recon_pending.clear();
        self.mix_circuit_state.rand_bit_recon_batches.clear();
        self.mix_circuit_state.rand_bit_recon_results.clear();
        self.mix_circuit_state.rand_bit_recon_broadcast.clear();

        //self.mix_circuit_state.rand_bit_sharings.extend(shares_next_depth);
        self.terminate("Preprocessing".to_string(), vec![]).await;
        // Start next depth and real circuit execution
        self.init_mixing().await;
    }
}