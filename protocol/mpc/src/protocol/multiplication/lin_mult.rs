use std::{collections::HashMap, ops::Add};

use crate::Context;

use bincode::{Result};
use crypto::hash::do_hash;
use lambdaworks_math::{polynomial::Polynomial};
use protocol::ByteConversion;
use protocol::{LargeField, LargeFieldSer, vandermonde_matrix, inverse_vandermonde, matrix_vector_multiply, matrix_matrix_multiply, powers_matrix};
use rayon::prelude::{ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use types::{Replica, WrapperMsg};

use crate::{msg::ProtMsg};

impl Context{
    /// Generic entry point: one DN inner product per `(a_vec_shares[g], b_vec_shares[g])`
    /// pair. Used by the tuple-verification path, where the operand pairs are
    /// unrelated to each other.
    pub async fn init_linear_multiplication_prot(&mut self, a_vec_shares: Vec<Vec<LargeField>>, b_vec_shares: Vec<Vec<LargeField>>, depth: usize) {
        // Recording every component of every gate is what makes the verification
        // state proportional to the total scalar-product count rather than the
        // gate count; at NN-inference scale that is ~10^10 elements, so it stays
        // off until the rolling/streaming verification lands.
        if self.verification_enabled && depth <= self.max_depth {
            let first_a_shares: Vec<LargeField> = a_vec_shares.iter().map(|x| x[0].clone()).collect();
            let first_b_shares: Vec<LargeField> = b_vec_shares.iter().map(|x| x[0].clone()).collect();
            log::info!("Adding shares to verification state with a:{} b:{} at depth {}", first_a_shares.len(), first_b_shares.len(), depth);
            self.verf_state.add_mult_inputs(depth, first_a_shares, first_b_shares);
        }

        let products: Vec<LargeField> = a_vec_shares
            .par_iter()
            .zip_eq(b_vec_shares.par_iter())
            .map(|(a, b)| Self::dot_product(a, b))
            .collect();
        drop(a_vec_shares);
        drop(b_vec_shares);
        Box::pin(self.run_dn_inner_products(products, depth)).await;
    }

    /// Dense-layer entry point: every activation vector is dotted against every
    /// weight column, so the whole stage's `b * d_out` products are one GEMM
    /// rather than `b * d_out` separate dot-product calls.
    ///
    /// This also avoids materialising the tiled left operand. Feeding the generic
    /// path would require `b * d_out` copies of the activation vector — at
    /// `b=256, x=4096` that is 4.3e9 field elements, ~34 GB. Here the memory is
    /// just the two operands.
    ///
    /// Output order is example-major: `products[i * d_out + j] = <x_i, W[:,j]>`.
    pub async fn init_layer_multiplication(
        &mut self,
        activations: Vec<Vec<LargeField>>,
        weight_cols: Vec<Vec<LargeField>>,
        depth: usize,
    ) {
        let d_out = weight_cols.len();
        let batch = activations.len();

        // row_major = true → evals[j][i] = <weight_cols[j], activations[i]>.
        let evals = matrix_matrix_multiply(&weight_cols, &activations, true);
        drop(weight_cols);
        drop(activations);

        let mut products = Vec::with_capacity(batch * d_out);
        for i in 0..batch {
            for j in 0..d_out {
                products.push(evals[j][i].clone());
            }
        }
        drop(evals);
        Box::pin(self.run_dn_inner_products(products, depth)).await;
    }

    /// The DN reduction proper. Takes the *unmasked* inner-product results
    /// `products[g] = <a_g, b_g>`, masks each with a fresh degree-`t` sharing,
    /// packs groups of `2t+1` into a degree-`2t` polynomial masked by a degree-`2t`
    /// sharing of zero, and ships one evaluation per party.
    pub async fn run_dn_inner_products(&mut self, mut products: Vec<LargeField>, depth: usize) {
        let multiple_of_val = 2*self.num_faults+1;
        let mut padding_length = multiple_of_val - (products.len()%multiple_of_val);
        if (products.len()%multiple_of_val) == 0{
            padding_length =0;
        }
        // Pad with zero products until the count is a multiple of 2t+1.
        for _ in 0..padding_length{
            products.push(LargeField::zero());
        }
        let tot_groups = products.len() / (2 * self.num_faults + 1);
        let tot_shares = products.len();
        
        {
            let depth_state;
            if !self.mult_state.depth_share_map.contains_key(&depth){
                depth_state = self.mult_state.get_single_depth_state(depth, true, tot_groups);
            }
            else{
                depth_state = self.mult_state.depth_share_map.get_mut(&depth).unwrap();
            }
            depth_state.padding_shares = padding_length;
        }

        // Drain the degree-t masks as one block. The old form pushed each mask into
        // both a local vector and the depth state, keeping two full copies alive for
        // the whole call; here the block is *moved* into the depth state once the
        // z-vectors have been built off it.
        if self.rand_sharings_state.rand_sharings_mult.len() < tot_shares {
            log::error!("Not enough random shares for linear multiplication: have {}, need {}",
                self.rand_sharings_state.rand_sharings_mult.len(), tot_shares);
            return;
        }
        let r_sharings: Vec<LargeField> = self.rand_sharings_state.rand_sharings_mult.drain(..tot_shares).collect();

        // One degree-2t zero sharing per (t+1) of each (2t+1)-sized group.
        let o_count = tot_groups * (self.num_faults + 1);
        if self.rand_sharings_state.rand_2t_sharings_mult.len() < o_count {
            log::error!("Not enough 2t sharings for linear multiplication: have {}, need {}",
                self.rand_sharings_state.rand_2t_sharings_mult.len(), o_count);
            return;
        }
        let o_sharings: Vec<LargeField> = self.rand_sharings_state.rand_2t_sharings_mult.drain(..o_count).collect();

        let total_chunks = tot_groups;

        let vandermonde_points: Vec<LargeField> = (2..self.num_nodes+2).into_iter().map(|x| LargeField::from(x as u64)).collect();
        let vdm_matrix = Self::vandermonde_matrix(vandermonde_points, self.num_faults);

        // Build every chunk's z_vector and o_vec first, then do ONE GEMM across all
        // chunks to evaluate them at the n party points. Per-chunk GEMM lost ~6x to
        // the scalar loop because each call paid Rayon setup overhead for a tiny
        // 16x11 product; bench `BatchedPartyEval` characterizes the right shape.
        let z_vector_len = multiple_of_val;
        let o_group_len = self.num_faults + 1;
        let party_powers = powers_matrix(&self.roots_of_unity, z_vector_len);

        // Chunks are read as slices of the flat vectors — the previous grouped
        // `chunks(..).map(to_vec)` copies duplicated every operand.
        let mut z_vectors: Vec<Vec<LargeField>> = Vec::with_capacity(total_chunks);
        let mut o_vecs: Vec<Vec<LargeField>> = Vec::with_capacity(total_chunks);
        for i in 0..total_chunks {
            o_vecs.push(Self::matrix_vector_multiply(
                &vdm_matrix,
                &o_sharings[i * o_group_len..(i + 1) * o_group_len],
            ));
            let base = i * z_vector_len;
            let mut z_vector = Vec::with_capacity(z_vector_len);
            for k in 0..z_vector_len {
                z_vector.push(products[base + k].clone().add(r_sharings[base + k].clone()));
            }
            z_vectors.push(z_vector);
        }
        drop(products);
        drop(o_sharings);

        {
            let depth_state = self.mult_state.depth_share_map.get_mut(&depth).unwrap();
            if depth_state.util_rand_sharings.is_empty() {
                depth_state.util_rand_sharings = r_sharings;
            } else {
                depth_state.util_rand_sharings.extend(r_sharings);
            }
        }

        // One GEMM: party_powers (n × (2t+1)) · z_vectors (chunks vectors of length (2t+1))
        // → evals (n × chunks). evals[p][chunk] is the share for party `p` in chunk `chunk`,
        // pre-`o_vec` add. Replaces the previous per-chunk `Polynomial::new(&z).evaluate(&el)`
        // loop; `Polynomial::new`'s trailing-zero trim is a no-op for correctness here since
        // zero coefficients contribute zero to the GEMM dot product.
        let evals = matrix_matrix_multiply(&party_powers, &z_vectors, true);

        let mut shares_party: HashMap<usize, Vec<LargeField>> = HashMap::default();
        for party in 0..self.num_nodes {
            shares_party.insert(party, Vec::with_capacity(tot_shares));
        }
        for i in 0..total_chunks {
            for p in 0..self.num_nodes {
                let share = evals[p][i].clone() + o_vecs[i][p].clone();
                shares_party.get_mut(&p).unwrap().push(share);
            }
        }

        // Send shares for all groups to all parties
        for (party,shares) in shares_party.into_iter(){
            let ser_shares: Vec<LargeFieldSer> = shares.into_iter().map(|share| {
                share.to_bytes_be()
            }).collect();
            // Encrypt shares before putting them in a message
            let ser_shares_bytes = bincode::serialize(&ser_shares).unwrap();
            let sec_key = self.sec_key_map.get(&party).clone().unwrap();

            // let encrypted_msg = encrypt(sec_key, ser_shares_bytes);
            let prot_msg = ProtMsg::SharesL1(ser_shares_bytes, depth);

            let wrapper_msg = WrapperMsg::new(prot_msg, self.myid, &sec_key);
            let cancel_handler = self.net_send.send(party, wrapper_msg).await;

            self.add_cancel_handler(cancel_handler);
        }
        self.verify_depth_mult_termination(depth).await;
    }

    pub async fn handle_l1_message(&mut self, ser_shares: Vec<u8>, depth: usize, sender: usize) {
        // Try deserializing the message now
        log::info!("Received L1 multiplication shares from party {} for depth {}", sender, depth);
        let shares_option: Result<Vec<LargeFieldSer>> = bincode::deserialize(&ser_shares);
        if shares_option.is_err() {
            log::error!("Error deserializing shares: {:?}", shares_option.err());
            return;
        }

        let shares_ser = shares_option.unwrap();
        
        // Received message as L1 share so multiplication at this depth must be linear
        
        let shares: Vec<LargeField> = shares_ser.into_iter().map(|share| {
            return LargeField::from_bytes_be(&share).unwrap();
        }).collect();

        let depth_state;
        if !self.mult_state.depth_share_map.contains_key(&depth){
            depth_state = self.mult_state.get_single_depth_state(depth, true, shares.len());
        }
        else{
            depth_state = self.mult_state.depth_share_map.get_mut(&depth).unwrap();
        }
        // At L1, the evaluation point is the point at which the polynomials have been evaluated. 
        let evaluation_point = Self::get_share_evaluation_point(sender, self.use_fft.clone(), self.roots_of_unity.clone());
        depth_state.l1_shares.0.push(evaluation_point);
        for (index, share) in shares.into_iter().enumerate(){
            depth_state.l1_shares.1[index].push(share);
        }
        
        depth_state.recv_share_count_l1 +=1;
        //depth_state.recv_share_count_l1 = depth_state.recv_share_count_l1.clone().add(1).into();
        let mut ser_shares = None;
        if depth_state.recv_share_count_l1 == self.num_nodes - self.num_faults {
            log::info!("Attempting L1 reconstruction at depth {}", depth);
            // Start reconstruction here
            let indices = depth_state.l1_shares.0.clone();
            let vdm_matrix = vandermonde_matrix(indices);

            let inv_vdm_matrix = inverse_vandermonde(vdm_matrix);
            let secrets: Vec<LargeField> = depth_state.l1_shares.1.par_iter().map(|group_shares|{
                let coefficients = matrix_vector_multiply(&inv_vdm_matrix, &group_shares);
                let poly = Polynomial::new(&coefficients);
                let secret = poly.evaluate(&LargeField::zero()); // Evaluate at zero to get the secret
                return secret;
            }).collect();

            depth_state.l1_shares_reconstructed.extend(secrets.clone());

            let shares_bytes: Vec<LargeFieldSer> = secrets.into_iter().map(|el| el.to_bytes_be()).collect();
            ser_shares = Some(bincode::serialize(&shares_bytes).unwrap());
        }

        if ser_shares.is_some(){
            log::info!("L1 reconstruction successful, sending L2 shares to all parties");
            self.broadcast(ProtMsg::SharesL2(ser_shares.unwrap(), depth)).await;
        }
        self.verify_depth_mult_termination(depth).await;
    }

    pub async fn handle_l2_message(&mut self, group_shares: Vec<u8>, depth: usize, sender: Replica){
        // Multiplication at this depth is of course using two levels of mult
        log::info!("Received L2 multiplication shares from party {} for depth {}", sender, depth);
        let group_shares: Vec<LargeFieldSer> = bincode::deserialize(&group_shares).unwrap();
        
        let depth_state;
        if !self.mult_state.depth_share_map.contains_key(&depth){
            depth_state = self.mult_state.get_single_depth_state(depth, true, group_shares.len());
        }
        else{
            depth_state = self.mult_state.depth_share_map.get_mut(&depth).unwrap();
        }
        
        // At this depth, we are using roots of unity to conduct evaluation
        let evaluation_point = self.roots_of_unity.get(sender).clone().unwrap();
        depth_state.l2_shares.0.push(evaluation_point.clone());
        for (state,group_share) in depth_state.l2_shares.1.iter_mut().zip(group_shares.into_iter()){
            let group_lf_share = LargeField::from_bytes_be(&group_share).unwrap();
            state.push(group_lf_share); // Store the share itself
        }

        depth_state.recv_share_count_l2 +=1;
        // Interpolate polynomial
        // Idempotence satisfied here
        if depth_state.recv_share_count_l2 == self.num_nodes - self.num_faults{
            log::info!("Attempting L2 reconstruction at depth {}", depth);
            // We have enough shares to reconstruct the polynomial
            let indices = depth_state.l2_shares.0.clone();
            let vdm_matrix = vandermonde_matrix(indices);

            let inv_vdm_matrix = inverse_vandermonde(vdm_matrix);
            
            let reconstructed_secrets: Vec<LargeField> = depth_state.l2_shares.1.par_iter().map(|group_shares|{
                let coefficients = matrix_vector_multiply(&inv_vdm_matrix, &group_shares);
                coefficients
            }).flatten().collect();

            depth_state.l2_shares_reconstructed.extend(reconstructed_secrets.clone());
            
            let mut appended_msg = Vec::new();
            for secret in reconstructed_secrets.iter(){
                appended_msg.extend(secret.to_bytes_be());
            }
            let hash = do_hash(&appended_msg);
            log::info!("Completed processing triples at depth {} with linear sharings, broadcasting hash {:?}", depth, hash);
            self.init_hash_broadcast(hash, depth).await;
            self.verify_depth_mult_termination(depth).await;
        }
    }
}