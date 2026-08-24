use std::{collections::HashMap, ops::{Add, Mul}};

use protocol::ByteConversion;
use protocol::{LargeField, LargeFieldSer, rand_field_element};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use types::{ProtSyncMsg, SyncMsg, SyncState};
use crate::{context::Context};

impl Context{
    pub async fn init_rand_sh(&mut self){
        let num_batches = self.tot_batches;
        let batch_size = self.total_sharings;

        // Three degree-t batches, all feeding the multiplication randomness pool.
        // (Batch 0 previously fed the mixing circuit's random-bit preparation,
        // which the NN has no use for.) Each batch is built and shipped one at a
        // time: holding all three at once doubles the preprocessing footprint for
        // no reason.
        for index in 0..3*num_batches{
            let batch: Vec<LargeFieldSer> = (0..batch_size)
                .into_par_iter()
                .map(|_| rand_field_element().to_bytes_be())
                .collect();
            log::info!("Initiating secret sharing in preprocessing phase for batch {}", index);
            let status = self.acss_ab_send.send((index,batch)).await;
            if status.is_err(){
                log::error!("Failed to send random values to ACSS protocol for batch {} because of error: {:?}", index, status.err().unwrap());
            }
        }

        for index in 0..3*num_batches{
            let batch: Vec<LargeFieldSer> = (0..batch_size)
                .into_par_iter()
                .map(|_| LargeField::zero().to_bytes_be())
                .collect();
            log::info!("Initiating 2t sharing in preprocessing phase for batch {}", index);
            let status = self.sh2t_send.send((index,batch)).await;
            if status.is_err(){
                log::error!("Failed to send random values to Sh2t protocol for batch {} because of error: {:?}", index, status.err().unwrap());
            }
        }

        // Random masks for output wires
        let random_masks: Vec<LargeFieldSer> = (0..self.output_mask_size)
            .into_par_iter()
            .map(|_| rand_field_element().to_bytes_be())
            .collect();
        let avss_status = self.avss_send.send((true, Some(random_masks), None)).await;
        if avss_status.is_err(){
            log::error!("Failed to send random values to AVSS protocol {:?}", avss_status.err().unwrap());
        }
    }

    pub async fn handle_acss_term_msg(&mut self, instance: usize, sender: usize, shares: Option<Vec<LargeFieldSer>>){
        log::info!("Received ACSS shares from sender {} for batch {}", sender, instance);
        if shares.is_none(){
            log::error!("Abort ACSS protocol of dealer {} and terminate MPC", sender);
            return;
        }
        
        if self.rand_sharings_state.rand_sharings_mult.len() > 0{
            log::info!("Finished processing random sharings, ignoring ACSS and SH2t for all subsequent batches and senders: sender {}", sender);
            return;
        }

        let shares_deser: Vec<LargeField> = shares.unwrap().into_par_iter().map(|x| 
            LargeField::from_bytes_be(&x).unwrap()
        ).collect();

        if !self.rand_sharings_state.shares.contains_key(&sender){
            self.rand_sharings_state.shares.insert(sender, HashMap::default());
        }

        let shares_batches_map = self.rand_sharings_state.shares.get_mut(&sender).unwrap();
        shares_batches_map.insert(instance, shares_deser);

        self.verify_sender_termination(sender).await;
    }

    pub async fn handle_sh2t_term_msg(&mut self, instance: usize, sender: usize, shares: Option<Vec<LargeFieldSer>>){
        log::info!("Received Sh2t shares from sender {} for batch {}", sender, instance);
        if shares.is_none(){
            log::error!("Abort 2t-sharing protocol of dealer {} and terminate MPC", sender);
            return;
        }

        if self.rand_sharings_state.rand_sharings_mult.len() > 0{
            log::info!("Finished processing random sharings, ignoring ACSS and SH2t for all subsequent batches and senders: sender {}", sender);
            return;
        }
        let shares_deser: Vec<LargeField> = shares.unwrap().into_par_iter().map(|x| 
            LargeField::from_bytes_be(&x).unwrap()
        ).collect();

        if !self.rand_sharings_state.sh2t_shares.contains_key(&sender){
            self.rand_sharings_state.sh2t_shares.insert(sender, HashMap::default());
        }

        let shares_batches_map = self.rand_sharings_state.sh2t_shares.get_mut(&sender).unwrap();
        shares_batches_map.insert(instance, shares_deser);

        self.verify_sender_termination(sender).await;
    }

    pub async fn verify_sender_termination(&mut self, sender: usize){
        if !self.rand_sharings_state.shares.contains_key(&sender) || !self.rand_sharings_state.sh2t_shares.contains_key(&sender) || !self.output_mask_state.avss_shares.contains_key(&sender){
            log::debug!("ACSS, Sh2t, and AVSS not completed for sender {} for all batches", sender);
            return;
        }
        if self.rand_sharings_state.acss_completed_parties.contains(&sender){
            log::debug!("ACSS, Sh2t, and AVSS already completed for sender {} for all batches", sender);
            return;
        }
        let shares_batches_map = self.rand_sharings_state.shares.get_mut(&sender).unwrap();
        let share_2t_batches_map = self.rand_sharings_state.sh2t_shares.get_mut(&sender).unwrap();
        if shares_batches_map.len() == (3*self.tot_batches) &&
            share_2t_batches_map.len() == 3*self.tot_batches &&
            self.output_mask_state.avss_shares.contains_key(&sender){
            // ACSS is complete. Wait for sh2t sharings now
            log::info!("ACSS, Sh2t, and AVSS completed for sender {} for all batches", sender);
            log::info!("Batches info: {:?} {:?}", shares_batches_map.keys(),share_2t_batches_map.keys());
            self.rand_sharings_state.acss_completed_parties.insert(sender);

            if self.rand_sharings_state.acss_completed_parties.len() == self.num_nodes-self.num_faults{
                let coins: Vec<consensus::LargeFieldSer> = (0..crate::context::NUM_CONSENSUS_COINS).into_iter().map(|x| consensus::LargeField::from(x as u64).to_bytes_be()).collect();
                let parties_set: Vec<usize> = self.rand_sharings_state.acss_completed_parties.clone().into_iter().collect();
                let ser_set = bincode::serialize(&parties_set).unwrap();
                let _status = self.acs_event_send.send((1, ser_set, coins)).await;
            }
            self.verify_termination().await;
        }
    }

    pub async fn handle_acs_output(&mut self, partyset: Vec<u8>){
        let deser_set: Vec<usize> = bincode::deserialize(&partyset).unwrap();
        self.rand_sharings_state.acs_output.extend(deser_set);
        // Check if all parties have completed ACSS and 2t-sharing
        self.verify_termination().await;
    }

    pub async fn verify_termination(&mut self){
        log::info!("Checking termination for random sharings");
        if self.rand_sharings_state.rand_sharings_mult.len() > 0{
            // Sharings already generated, return back
            return;
        } 
        if self.rand_sharings_state.acs_output.len() > 0{
            let mut flag = true;
            for party in self.rand_sharings_state.acs_output.clone().into_iter(){
                flag =  flag && self.rand_sharings_state.acss_completed_parties.contains(&party);
            }
            if flag{
                // All parties in the ACS state have completed ACSS and 2t-sharing
                // Generate random sharings
                // Vandermonde matrix
                
                let x_values: Vec<LargeField> = (2..self.num_faults+3).into_iter().map(|x| LargeField::from(x as u64)).collect();
                let vandermonde_matrix = Self::vandermonde_matrix(x_values, 2*self.num_faults+1);

                // Extract each degree-t batch and fold it into the pool one at a
                // time, dropping the intermediate group vectors between batches.
                let mut total_extracted = 0usize;
                for batch_index in 0..3{
                    let groups = self.gen_random_sharings(batch_index*self.tot_batches);
                    let extracted: Vec<LargeField> = groups.into_par_iter().map(|x| {
                        Self::matrix_vector_multiply(&vandermonde_matrix, &x)
                    }).flatten().collect();
                    total_extracted += extracted.len();
                    self.rand_sharings_state.rand_sharings_mult.extend(extracted);
                }

                let acs_indexed_2t_share_groups = self.gen_2t_sharings();
                let rand_sharings_2t_mult: Vec<LargeField> = acs_indexed_2t_share_groups.into_par_iter().map(|x| {
                    Self::matrix_vector_multiply(&vandermonde_matrix, &x)
                }).flatten().collect();

                log::info!("Completed preprocessing and generated {} random sharings and {} random 2t sharings",
                        total_extracted, rand_sharings_2t_mult.len());

                // Set aside sharings for the verification phase's common coins.
                let coin_count = self.total_sharings_for_coins.min(self.rand_sharings_state.rand_sharings_mult.len());
                let rand_sharings_coin: Vec<LargeField> = self.rand_sharings_state.rand_sharings_mult
                    .drain(self.rand_sharings_state.rand_sharings_mult.len()-coin_count..)
                    .collect();
                self.rand_sharings_state.rand_sharings_coin.extend(rand_sharings_coin);
                self.rand_sharings_state.rand_2t_sharings_mult.extend(rand_sharings_2t_mult);

                // Release the raw ACSS/Sh2t shares — they are the single largest
                // preprocessing allocation and are dead once extraction is done.
                self.rand_sharings_state.shares.clear();
                self.rand_sharings_state.shares.shrink_to_fit();
                self.rand_sharings_state.sh2t_shares.clear();
                self.rand_sharings_state.sh2t_shares.shrink_to_fit();

                self.generate_random_mask_shares(self.rand_sharings_state.acs_output.clone(),vandermonde_matrix).await;
                self.terminate("Preprocessing".to_string(), vec![]).await;
                // Weights and input activations are shared in the input phase,
                // which gates the online phase on *all* n dealers.
                self.init_input_phase().await;
            }
        }
    }

    /// Constructs the Vandermonde matrix for a given set of x-values. Note that the x-values are parties and are converted to the ith root of unity for the evaluation
    pub fn vandermonde_matrix(x_values: Vec<LargeField>, y_vals_target: usize) -> Vec<Vec<LargeField>> {
        let n = x_values.len();
        let mut matrix = vec![vec![LargeField::zero(); y_vals_target]; n];

        for (row, x) in x_values.iter().enumerate() {
            let mut value = LargeField::one();
            for col in 0..y_vals_target {
                matrix[row][col] = value.clone();
                value = value * x;
            }
        }
        matrix
    }

    pub fn matrix_vector_multiply(
        matrix: &[Vec<LargeField>],
        vector: &[LargeField],
    ) -> Vec<LargeField> {
        matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(vector)
                    .fold(LargeField::zero(), |sum, (a, b)| sum.add(a.mul(b)))
            })
            .collect()
    }

    pub fn gen_random_sharings(&self, offset: usize)-> Vec<Vec<LargeField>>{
        let mut acs_indexed_share_groups: Vec<Vec<LargeField>> = Vec::new();
        
        (0..self.tot_batches*self.total_sharings).into_iter().for_each(|_|{
            acs_indexed_share_groups.push(Vec::new());
        });
        for party in 0..self.num_nodes{
            if self.rand_sharings_state.acs_output.contains(&party){
                // First sharing
                let shares = self.rand_sharings_state.shares.get(&party).unwrap();
                let mut index: usize = 0;
                for batch in offset..(self.tot_batches+offset){
                    if !shares.contains_key(&batch){
                        log::error!("Batch {} not found in shares_batch", batch);
                    }
                    else{
                        let shares_batch = shares.get(&batch).unwrap();
                        for share in shares_batch{
                            acs_indexed_share_groups[index].push(share.clone());
                            index += 1;
                        }
                    }
                }       
            }
        }
        acs_indexed_share_groups
    }

    pub fn gen_2t_sharings(&self) -> Vec<Vec<LargeField>>{
        let mut acs_indexed_2t_share_groups: Vec<Vec<LargeField>> = Vec::new();
        (0..3*self.tot_batches*self.total_sharings).into_iter().for_each(|_|{
            acs_indexed_2t_share_groups.push(Vec::new());
        });
        for party in 0..self.num_nodes{
            if self.rand_sharings_state.acs_output.contains(&party){
                // Sh2t sharing
                let shares_2t = self.rand_sharings_state.sh2t_shares.get(&party).unwrap();
                let mut index = 0;
                // All 3*tot_batches Sh2t batches are consumed. The previous bound
                // of `tot_batches` left two thirds of the groups empty, and an
                // empty group folds to *exactly zero* through the Vandermonde
                // multiply below — producing degree-2t "masks" of zero, which
                // mask nothing in the DN reduction.
                for batch in 0..3*self.tot_batches{
                    if !shares_2t.contains_key(&batch){
                        log::error!("Batch {} not found in shares_batch for 2t shares", batch);
                    }
                    else{
                        let shares_batch = shares_2t.get(&batch).unwrap();
                        for share in shares_batch{
                            acs_indexed_2t_share_groups[index].push(share.clone());
                            index += 1;
                        }
                    }
                }
            }
        }
        acs_indexed_2t_share_groups
    }
    //Invoke this function once you terminate the protocol
    pub async fn terminate(&mut self, status: String, value: Vec<u8>) {
        let rbc_sync_msg = ProtSyncMsg{
            id: 1,
            status,
            value
        };

        let ser_msg = bincode::serialize(&rbc_sync_msg).unwrap();
        let cancel_handler = self
            .sync_send
            .send(
                0,
                SyncMsg {
                    sender: self.myid,
                    state: SyncState::COMPLETED,
                    value: ser_msg,
                },
            )
            .await;
        self.add_cancel_handler(cancel_handler);
    }
}