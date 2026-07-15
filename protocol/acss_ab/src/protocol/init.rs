use std::collections::HashMap;
use std::{ops::{Mul, Add, Sub}};

use crate::Context;
use crypto::{hash::Hash, aes_hash::{HashState, MerkleTree}};
use lambdaworks_math::polynomial::Polynomial;
use protocol::ByteConversion;
use protocol::{LargeField, LargeFieldSer, generate_evaluation_points_fft, generate_evaluation_points, generate_evaluation_points_opt, sample_polynomials_from_prf, rand_field_element};

use types::Replica;

use super::ACSSABState;

/// Deterministic dealer outputs, shared by the CPU and GPU dealer paths.
/// The GPU parity tests compare every field bitwise across the two paths.
pub struct DealerArtifacts {
    pub party_wise_shares: Vec<Vec<LargeFieldSer>>,
    pub nonce_evals_ser: Vec<LargeFieldSer>,
    pub blinding_nonce_evals_ser: Vec<LargeFieldSer>,
    pub commitments: Vec<Hash>,
    pub blinding_commitments: Vec<Hash>,
    pub ser_dzk_coeffs: Vec<LargeFieldSer>,
}

/// Common dealer tail: party-wise share packing, AES commitments, Merkle
/// roots, Fiat-Shamir challenge, and the DZK coefficient combination.
pub fn commitments_and_dzk(
    evaluations: Vec<Vec<LargeField>>,
    coefficients: Vec<Polynomial<LargeField>>,
    nonce_evaluations: Vec<LargeField>,
    blinding_poly_evaluations: Vec<LargeField>,
    blinding_poly_coefficients: Polynomial<LargeField>,
    nonce_blinding_poly_evaluations: Vec<LargeField>,
    num_nodes: usize,
    hash_context: &HashState,
) -> DealerArtifacts {
    // Transform the shares to element wise shares
    let mut party_wise_shares: Vec<Vec<LargeFieldSer>> = Vec::new();
    let mut party_appended_shares: Vec<Vec<u8>> = Vec::new();
    for i in 0..num_nodes{
        let mut party_shares = Vec::new();
        let mut appended_share = Vec::new();
        for j in 0..evaluations.len(){
            party_shares.push(evaluations[j][i].clone().to_bytes_be());
            appended_share.extend(evaluations[j][i].clone().to_bytes_be());
        }
        party_wise_shares.push(party_shares);

        // Append nonce shares to shares for generating commitment
        appended_share.extend(nonce_evaluations[i].clone().to_bytes_be());
        party_appended_shares.push(appended_share);
    }

    // Generate Commitments here
    // There should be $n$ commitments overall
    let commitments: Vec<Hash> = party_appended_shares.into_iter().map(|share| {
        hash_context.do_hash_aes(&share)
    }).collect();

    let merkle_tree = MerkleTree::new(commitments.clone(), hash_context);
    let share_root_comm = merkle_tree.root();

    let mut blinding_commitments = Vec::new();
    for i in 0..num_nodes{
        blinding_commitments.push(hash_context.hash_two(
            blinding_poly_evaluations[i].clone().to_bytes_be().try_into().unwrap(),
            nonce_blinding_poly_evaluations[i].clone().to_bytes_be().try_into().unwrap()));
    }

    let blinding_mt_root = MerkleTree::new(blinding_commitments.clone(), hash_context).root();
    // Generate DZK coefficients

    let root_comm = hash_context.hash_two(share_root_comm, blinding_mt_root);
    // Convert root commitment to field element
    let root_comm_fe = LargeField::from_bytes_be(&root_comm).unwrap();
    log::info!("Root_comm_fe: {:?}", root_comm_fe);

    let mut root_comm_fe_mul = root_comm_fe.clone();
    let mut dzk_coeffs = blinding_poly_coefficients.clone();
    for poly in coefficients.into_iter(){
        dzk_coeffs = dzk_coeffs.add(poly.mul(root_comm_fe_mul.clone()));
        root_comm_fe_mul = root_comm_fe_mul.mul(root_comm_fe.clone());
    }

    let ser_dzk_coeffs: Vec<LargeFieldSer> = dzk_coeffs.coefficients.into_iter().map(|el| el.to_bytes_be()).collect();
    let nonce_evals_ser = nonce_evaluations.into_iter().map(|el| el.to_bytes_be()).collect();
    let blinding_nonce_evals_ser = nonce_blinding_poly_evaluations.into_iter().map(|el| el.to_bytes_be()).collect();

    DealerArtifacts{
        party_wise_shares,
        nonce_evals_ser,
        blinding_nonce_evals_ser,
        commitments,
        blinding_commitments,
        ser_dzk_coeffs,
    }
}

/// Deterministic CPU dealer core for the non-FFT path: given the batch of
/// secrets and the three (normally random) auxiliary secrets, produce the
/// exact byte-level artifacts `init_acss_ab` broadcasts.
pub async fn dealer_core_cpu(
    secrets: Vec<LargeField>,
    nonce_secret: LargeField,
    blinding_secret: LargeField,
    blinding_nonce_secret: LargeField,
    sec_key_map: HashMap<Replica, Vec<u8>>,
    num_faults: usize,
    num_nodes: usize,
    hash_context: &HashState,
) -> DealerArtifacts {
    // Generate evaluations right here
    let evaluations_prf = sample_polynomials_from_prf(
        secrets,
        sec_key_map.clone(),
        num_faults,
        false,
        1u8
    );
    let (evaluations, coefficients): (Vec<Vec<LargeField>>, Vec<Polynomial<LargeField>>) = generate_evaluation_points_opt(
        evaluations_prf,
        num_faults,
        num_nodes,
    ).await;

    // Generate nonce evaluations
    let evaluations_nonce_prf = sample_polynomials_from_prf(
        vec![nonce_secret],
        sec_key_map.clone(),
        num_faults,
        true,
        1u8
    );
    let (nonce_evaluations_ret,_nonce_coefficients) = generate_evaluation_points(
        evaluations_nonce_prf,
        num_faults,
        num_nodes
    ).await;
    let nonce_evaluations = nonce_evaluations_ret[0].clone();

    // Generate the DZK proofs and commitments and utilize RBC to broadcast these proofs
    // Sample blinding polynomial
    let blinding_prf = sample_polynomials_from_prf(
        vec![blinding_secret],
        sec_key_map.clone(),
        num_faults,
        true,
        2u8
    );
    let (blinding_poly_evaluations_vec, blinding_poly_coefficients_vec) = generate_evaluation_points(
        blinding_prf,
        num_faults,
        num_nodes
    ).await;

    let blinding_poly_evaluations = blinding_poly_evaluations_vec[0].clone();
    let blinding_poly_coefficients = blinding_poly_coefficients_vec[0].clone();

    let blinding_nonce_prf = sample_polynomials_from_prf(
        vec![blinding_nonce_secret],
        sec_key_map.clone(),
        num_faults,
        true,
        3u8
    );

    let (nonce_blinding_poly_evaluations_vec, _nonce_blinding_poly_coefficients_vec) = generate_evaluation_points(
        blinding_nonce_prf,
        num_faults,
        num_nodes,
    ).await;
    let nonce_blinding_poly_evaluations = nonce_blinding_poly_evaluations_vec[0].clone();

    commitments_and_dzk(
        evaluations,
        coefficients,
        nonce_evaluations,
        blinding_poly_evaluations,
        blinding_poly_coefficients,
        nonce_blinding_poly_evaluations,
        num_nodes,
        hash_context,
    )
}

impl Context{
    pub async fn init_acss_ab(&mut self, secrets: Vec<LargeField>, instance_id: usize){
        if !self.acss_ab_state.contains_key(&instance_id){
            let acss_ab_state = ACSSABState::new();
            self.acss_ab_state.insert(instance_id, acss_ab_state);
        }

        let tot_sharings = secrets.len();

        // Own-dealer spans: total ACSS latency and the local dealer-compute
        // portion (interpolation/eval + commitments + serialization), keyed by
        // (instance_id, myid). Closed in broadcast_dealer_artifacts and
        // check_termination respectively.
        self.profiler.start("ACSS_total", instance_id, self.myid);
        self.profiler.start("ACSS_dealer_compute", instance_id, self.myid);

        #[cfg(feature = "gpu")]
        if !self.use_fft {
            return self.init_acss_ab_gpu(secrets, instance_id).await;
        }

        let artifacts;
        if !self.use_fft{
            artifacts = dealer_core_cpu(
                secrets,
                rand_field_element(),
                rand_field_element(),
                rand_field_element(),
                self.sec_key_map.clone(),
                self.num_faults,
                self.num_nodes,
                &self.hash_context,
            ).await;
        }
        else{
            // FFT branch: polynomial coefficients are sampled internally by
            // generate_evaluation_points_fft, so this path stays CPU-only.
            let (evaluations, coefficients) = generate_evaluation_points_fft(
                secrets,
                self.num_faults-1,
                self.num_nodes
            ).await;

            // Generate nonce evaluations
            let (nonce_evaluations_ret,_nonce_coefficients) = generate_evaluation_points_fft(
                vec![rand_field_element()],
                self.num_faults-1,
                self.num_nodes,
            ).await;
            let nonce_evaluations = nonce_evaluations_ret[0].clone();

            let (blinding_poly_evaluations_vec, blinding_poly_coefficients_vec) = generate_evaluation_points_fft(vec![rand_field_element()],
                self.num_faults-1,
                self.num_nodes
            ).await;
            let blinding_poly_evaluations = blinding_poly_evaluations_vec[0].clone();
            let blinding_poly_coefficients = blinding_poly_coefficients_vec[0].clone();

            let (nonce_blinding_evaluations_vec, _nonce_coefficients_vec) = generate_evaluation_points_fft(vec![rand_field_element()]
                ,
                self.num_faults-1,
                self.num_nodes
            ).await;
            let nonce_blinding_poly_evaluations = nonce_blinding_evaluations_vec[0].clone();

            artifacts = commitments_and_dzk(
                evaluations,
                coefficients,
                nonce_evaluations,
                blinding_poly_evaluations,
                blinding_poly_coefficients,
                nonce_blinding_poly_evaluations,
                self.num_nodes,
                &self.hash_context,
            );
        }

        self.broadcast_dealer_artifacts(instance_id, tot_sharings, artifacts).await;
    }

    /// Serialize dealer artifacts and disperse them: CTRBC for the
    /// commitments + DZK polynomial, AVID for the per-party shares.
    /// Shared tail of the CPU and GPU dealer paths.
    pub async fn broadcast_dealer_artifacts(&mut self, instance_id: usize, tot_sharings: usize, artifacts: DealerArtifacts){
        let DealerArtifacts{
            party_wise_shares,
            nonce_evals_ser,
            blinding_nonce_evals_ser,
            commitments,
            blinding_commitments,
            ser_dzk_coeffs,
        } = artifacts;

        let broadcast_vec = (commitments, blinding_commitments, ser_dzk_coeffs, tot_sharings);
        let serialized_broadcase_vec = (instance_id, broadcast_vec);
        let ser_vec = bincode::serialize(&serialized_broadcase_vec).unwrap();

        let mut shares: Vec<(Replica,Option<Vec<u8>>)> = Vec::new();
        for rep in 0..self.num_nodes{
            // prepare shares
            // even need to encrypt shares
            if (self.use_fft) || (!self.use_fft && rep >= self.num_faults){
                let shares_party = party_wise_shares[rep].clone();
                let nonce_share = nonce_evals_ser[rep].clone();
                let blinding_nonce_share = blinding_nonce_evals_ser[rep].clone();

                let shares_full = (shares_party, nonce_share, blinding_nonce_share);
                let serialized_shares = (instance_id, shares_full);
                let shares_ser = bincode::serialize(&serialized_shares).unwrap();

                //let enc_shares = encrypt(sec_key.as_slice(), shares_ser);
                shares.push((rep, Some(shares_ser)));
            }
        }
        // Dealer compute finished; the two dispersal building blocks begin.
        self.profiler.stop("ACSS_dealer_compute", instance_id, self.myid);
        self.profiler.start("CTRBC", instance_id, self.myid);
        self.profiler.start("AVID", instance_id, self.myid);

        // Reliably broadcast this vector
        let _rbc_status = self.inp_ctrbc.send(ser_vec).await;

        // Invoke AVID on vectors of shares
        // Use AVID to send the shares to parties
        let _avid_status = self.inp_avid_channel.send(shares).await;
    }

    pub async fn verify_shares(&mut self, sender: Replica, instance_id: usize){
        log::info!("Verifying shares from sender: {} for instance: {}", sender, instance_id);
        if !self.acss_ab_state.contains_key(&instance_id){
            let acss_state = ACSSABState::new();
            self.acss_ab_state.insert(instance_id, acss_state);
        }

        let acss_ab_state = self.acss_ab_state.get_mut(&instance_id).unwrap();
        if acss_ab_state.acss_status.contains(&sender){
            // Dealer already terminated and its state was cleared; ignore.
            return;
        }
        if acss_ab_state.verification_status.contains_key(&sender){
            // Already verified status, abandon sharing
            return;
        }

        if !acss_ab_state.commitments.contains_key(&sender) || !acss_ab_state.shares.contains_key(&sender){
            // AVID and CTRBC did not yet terminate
            return;
        }

        let shares_full = acss_ab_state.shares.get(&sender).unwrap().clone();
        let shares = shares_full.0;
        let nonce_share = shares_full.1;
        let blinding_nonce_share = shares_full.2;

        let commitments_full = acss_ab_state.commitments.get(&sender).unwrap().clone();
        let share_commitments = commitments_full.0;
        let blinding_commitments = commitments_full.1;
        let dzk_coeffs = commitments_full.2;
        let blinding_comm_sender = blinding_commitments[self.myid].clone();

        // First, verify share commitments
        let mut appended_share = Vec::new();
        for share in shares.clone().into_iter(){
            appended_share.extend(share);
        }
        appended_share.extend(nonce_share);
        let comm_hash = self.hash_context.do_hash_aes(appended_share.as_slice());
        if comm_hash != share_commitments[self.myid]{
            // Invalid share commitments
            log::error!("Invalid share commitments from {}", sender);
            acss_ab_state.verification_status.insert(sender, false);
            return;
        }

        // Second, verify DZK proof
        let shares_ff: Vec<LargeField> = shares.into_iter().map(|el| LargeField::from_bytes_be(el.as_slice()).unwrap()).collect();
        let dzk_poly_coeffs: Vec<LargeField> = dzk_coeffs.into_iter().map(|el| LargeField::from_bytes_be(el.as_slice()).unwrap()).collect();
        let dzk_poly = Polynomial::new(dzk_poly_coeffs.as_slice());
        // Change this to be root of unity

        let share_root = MerkleTree::new(share_commitments, &self.hash_context).root();
        let blinding_root = MerkleTree::new(blinding_commitments, &self.hash_context).root();
        let root_comm = self.hash_context.hash_two(share_root, blinding_root);
        let root_comm_fe = LargeField::from_bytes_be(&root_comm).unwrap();

        log::info!("Root_comm_fe: {:?} for sender {} instance_id {}",root_comm_fe, sender, instance_id);
        let verf_status = self.evaluate_dzk_poly(
            root_comm_fe, 
            self.myid, 
            &dzk_poly, 
            &shares_ff, 
            blinding_comm_sender, 
            blinding_nonce_share
        );
        let acss_ab_state = self.acss_ab_state.get_mut(&instance_id).unwrap();
        if !verf_status{
            // Invalid DZK proof
            log::error!("Invalid DZK proof from {}", sender);
            acss_ab_state.verification_status.insert(sender,false);
            return;
        }
        log::info!("Share from {} verified", sender);
        // If successful, add to verified list
        // Reborrow share
        
        acss_ab_state.verification_status.insert(sender,true);
        // Start reliable agreement (RA span closes in handle_ra_termination).
        self.profiler.start("RA", instance_id, sender);
        let _status = self.inp_ra_channel.send((sender,1,instance_id)).await;
        self.check_termination(sender, instance_id).await;
    }

    pub fn evaluate_dzk_poly(
        &self,
        root_comm_fe: LargeField,
        share_sender: Replica,
        dzk_poly: &Polynomial<LargeField>, 
        shares: &Vec<LargeField>, 
        blinding_comm: Hash,
        blinding_nonce: LargeFieldSer,
    )-> bool{
        // Change this to be root of unity
        let dzk_point;
        if !self.use_fft{
            dzk_point = dzk_poly.evaluate(&LargeField::from((share_sender+1) as u64));
        }
        else{
            // get point of evaluation
            let eval_point = self.roots_of_unity[share_sender].clone();
            dzk_point = dzk_poly.evaluate(&eval_point);
        }
        let mut agg_shares_point = LargeField::zero();
        let mut root_comm_fe_mul = root_comm_fe.clone();
        for share in shares{
            agg_shares_point = agg_shares_point.add(share.mul(root_comm_fe_mul.clone()));
            root_comm_fe_mul = root_comm_fe_mul.mul(root_comm_fe.clone());
        }

        let blinding_poly_share_bytes = dzk_point.sub(agg_shares_point).to_bytes_be();
        let blinding_hash = self.hash_context.hash_two(
            blinding_poly_share_bytes.try_into().unwrap(),
            blinding_nonce.try_into().unwrap()
        );
        if blinding_hash != blinding_comm{
            log::info!("Blinding hash: {:?}, blinding_comm: {:?}", blinding_hash, blinding_comm);
            // Invalid DZK proof
            return false;
        }
        true
    }
}