use std::collections::{HashMap, HashSet, VecDeque};

use protocol::LargeField;
use types::Replica;

pub struct MixCircuitState{
    pub rand_bit_inp_shares: Vec<LargeField>,
    pub rand_bit_recon_shares: HashMap<usize, Vec<LargeField>>,

    // Phase-wise (batched) reconstruction state. `rand_bit_recon_pending` holds
    // this party's own shares split into batches; we broadcast them one at a time.
    // `rand_bit_recon_cursor` is the next batch index to broadcast (it
    // fast-forwards past batches already reconstructed). `rand_bit_recon_batches`
    // buffers received shares keyed by batch -> sender, and is cleared per-batch
    // once that batch is reconstructed. `rand_bit_recon_completed` is the set of
    // batch indices already reconstructed (batches complete independently / out of
    // order). `rand_bit_recon_results` holds each batch's reconstructed values,
    // assembled in batch order at the end.
    pub rand_bit_recon_pending: Vec<Vec<LargeField>>,
    pub rand_bit_recon_cursor: usize,
    pub rand_bit_recon_batches: HashMap<usize, HashMap<usize, Vec<LargeField>>>,
    pub rand_bit_recon_completed: HashSet<usize>,
    pub rand_bit_recon_results: HashMap<usize, Vec<LargeField>>,
    // Batch indices for which we have already broadcast our shares, so each
    // ReconstructRandBitShares is sent at most once per batch.
    pub rand_bit_recon_broadcast: HashSet<usize>,

    pub rand_bit_sharings: VecDeque<LargeField>,
    pub rand_bit_reconstruction: HashMap<usize, Vec<LargeField>>,

    pub input_acss_shares: HashMap<Replica, HashMap<usize,Vec<LargeField>>>,
    pub input_sharings: Vec<LargeField>,
    
    // log_^2(k) depths, k wires on each depth
    pub mult_result: HashMap<usize, Vec<LargeField>>,
    pub wire_sharings: HashMap<usize,Vec<LargeField>>,
    
    // k/2 pairs of wires on each depth
    pub wire_pairs: HashMap<usize, Vec<(LargeField, LargeField)>>,
    pub two_inverse: LargeField
}

impl MixCircuitState{
    pub fn new() -> Self {
        MixCircuitState{
            rand_bit_inp_shares: Vec::new(),
            rand_bit_recon_shares: HashMap::new(),

            rand_bit_recon_pending: Vec::new(),
            rand_bit_recon_cursor: 0,
            rand_bit_recon_batches: HashMap::new(),
            rand_bit_recon_completed: HashSet::new(),
            rand_bit_recon_results: HashMap::new(),
            rand_bit_recon_broadcast: HashSet::new(),

            rand_bit_sharings: VecDeque::new(),
            rand_bit_reconstruction: HashMap::default(),
            
            input_acss_shares: HashMap::default(),
            input_sharings: Vec::new(),

            mult_result: HashMap::new(),
            wire_sharings: HashMap::new(),
            wire_pairs: HashMap::new(),

            two_inverse: LargeField::from(2 as u64).inv().unwrap()
        }
    }
}