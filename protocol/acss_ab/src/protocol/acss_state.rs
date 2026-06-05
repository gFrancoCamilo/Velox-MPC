use std::collections::{HashMap, HashSet};

use crypto::hash::Hash;
use lambdaworks_math::polynomial::Polynomial;
use protocol::{LargeField, AvssShare, LargeFieldSer};
use types::Replica;

#[derive(Clone, Debug)]
pub struct ACSSABState{
    // Shares, Nonce, Blinding nonce share in each tuple
    pub shares: HashMap<Replica, AvssShare>,
    // Commitments to shares, commitments to blinding polynomial, and DZK polynomial
    pub commitments: HashMap<Replica, (Vec<Hash>, Vec<Hash>, Vec<LargeFieldSer>)>,
    // Reliable Agreement
    pub ra_outputs: HashSet<Replica>,
    // Verification status for each party
    pub verification_status: HashMap<Replica, bool>,
    pub acss_status: HashSet<Replica>,

    pub dzk_poly: HashMap<Replica,Polynomial<LargeField>>,
    pub commitment_root_fe: HashMap<Replica, LargeField>,
}

impl ACSSABState{
    pub fn new() -> Self{
        Self{
            shares: HashMap::default(),
            commitments: HashMap::default(),
            ra_outputs: HashSet::default(),
            verification_status: HashMap::default(),
            acss_status: HashSet::default(),

            dzk_poly: HashMap::default(),
            commitment_root_fe: HashMap::default(),
        }
    }

    /// Drop all per-dealer state held for `dealer` once its sharing has
    /// terminated and the output has been delivered. `acss_status` is retained
    /// as a tombstone so that late/duplicate CTRBC/AVID/RA messages for the same
    /// dealer are short-circuited instead of resurrecting the cleared buffers.
    ///
    /// NOTE: This must not be called for the AVSS instance, whose
    /// `commitments`, `dzk_poly` and `commitment_root_fe` are deliberately kept
    /// alive to serve later share-validity reconstruction requests.
    pub fn clear_dealer_state(&mut self, dealer: &Replica){
        self.shares.remove(dealer);
        self.commitments.remove(dealer);
        self.verification_status.remove(dealer);
        self.ra_outputs.remove(dealer);
        self.dzk_poly.remove(dealer);
        self.commitment_root_fe.remove(dealer);
    }
}