use std::collections::{HashMap, HashSet};

use crypto::{hash::Hash};
use types::Replica;

use crate::msg::{AVIDMsg, AVIDShard};

pub struct AVIDState{
    pub sender: usize,
    
    pub fragments: Option<AVIDMsg>,
    // Only for the recipient
    // deliveries tracked by the root Hash value
    
    pub deliveries: HashMap<Hash,HashMap<Replica,AVIDShard>>,
    pub message: Option<Vec<u8>>,

    pub echos: (HashMap<Hash, HashSet<usize>>, HashMap<Hash, HashSet<usize>>),
    // root Hash followed by all other composing hashes
    pub agreed_root: Option<Hash>,

    pub readys: HashMap<Hash, HashSet<usize>>,

    pub terminated: bool
}

impl AVIDState{
    
    pub fn new(sender: Replica)-> AVIDState{
        AVIDState {
            sender: sender,

            fragments: None,
            message: None,
            deliveries: HashMap::default(),

            echos: (HashMap::default(), HashMap::default()),
            agreed_root: None,

            readys: HashMap::default(),

            terminated:false
        }
    }

    /// Release all memory-heavy buffers held for this instance once it has
    /// terminated. The `sender` and `terminated` flag are retained so that late
    /// duplicate Init/Echo/Ready messages for the same instance are still
    /// recognized and short-circuited instead of resurrecting the state.
    pub fn clear_state(&mut self){
        self.fragments = None;
        self.message = None;
        self.deliveries = HashMap::default();
        self.echos = (HashMap::default(), HashMap::default());
        self.agreed_root = None;
        self.readys = HashMap::default();
    }
}