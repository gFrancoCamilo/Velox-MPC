mod types;
pub use types::*;

pub mod byte_conv;
pub use byte_conv::ByteConversion;

pub mod mersenne_61;

pub mod poly;
pub use poly::*;

#[cfg(feature = "gpu")]
pub mod gpu_ffi;
#[cfg(feature = "gpu")]
pub use gpu_ffi::{gpu_matrix_matrix_multiply, read_elem_bytes, write_elem_bytes, PreparedFieldGemm};

#[cfg(feature = "gpu")]
pub mod gpu_mem;
#[cfg(feature = "gpu")]
pub mod gpu_acss_ffi;
#[cfg(feature = "gpu")]
pub mod gpu_aes_ffi;