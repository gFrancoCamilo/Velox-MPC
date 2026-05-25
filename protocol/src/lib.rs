mod types;
pub use types::*;

pub mod poly;
pub use poly::*;

#[cfg(feature = "gpu")]
pub mod gpu_ffi;
#[cfg(feature = "gpu")]
pub use gpu_ffi::gpu_matrix_matrix_multiply;