//! RAII wrappers around raw CUDA allocations, streams, and pinned host
//! memory, used by the GPU ACSS dealer path to keep persistent scratch
//! buffers alive across protocol instances.
//!
//! Compiled only with `--features gpu`. Vendored from
//! async_mpc/secret_sharing/acss_ab/src/protocol/acss_kernels_ffi.rs.

extern "C" {
    fn cudaMalloc(dev_ptr: *mut *mut u8, size: usize) -> i32;
    fn cudaFree(dev_ptr: *mut u8) -> i32;
    fn cudaMallocHost(ptr: *mut *mut u8, size: usize) -> i32;
    fn cudaFreeHost(ptr: *mut u8) -> i32;
    fn cudaMemcpy(dst: *mut u8, src: *const u8, count: usize, kind: i32) -> i32;
    fn cudaMemcpy2D(
        dst: *mut u8,
        dpitch: usize,
        src: *const u8,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
    ) -> i32;
    fn cudaStreamCreate(stream: *mut *mut std::ffi::c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut u8,
        src: *const u8,
        count: usize,
        kind: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;

/// Lazily-grown CUDA device memory buffer.
pub struct DeviceBuffer {
    ptr: *mut u8,
    capacity: usize,
}

impl DeviceBuffer {
    pub const fn new() -> Self {
        DeviceBuffer {
            ptr: std::ptr::null_mut(),
            capacity: 0,
        }
    }

    /// Ensure the buffer holds at least `needed` bytes.
    /// Reallocates (and discards old content) only when capacity is exceeded.
    pub fn ensure(&mut self, needed: usize) -> Result<(), String> {
        if needed <= self.capacity {
            return Ok(());
        }
        // Free old allocation
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
            self.capacity = 0;
        }
        // Allocate new
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let rc = unsafe { cudaMalloc(&mut ptr as *mut *mut u8, needed) };
        if rc != 0 || ptr.is_null() {
            return Err(format!("cudaMalloc({} bytes) failed, rc={}", needed, rc));
        }
        self.ptr = ptr;
        self.capacity = needed;
        Ok(())
    }

    /// Raw device pointer. Null if nothing has been allocated yet.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Raw mutable device pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Copy `data` from host into this buffer (H2D). Grows buffer if needed.
    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<(), String> {
        self.ensure(data.len())?;
        let rc = unsafe { cudaMemcpy(self.ptr, data.as_ptr(), data.len(), CUDA_MEMCPY_H2D) };
        if rc != 0 {
            return Err(format!("cudaMemcpy H2D failed, rc={}", rc));
        }
        Ok(())
    }

    /// Copy `data` from host into this buffer asynchronously (H2D). Grows buffer if needed.
    /// Returns immediately; the transfer completes after `stream.synchronize()`.
    pub fn copy_from_host_async(&mut self, data: &[u8], stream: &CudaStream) -> Result<(), String> {
        self.ensure(data.len())?;
        let rc = unsafe {
            cudaMemcpyAsync(self.ptr, data.as_ptr(), data.len(), CUDA_MEMCPY_H2D, stream.raw())
        };
        if rc != 0 {
            return Err(format!("cudaMemcpyAsync H2D failed, rc={}", rc));
        }
        Ok(())
    }

    /// Copy `count` bytes from this device buffer into `dst` (D2H).
    /// `dst` must be at least `count` bytes long.
    pub fn copy_to_host(&self, dst: &mut [u8], count: usize) -> Result<(), String> {
        assert!(dst.len() >= count, "destination slice too small");
        assert!(self.capacity >= count, "device buffer too small");
        let rc = unsafe {
            cudaMemcpy(
                dst.as_mut_ptr(),
                self.ptr as *const u8,
                count,
                CUDA_MEMCPY_D2H,
            )
        };
        if rc != 0 {
            return Err(format!("cudaMemcpy D2H failed, rc={}", rc));
        }
        Ok(())
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr);
            }
        }
    }
}

// SAFETY: the pointer is only valid on the CUDA device and is never
// dereferenced from the host — safe to move between threads.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

/// RAII wrapper around a CUDA stream handle.
pub struct CudaStream {
    handle: *mut std::ffi::c_void,
}

impl CudaStream {
    pub fn create() -> Result<Self, String> {
        let mut handle = std::ptr::null_mut();
        let rc = unsafe { cudaStreamCreate(&mut handle) };
        if rc != 0 {
            return Err(format!("cudaStreamCreate failed, rc={}", rc));
        }
        Ok(Self { handle })
    }

    pub fn synchronize(&self) -> Result<(), String> {
        let rc = unsafe { cudaStreamSynchronize(self.handle) };
        if rc != 0 {
            return Err(format!("cudaStreamSynchronize failed, rc={}", rc));
        }
        Ok(())
    }

    pub fn raw(&self) -> *mut std::ffi::c_void {
        self.handle
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                cudaStreamDestroy(self.handle);
            }
        }
    }
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

/// Page-locked byte buffer for large CUDA D2H transfers.
pub struct PinnedHostBuffer {
    ptr: *mut u8,
    len: usize,
}

impl PinnedHostBuffer {
    pub fn alloc(len: usize) -> Result<Self, String> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                len,
            });
        }
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let rc = unsafe { cudaMallocHost(&mut ptr as *mut *mut u8, len) };
        if rc != 0 || ptr.is_null() {
            return Err(format!("cudaMallocHost({} bytes) failed, rc={}", len, rc));
        }
        Ok(Self { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        if self.len != 0 {
            unsafe {
                cudaFreeHost(self.ptr);
            }
        }
    }
}

unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

/// Copy `count` bytes from an arbitrary device pointer `d_src` into host slice `dst`.
///
/// # Safety
/// `d_src` must be a valid CUDA device pointer pointing to at least `count` bytes.
pub unsafe fn device_to_host(d_src: *const u8, dst: &mut [u8], count: usize) -> Result<(), String> {
    assert!(dst.len() >= count, "destination buffer too small");
    let rc = cudaMemcpy(dst.as_mut_ptr(), d_src, count, CUDA_MEMCPY_D2H);
    if rc != 0 {
        Err(format!("cudaMemcpy D2H failed, rc={}", rc))
    } else {
        Ok(())
    }
}

/// Copy `rows` rows from a pitched device buffer into a compact host buffer.
///
/// # Safety
/// `d_src` must point to at least `rows` rows of `src_pitch` bytes.
pub unsafe fn copy_rows_to_host(
    d_src: *const u8,
    src_pitch: usize,
    dst: &mut [u8],
    row_bytes: usize,
    rows: usize,
) -> Result<(), String> {
    assert!(
        dst.len() >= row_bytes * rows,
        "destination buffer too small"
    );
    let rc = cudaMemcpy2D(
        dst.as_mut_ptr(),
        row_bytes,
        d_src,
        src_pitch,
        row_bytes,
        rows,
        CUDA_MEMCPY_D2H,
    );
    if rc != 0 {
        Err(format!("cudaMemcpy2D D2H failed, rc={}", rc))
    } else {
        Ok(())
    }
}
