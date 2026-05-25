//! Optional CUDA build for the `gpu` feature.
//!
//! With the feature off (the default), this is a no-op so non-CUDA hosts —
//! including macOS — build the crate cleanly.
//!
//! With the feature on, this compiles every `.cu` file under `protocol/cuda/`
//! into a static archive (`libbn254_gemm_cuda.a`) and links it plus the CUDA
//! runtime (`cudart`) into the crate. Mirrors `async_mpc/fields/build.rs`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if env::var_os("CARGO_FEATURE_GPU").is_none() {
        // Default path on any host: do nothing. The protocol crate compiles
        // pure-Rust and the `gpu_ffi` module is gated out at compile time.
        return;
    }

    // ----- GPU feature enabled: invoke nvcc -----

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let cuda_home = PathBuf::from(cuda_home);
    let nvcc = cuda_home.join("bin").join("nvcc");
    if !nvcc.exists() {
        panic!(
            "nvcc not found at {} — install the CUDA Toolkit or set $CUDA_HOME",
            nvcc.display()
        );
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set by cargo"));
    let cuda_src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("cuda");

    // Compile every .cu in cuda/ into a single static archive.
    let mut object_files: Vec<PathBuf> = Vec::new();
    let entries = std::fs::read_dir(&cuda_src_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", cuda_src_dir.display(), e));
    for entry in entries {
        let entry = entry.expect("read_dir entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("cu") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let obj = out_dir.join(format!("{}.o", stem));

        let status = Command::new(&nvcc)
            .args([
                "-arch=sm_70",
                "-O3",
                "-std=c++17",
                "--compiler-options",
                "-fPIC",
                "-c",
            ])
            .arg(&path)
            .arg("-o")
            .arg(&obj)
            .status()
            .unwrap_or_else(|e| panic!("failed to invoke nvcc: {}", e));
        if !status.success() {
            panic!("nvcc failed on {}", path.display());
        }
        object_files.push(obj);
    }

    if object_files.is_empty() {
        panic!(
            "GPU feature enabled but no .cu sources found in {}",
            cuda_src_dir.display()
        );
    }

    let lib_path = out_dir.join("libbn254_gemm_cuda.a");
    let _ = std::fs::remove_file(&lib_path);
    let mut ar = Command::new("ar");
    ar.arg("rcs").arg(&lib_path);
    for obj in &object_files {
        ar.arg(obj);
    }
    let status = ar.status().expect("failed to invoke ar");
    if !status.success() {
        panic!("ar rcs failed when packaging CUDA objects");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=bn254_gemm_cuda");
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_home.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
    // nvcc-compiled object files may need libstdc++ at link time.
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
