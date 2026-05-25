use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::anyhow;
use protocol::{ByteConversion, LargeField};

/// Reads first k lines from the primary file, or fallback file if primary doesn't exist
/// Validates that each line can be represented as a 250-bit binary string
pub fn read_input_from_files(
    primary_file_path: &str,
    fallback_file_path: &str,
    k: usize,
) -> Result<Vec<LargeField>,anyhow::Error> {
    // Check which file to use
    let file_path = if Path::new(primary_file_path).exists() {
        log::info!("Primary file exists, reading from: {}", primary_file_path);
        primary_file_path
    } else {
        log::info!("Primary file doesn't exist, reading from fallback: {}", fallback_file_path);
        fallback_file_path
    };

    // Open and read the file
    let file = File::open(file_path)
        .map_err(|e| anyhow!("Failed to open file {}: {}", file_path, e))?;
    
    let reader = BufReader::new(file);
    let mut converted_fes = Vec::new();
    let mut line_count = 0;

    for line_result in reader.lines() {
        if line_count >= k {
            break;
        }

        let line = line_result
            .map_err(|e| anyhow!("Failed to read line {}: {}", line_count + 1, e))?;
        
        // Validate that the line can be represented as a 250-bit binary string
        let conversion_output = convert_string_to_large_field(&line);
        if conversion_output.is_none() {
            return Err(anyhow!(
                "Line {} cannot be represented as a 250-bit binary string: '{}'", 
                line_count + 1, 
                line
            ));
        }

        converted_fes.push(conversion_output.unwrap());
        line_count += 1;
    }

    if line_count < k {
        log::error!("File {} contains only {} inputs, but {} inputs were requested", 
                    file_path, line_count, k);
        return Err(anyhow!("Insufficient inputs in file {}", file_path));
    }

    log::info!("Successfully read {} lines from {}", converted_fes.len(), file_path);
    Ok(converted_fes)
}

/// Convert a free-form text line into a `LargeField` (Fp4_61) element by treating its
/// bytes as a big-endian 32-byte buffer (zero-padded on the left when shorter, rejected
/// when longer). The previous BN254-based code did this implicitly via `from_hex`; we
/// keep the same behaviour for input compatibility.
fn convert_string_to_large_field(input: &str) -> Option<LargeField> {
    let mut bytes = input.as_bytes().to_vec();
    if bytes.len() > 32 {
        return None;
    }
    let mut padded = [0u8; 32];
    padded[32 - bytes.len()..].copy_from_slice(&bytes);
    bytes.clear();
    LargeField::from_bytes_be(&padded).ok()
}