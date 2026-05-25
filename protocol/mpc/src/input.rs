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

/// Convert a free-form text line into a `LargeField` (Fp4_61) element.
///
/// `LargeField` = `FieldElement<Mersenne61Degree4ExtensionField>` is laid out as four
/// 8-byte big-endian Mersenne-61 base-field limbs. Reading 8 bytes of arbitrary
/// text into a u64 and folding it modulo `2^61 - 1` is lossy whenever the high
/// byte has bit 5 or 6 set (which it always does for ASCII letters `0x60..0x7F`),
/// so the round-trip text → field → text mangles the input.
///
/// We sidestep the fold by leaving the high byte of each 8-byte limb at `0x00`
/// and packing 7 payload bytes into the low 7 bytes of each limb. Every limb's
/// u64 value then stays below `2^56 < 2^61`, so the Mersenne reduction is a
/// no-op and the round-trip is lossless.
///
/// Capacity: 4 limbs × 7 payload bytes = **28 bytes** per input line.
/// Inputs are right-aligned across the limbs — the *last* limb gets the last 7
/// input bytes (or fewer + leading zeros), then the previous limb, etc.
pub(crate) const MAX_INPUT_PAYLOAD: usize = 28;

fn convert_string_to_large_field(input: &str) -> Option<LargeField> {
    let bytes = input.as_bytes();
    if bytes.len() > MAX_INPUT_PAYLOAD {
        return None;
    }
    let mut padded = [0u8; 32];
    let mut remaining = bytes;
    for chunk in (0..4).rev() {
        let take = remaining.len().min(7);
        if take == 0 {
            break;
        }
        let chunk_base = chunk * 8;
        // Byte 0 of the chunk stays 0x00; payload occupies bytes 1..8, right-aligned
        // within those 7 bytes when the chunk is partially filled.
        let dest_start = chunk_base + 1 + (7 - take);
        padded[dest_start..chunk_base + 8]
            .copy_from_slice(&remaining[remaining.len() - take..]);
        remaining = &remaining[..remaining.len() - take];
    }
    LargeField::from_bytes_be(&padded).ok()
}