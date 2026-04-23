//! Tests for R3: bidirectional valid-positions mask in Gemma4 vision attention.

use mlx_core::DType;
use mlx_models::gemma4::build_vision_attention_mask;

#[test]
fn vision_mask_shape_is_correct() {
    let num_real = 4;
    let max_patches = 6;
    let mask = build_vision_attention_mask(num_real, max_patches, DType::Float32).unwrap();

    assert_eq!(mask.shape_raw(), vec![1, 1, 6, 6]);
}

#[test]
fn vision_mask_valid_positions_are_zero() {
    let num_real = 3;
    let max_patches = 5;
    let mask = build_vision_attention_mask(num_real, max_patches, DType::Float32).unwrap();
    let data: Vec<f32> = mask.to_vec_f32().unwrap();

    // All valid x valid positions should be 0.0
    for i in 0..num_real {
        for j in 0..num_real {
            let idx = i * max_patches + j;
            assert_eq!(
                data[idx], 0.0,
                "position ({},{}) should be 0.0 but was {}",
                i, j, data[idx]
            );
        }
    }
}

#[test]
fn vision_mask_padding_positions_are_neg_inf() {
    let num_real = 2;
    let max_patches = 4;
    let mask = build_vision_attention_mask(num_real, max_patches, DType::Float32).unwrap();
    let data: Vec<f32> = mask.to_vec_f32().unwrap();

    // Any position involving padding (index >= num_real) should be -inf
    for i in 0..max_patches {
        for j in 0..max_patches {
            let idx = i * max_patches + j;
            if i >= num_real || j >= num_real {
                assert!(
                    data[idx].is_infinite() && data[idx].is_sign_negative(),
                    "position ({},{}) should be -inf but was {}",
                    i, j, data[idx]
                );
            }
        }
    }
}

#[test]
fn vision_mask_no_padding() {
    // When num_real == max_patches, there should be no padding at all
    let num_real = 5;
    let max_patches = 5;
    let mask = build_vision_attention_mask(num_real, max_patches, DType::Float32).unwrap();
    let data: Vec<f32> = mask.to_vec_f32().unwrap();

    for &val in &data {
        assert_eq!(val, 0.0, "all values should be 0.0 when no padding");
    }
}

#[test]
fn vision_mask_all_padding() {
    // When num_real is 0, everything should be -inf
    let num_real = 0;
    let max_patches = 3;
    let mask = build_vision_attention_mask(num_real, max_patches, DType::Float32).unwrap();
    let data: Vec<f32> = mask.to_vec_f32().unwrap();

    for &val in &data {
        assert!(
            val.is_infinite() && val.is_sign_negative(),
            "all values should be -inf when all padding, but got {}",
            val
        );
    }
}

#[test]
fn vision_mask_dtype_conversion() {
    // Test that the mask is correctly converted to the requested dtype
    let num_real = 2;
    let max_patches = 3;
    let mask_f32 = build_vision_attention_mask(num_real, max_patches, DType::Float32).unwrap();
    let mask_f16 = build_vision_attention_mask(num_real, max_patches, DType::Float16).unwrap();

    assert_eq!(mask_f32.dtype(), DType::Float32);
    assert_eq!(mask_f16.dtype(), DType::Float16);

    // Values should still represent the same mask
    let f32_data: Vec<f32> = mask_f32.to_vec_f32().unwrap();
    let f16_data: Vec<f32> = mask_f16.to_vec_f32().unwrap();

    for (a, b) in f32_data.iter().zip(f16_data.iter()) {
        if a.is_infinite() {
            assert!(b.is_infinite() && b.is_sign_negative());
        } else {
            assert!((a - b).abs() < 1e-3, "f32 {} vs f16 {}", a, b);
        }
    }
}

#[test]
fn vision_mask_bidirectional_symmetry() {
    // The mask should be symmetric: mask[i,j] == mask[j,i]
    let num_real = 3;
    let max_patches = 6;
    let mask = build_vision_attention_mask(num_real, max_patches, DType::Float32).unwrap();
    let data: Vec<f32> = mask.to_vec_f32().unwrap();

    for i in 0..max_patches {
        for j in 0..max_patches {
            let idx_ij = i * max_patches + j;
            let idx_ji = j * max_patches + i;
            assert_eq!(
                data[idx_ij], data[idx_ji],
                "mask should be symmetric at ({},{})",
                i, j
            );
        }
    }
}

#[test]
fn vision_mask_single_patch() {
    // Edge case: only one valid patch
    let num_real = 1;
    let max_patches = 4;
    let mask = build_vision_attention_mask(num_real, max_patches, DType::Float32).unwrap();
    let data: Vec<f32> = mask.to_vec_f32().unwrap();

    // [0,0] should be 0.0 (valid attends to valid)
    assert_eq!(data[0], 0.0);

    // Everything else should be -inf
    for idx in 1..data.len() {
        assert!(
            data[idx].is_infinite() && data[idx].is_sign_negative(),
            "index {} should be -inf but was {}",
            idx, data[idx]
        );
    }
}
