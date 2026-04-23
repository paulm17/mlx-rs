//! Tests for R2: masked_scatter used in Gemma4 vision feature injection.

use mlx_core::{Array, DType};
use mlx_models::gemma4::Gemma4;

fn bool_mask(values: &[bool]) -> Array {
    let ints: Vec<i32> = values.iter().map(|&b| if b { 1 } else { 0 }).collect();
    Array::from_slice_i32(&ints)
        .unwrap()
        .as_type(DType::Bool)
        .unwrap()
}

#[test]
fn masked_scatter_basic() {
    // Destination: [1, 2, 3, 4, 5]
    let dest = Array::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    // Mask: [False, True, False, True, False]
    let mask = bool_mask(&[false, true, false, true, false]);
    // Source: [10, 20]
    let source = Array::from_slice_f32(&[10.0, 20.0]).unwrap();

    let result = Gemma4::masked_scatter(&dest, &mask, &source).unwrap();
    let result_slice: Vec<f32> = result.to_vec_f32().unwrap();

    assert_eq!(result_slice, vec![1.0, 10.0, 3.0, 20.0, 5.0]);
}

#[test]
fn masked_scatter_all_true() {
    let dest = Array::from_slice_f32(&[1.0, 2.0, 3.0]).unwrap();
    let mask = bool_mask(&[true, true, true]);
    let source = Array::from_slice_f32(&[10.0, 20.0, 30.0]).unwrap();

    let result = Gemma4::masked_scatter(&dest, &mask, &source).unwrap();
    let result_slice: Vec<f32> = result.to_vec_f32().unwrap();

    assert_eq!(result_slice, vec![10.0, 20.0, 30.0]);
}

#[test]
fn masked_scatter_all_false() {
    let dest = Array::from_slice_f32(&[1.0, 2.0, 3.0]).unwrap();
    let mask = bool_mask(&[false, false, false]);
    let source = Array::from_slice_f32(&[10.0, 20.0, 30.0]).unwrap();

    let result = Gemma4::masked_scatter(&dest, &mask, &source).unwrap();
    let result_slice: Vec<f32> = result.to_vec_f32().unwrap();

    assert_eq!(result_slice, vec![1.0, 2.0, 3.0]);
}

#[test]
fn masked_scatter_source_shorter_wraps() {
    // Python masked_scatter uses cumsum indices modulo source length
    let dest = Array::from_slice_f32(&[0.0, 0.0, 0.0, 0.0]).unwrap();
    let mask = bool_mask(&[true, true, true, true]);
    let source = Array::from_slice_f32(&[7.0, 8.0]).unwrap();

    let result = Gemma4::masked_scatter(&dest, &mask, &source).unwrap();
    let result_slice: Vec<f32> = result.to_vec_f32().unwrap();

    // indices = [0,1,2,3], mod 2 -> [0,1,0,1], source = [7,8,7,8]
    assert_eq!(result_slice, vec![7.0, 8.0, 7.0, 8.0]);
}

#[test]
fn masked_scatter_image_token_injection_pattern() {
    // Simulate the actual use case in Gemma4:
    // text embeddings shape [1, 5, 3] with image token at position 2
    let embeddings_data: Vec<f32> = vec![
        // token 0
        1.0, 1.0, 1.0,
        // token 1
        2.0, 2.0, 2.0,
        // token 2 (image placeholder)
        0.0, 0.0, 0.0,
        // token 3
        4.0, 4.0, 4.0,
        // token 4
        5.0, 5.0, 5.0,
    ];
    let embeddings = Array::from_slice_f32(&embeddings_data)
        .unwrap()
        .reshape(&[1, 5, 3])
        .unwrap();

    // mask = input_ids == image_token_id -> only position 2 is True
    // In actual usage the mask is expanded to match embedding shape before calling masked_scatter
    let mask = bool_mask(&[false, false, true, false, false])
        .reshape(&[1, 5])
        .unwrap()
        .expand_dims(-1)
        .unwrap()
        .broadcast_to(&[1, 5, 3])
        .unwrap();

    // vision features for that single image token
    let vision_features = Array::from_slice_f32(&[9.0, 9.0, 9.0])
        .unwrap()
        .reshape(&[1, 1, 3])
        .unwrap();

    let result = Gemma4::masked_scatter(&embeddings, &mask, &vision_features).unwrap();
    let result_vec: Vec<f32> = result.to_vec_f32().unwrap();

    assert_eq!(result.shape_raw(), vec![1, 5, 3]);
    assert_eq!(
        result_vec,
        vec![
            1.0, 1.0, 1.0, // token 0 unchanged
            2.0, 2.0, 2.0, // token 1 unchanged
            9.0, 9.0, 9.0, // token 2 replaced with vision features
            4.0, 4.0, 4.0, // token 3 unchanged
            5.0, 5.0, 5.0, // token 4 unchanged
        ]
    );
}

#[test]
fn masked_scatter_preserves_shape() {
    let dest = Array::from_slice_f32(&[1.0; 24]).unwrap().reshape(&[2, 3, 4]).unwrap();
    let mask = bool_mask(&[true, false, false, true, false, false])
        .reshape(&[2, 3])
        .unwrap()
        .expand_dims(-1)
        .unwrap()
        .broadcast_to(&[2, 3, 4])
        .unwrap();
    let source = Array::from_slice_f32(&[7.0, 8.0]).unwrap();

    let result = Gemma4::masked_scatter(&dest, &mask, &source).unwrap();

    assert_eq!(result.shape_raw(), vec![2, 3, 4]);
}
