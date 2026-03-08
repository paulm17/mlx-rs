use mlx_core::{Array, Result};

/// Apply scaled dot-product attention using MLX's optimized kernel.
///
/// # Arguments
/// * `q` — queries, shape `[batch, heads, seq_len, head_dim]`
/// * `k` — keys, shape `[batch, kv_heads, kv_len, head_dim]`
/// * `v` — values, shape `[batch, kv_heads, kv_len, head_dim]`
/// * `scale` — attention scale (typically `1 / sqrt(head_dim)`)
/// * `mask` — optional attention mask
pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: Option<&Array>,
) -> Result<Array> {
    q.fast_scaled_dot_product_attention(k, v, scale, "none", mask)
}

/// Apply scaled dot-product attention with a causal mask.
pub fn causal_attention(q: &Array, k: &Array, v: &Array, scale: f32) -> Result<Array> {
    q.fast_scaled_dot_product_attention(k, v, scale, "causal", None)
}

/// Repeat key/value heads to match the number of query heads (GQA support).
///
/// If `n_rep == 1`, returns the input unchanged.
pub fn repeat_kv(x: &Array, n_rep: usize) -> Result<Array> {
    if n_rep <= 1 {
        return Ok(x.clone());
    }
    let shape = x.shape_raw();
    // shape: [batch, kv_heads, seq_len, head_dim]
    let batch = shape[0];
    let kv_heads = shape[1];
    let seq_len = shape[2];
    let head_dim = shape[3];

    // Expand and reshape: [B, kv_heads, 1, S, D] -> [B, kv_heads, n_rep, S, D] -> [B, kv_heads*n_rep, S, D]
    let expanded = x.reshape(&[batch, kv_heads, 1, seq_len, head_dim])?;

    // Repeat along the singleton axis by taking index 0 n_rep times.
    let rep_idx = vec![0i32; n_rep];
    let rep_idx = Array::from_slice_i32(&rep_idx)?;
    let repeated = expanded.take(&rep_idx, 2)?;
    let target = vec![batch, kv_heads, n_rep as i32, seq_len, head_dim];
    if repeated.shape_raw() != target {
        return Err(mlx_core::Error::Message(format!(
            "repeat_kv shape mismatch: got {:?}, expected {:?}",
            repeated.shape_raw(),
            target
        )));
    }
    repeated.reshape(&[batch, kv_heads * n_rep as i32, seq_len, head_dim])
}
