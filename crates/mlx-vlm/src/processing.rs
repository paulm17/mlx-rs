use anyhow::Result;
use image::imageops::FilterType;

/// A processed image ready for the vision model.
pub struct ProcessedImage {
    /// Flattened pixel values in channels-first order [C, H, W].
    pub pixel_values: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// Number of soft tokens this image produces after the vision tower.
    pub num_soft_tokens: usize,
}

/// Image processor for Gemma4 vision encoder.
///
/// Matches Python `mlx_vlm.models.gemma4.processing_gemma4.Gemma4ImageProcessor`.
pub struct Gemma4ImageProcessor {
    pub patch_size: usize,
    pub max_soft_tokens: usize,
    pub pooling_kernel_size: usize,
}

impl Gemma4ImageProcessor {
    /// Create a new processor with the given config values.
    pub fn new(patch_size: usize, max_soft_tokens: usize, pooling_kernel_size: usize) -> Self {
        Self {
            patch_size,
            max_soft_tokens,
            pooling_kernel_size,
        }
    }

    /// Compute target resize dimensions preserving aspect ratio while fitting
    /// within the patch budget.
    ///
    /// Returns (target_width, target_height).
    fn compute_target_size(&self, width: u32, height: u32) -> (u32, u32) {
        let max_patches = self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size;
        let target_px = max_patches * self.patch_size * self.patch_size;
        let area = (width as f64) * (height as f64);
        let factor = (target_px as f64 / area).sqrt();
        let side_mult = (self.pooling_kernel_size * self.patch_size) as i32;

        let mut target_height = ((factor * height as f64 / side_mult as f64).floor() as i32 * side_mult).max(0) as u32;
        let mut target_width = ((factor * width as f64 / side_mult as f64).floor() as i32 * side_mult).max(0) as u32;
        let max_side_length = (self.max_soft_tokens / (self.pooling_kernel_size * self.pooling_kernel_size))
            * (self.pooling_kernel_size * self.patch_size);

        if target_height == 0 && target_width == 0 {
            // Should not happen for reasonable images, but fallback
            target_height = side_mult as u32;
            target_width = side_mult as u32;
        } else if target_height == 0 {
            target_height = side_mult as u32;
            target_width = (((width as f64 / height as f64).floor() as usize) * self.pooling_kernel_size * self.patch_size)
                .min(max_side_length) as u32;
        } else if target_width == 0 {
            target_width = side_mult as u32;
            target_height = (((height as f64 / width as f64).floor() as usize) * self.pooling_kernel_size * self.patch_size)
                .min(max_side_length) as u32;
        }

        (target_width, target_height)
    }

    /// Load and preprocess an image.
    ///
    /// Resizes preserving aspect ratio to fit within the patch budget,
    /// rescales to [0, 1], and returns channels-first data.
    pub fn process(&self, image_path: &std::path::Path) -> Result<ProcessedImage> {
        let img = image::open(image_path)?;
        let (orig_w, orig_h) = (img.width(), img.height());

        let (target_w, target_h) = self.compute_target_size(orig_w, orig_h);

        // Use resize_exact because target_w/target_h already preserve aspect ratio.
        let resized = img.resize_exact(target_w, target_h, FilterType::Lanczos3);
        let rgb = resized.to_rgb8();

        let img_w = rgb.width();
        let img_h = rgb.height();

        let mut pixel_values = vec![0.0f32; 3 * img_w as usize * img_h as usize];

        for y in 0..img_h {
            for x in 0..img_w {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let idx = c * img_w as usize * img_h as usize
                        + y as usize * img_w as usize
                        + x as usize;
                    pixel_values[idx] = pixel[c] as f32 / 255.0;
                }
            }
        }

        let num_patches = (img_h as usize / self.patch_size) * (img_w as usize / self.patch_size);
        let num_soft_tokens = num_patches / (self.pooling_kernel_size * self.pooling_kernel_size);

        Ok(ProcessedImage {
            pixel_values,
            width: img_w as usize,
            height: img_h as usize,
            num_soft_tokens,
        })
    }

    /// Convert processed image pixels into an MLX Array.
    ///
    /// Shape: [1, 3, H, W]
    pub fn to_array(&self, processed: &ProcessedImage) -> Result<mlx_core::Array> {
        let shape = vec![
            1i32,
            3i32,
            processed.height as i32,
            processed.width as i32,
        ];
        Ok(mlx_core::Array::from_slice_f32(&processed.pixel_values)?.reshape(&shape)?)
    }
}
