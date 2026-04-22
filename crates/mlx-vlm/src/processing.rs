use anyhow::Result;
use image::imageops::FilterType;

/// A processed image ready for the vision model.
pub struct ProcessedImage {
    /// Flattened pixel values in channels-first order [C, H, W].
    pub pixel_values: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// Image processor for Gemma4 vision encoder.
pub struct Gemma4ImageProcessor {
    pub target_size: u32,
}

impl Gemma4ImageProcessor {
    /// Create a new processor with the given target square size.
    pub fn new(target_size: u32) -> Self {
        Self { target_size }
    }

    /// Load and preprocess an image.
    ///
    /// Resizes preserving aspect ratio to fit within `target_size`, centre-pads
    /// to a square, rescales to [0, 1], and returns channels-first data.
    pub fn process(&self, image_path: &std::path::Path) -> Result<ProcessedImage> {
        let img = image::open(image_path)?;
        let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);

        // Preserve aspect ratio, fit within target_size
        let scale =
            (self.target_size as f32 / orig_w).min(self.target_size as f32 / orig_h);
        let new_w = (orig_w * scale) as u32;
        let new_h = (orig_h * scale) as u32;

        let resized = img.resize(new_w, new_h, FilterType::Lanczos3);
        let rgb = resized.to_rgb8();

        // Centre-pad to target_size square
        let mut pixel_values =
            vec![0.0f32; 3 * self.target_size as usize * self.target_size as usize];
        let x_offset = (self.target_size - new_w) / 2;
        let y_offset = (self.target_size - new_h) / 2;

        for y in 0..new_h {
            for x in 0..new_w {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let idx = c * self.target_size as usize * self.target_size as usize
                        + (y_offset + y) as usize * self.target_size as usize
                        + (x_offset + x) as usize;
                    pixel_values[idx] = pixel[c] as f32 / 255.0;
                }
            }
        }

        Ok(ProcessedImage {
            pixel_values,
            width: self.target_size as usize,
            height: self.target_size as usize,
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
