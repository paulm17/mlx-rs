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
    /// Resizes to a square, rescales to [0, 1], and returns channels-first data.
    pub fn process(&self, image_path: &std::path::Path) -> Result<ProcessedImage> {
        let img = image::open(image_path)?;
        let resized =
            img.resize_to_fill(self.target_size, self.target_size, FilterType::Lanczos3);
        let rgb = resized.to_rgb8();

        let mut pixel_values =
            Vec::with_capacity(3 * self.target_size as usize * self.target_size as usize);
        for c in 0..3 {
            for y in 0..self.target_size {
                for x in 0..self.target_size {
                    let pixel = rgb.get_pixel(x, y);
                    pixel_values.push(pixel[c] as f32 / 255.0);
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
