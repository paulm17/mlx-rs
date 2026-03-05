use std::fmt;

/// Represents the shape (dimensions) of an array.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Create a shape from a slice of dimensions.
    pub fn from_dims(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// The dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Total number of elements.
    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    /// Get a specific dimension. Supports negative indexing from the end.
    pub fn dim(&self, idx: i32) -> usize {
        let rank = self.0.len() as i32;
        let actual = if idx < 0 { rank + idx } else { idx };
        self.0[actual as usize]
    }

    /// Number of dimensions (alias for rank).
    pub fn ndim(&self) -> usize {
        self.rank()
    }

    /// Convert to a Vec of i32 (for passing to mlx-c).
    pub fn to_i32(&self) -> Vec<i32> {
        self.0.iter().map(|&d| d as i32).collect()
    }

    /// Create from a Vec of i32 (from mlx-c).
    pub fn from_i32(dims: &[i32]) -> Self {
        Self(dims.iter().map(|&d| d as usize).collect())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::from_dims(dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl From<(usize,)> for Shape {
    fn from(dims: (usize,)) -> Self {
        Self(vec![dims.0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from(dims: (usize, usize)) -> Self {
        Self(vec![dims.0, dims.1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(dims: (usize, usize, usize)) -> Self {
        Self(vec![dims.0, dims.1, dims.2])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(dims: (usize, usize, usize, usize)) -> Self {
        Self(vec![dims.0, dims.1, dims.2, dims.3])
    }
}

/// Trait for types that can be converted into a Shape.
pub trait IntoShape {
    fn into_shape(self) -> Shape;
}

impl IntoShape for Shape {
    fn into_shape(self) -> Shape {
        self
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Shape {
        Shape::from_dims(self)
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Shape {
        Shape(self)
    }
}

impl IntoShape for (usize,) {
    fn into_shape(self) -> Shape {
        Shape::from(self)
    }
}

impl IntoShape for (usize, usize) {
    fn into_shape(self) -> Shape {
        Shape::from(self)
    }
}

impl IntoShape for (usize, usize, usize) {
    fn into_shape(self) -> Shape {
        Shape::from(self)
    }
}

impl IntoShape for (usize, usize, usize, usize) {
    fn into_shape(self) -> Shape {
        Shape::from(self)
    }
}
