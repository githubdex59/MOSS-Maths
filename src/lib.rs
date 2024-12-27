use std::f64;
use rand::Rng;

pub fn random_f64() -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen::<f64>()
}

pub fn add(a: i32, b: i32) -> i32 {
    a+b
}
pub fn sub(a: i32, b: i32) -> i32 {
    a-b
}
pub fn mul(a: i32, b: i32) -> i32 {
    a*b
}
pub fn div(a: f32, b: f32) -> f32 {
    a/b
}

fn fade(t: f64) -> f64 {
    // Smoothstep fade function
    6.0 * t.powi(5) - 15.0 * t.powi(4) + 10.0 * t.powi(3)
}


/// Classic lerp
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    // Linear interpolation
    a + t * (b - a)
}

fn gradient(hash: u8, x: f64, y: f64) -> f64 {
    // Use hash to select a gradient vector
    match hash & 0x3 {
        0 => x + y,  // (1, 1)
        1 => -x + y, // (-1, 1)
        2 => x - y,  // (1, -1)
        3 => -x - y, // (-1, -1)
        _ => 0.0,    // Should never reach here
    }
}

/// Implements the perlin noise algorithm.
/// 
/// Using this is better than writing it yourself.
/// Unless you're willing to debug for five odd hours before you finish the first line.
/// However you could probably do it better than me ;)
/// 
/// # Examples
/// 
/// ```
/// for y in 0..10 {
///     for x in 0..10 {
///         let value = perlin(x as f64 * 0.1, y as f64 * 0.1); // Scale coordinate
///         print!("{:.2} ", value);
///     }
///     println!();
/// }
/// ```
#[inline]
pub fn perlin(x: f64, y: f64) -> f64 {
    // Find the grid cell
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    // Relative coordinates within the grid cell
    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    // Hash function to get pseudo-random values
    let hash = |x: i32, y: i32| -> u8 {
        // A simple pseudo-random hash function
        let seed = 42; // Change for different noise patterns
        (x * 374761393 + y * 668265263 + seed) as u8
    };

    // Compute hash values for each corner of the grid cell
    let h00 = hash(x0, y0);
    let h10 = hash(x1, y0);
    let h01 = hash(x0, y1);
    let h11 = hash(x1, y1);

    // Compute dot products
    let dot00 = gradient(h00, dx, dy);
    let dot10 = gradient(h10, dx - 1.0, dy);
    let dot01 = gradient(h01, dx, dy - 1.0);
    let dot11 = gradient(h11, dx - 1.0, dy - 1.0);

    // Fade curves for interpolation
    let u = fade(dx);
    let v = fade(dy);

    // Interpolate
    let x1 = lerp(dot00, dot10, u); // Bottom row
    let x2 = lerp(dot01, dot11, u); // Top row
    lerp(x1, x2, v) // Final interpolation
}

pub fn sin(x: f64) -> f64 {
    x.sin()
}

pub fn cos(x: f64) -> f64 {
    x.cos()
}

pub fn tan(x: f64) -> f64 {
    x.tan()
}

pub fn asin(x: f64) -> f64 {
    x.asin()
}

pub fn acos(x: f64) -> f64 {
    x.acos()
}

pub fn atan(x: f64) -> f64 {
    x.atan()
}

pub fn exp(x: f64) -> f64 {
    x.exp()
}

pub fn ln(x: f64) -> f64 {
    x.ln()
}

pub fn log(x: f64, base: f64) -> f64 {
    x.log(base)
}

pub fn mean(data: &[f64]) -> f64 {
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

pub fn median(data: &mut [f64]) -> f64 {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = data.len() / 2;
    if data.len() % 2 == 0 {
        (data[mid - 1] + data[mid]) / 2.0
    } else {
        data[mid]
    }
}

pub fn std_dev(data: &[f64]) -> f64 {
    let mean = mean(data);
    let variance: f64 = data.iter().map(|value| {
        let diff = mean - *value;
        diff * diff
    }).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Vector 2D
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}
impl Vec2 {
    pub fn add(&self, other: &Vec2) -> Vec2 {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    pub fn sub(&self, other: &Vec2) -> Vec2 {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    pub fn dot(&self, other: &Vec2) -> f64 {
        self.x * other.x + self.y * other.y
    }

    pub fn cross(&self, other: &Vec2) -> f64 {
        self.x * other.y - self.y * other.x
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    // Create a new matrix with given rows and columns, initialized to zero
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    // Create a new matrix from a 2D vector
    pub fn from_vec(data: Vec<Vec<f64>>) -> Matrix {
        let rows = data.len();
        let cols = data[0].len();
        Matrix { rows, cols, data }
    }

    // Add two matrices
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    // Subtract two matrices
    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        result
    }

    // Multiply two matrices
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }

    // Transpose the matrix
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    // Print the matrix
    pub fn print(&self) {
        for row in &self.data {
            for val in row {
                print!("{:.2} ", val);
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin() {
        for y in 0..10 {
            for x in 0..10 {
                let value = perlin(x as f64 * 0.1, y as f64 * 0.1);
                print!("{:.2} ", value);
            }
            println!();
        }
        
    }
}






