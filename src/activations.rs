//! Activation functions.

use crate::Float;

/// SELU factors for a Normal(0, 1) data distribution from https://arxiv.org/pdf/1706.02515.pdf
const SELU_LAMBDA: Float = 1.0507;
/// SELU factors for a Normal(0, 1) data distribution from https://arxiv.org/pdf/1706.02515.pdf
const SELU_ALPHA: Float = 1.6733;
// /// SELU factors for a Normal(0, 2) data distribution from https://arxiv.org/pdf/1706.02515.pdf
// const SELU_LAMBDA:Float = 1.06071;
// /// SELU factors for a Normal(0, 2) data distribution from https://arxiv.org/pdf/1706.02515.pdf
// const SELU_ALPHA:Float = 1.97126;

/// Linear activation function.
#[inline]
pub fn linear(x: Float) -> Float {
	x
}

/// ReLU activation function.
#[inline]
pub fn relu(x: Float) -> Float {
	x.max(0.0)
}

/// Leaky ReLU activation function.
#[inline]
pub fn lrelu(x: Float, factor: Float) -> Float {
	if x < 0.0 {
		factor * x
	} else {
		x
	}
}

/// ELU activation function.
#[inline]
pub fn elu(x: Float) -> Float {
	if x < 0.0 {
		x.exp_m1()
	} else {
		x
	}
}

/// PELU activation function.
#[inline]
pub fn pelu(x: Float, a: Float, b: Float) -> Float {
	if x < 0.0 {
		a * (x / b).exp() - a
	} else {
		(a / b) * x
	}
}

/// SELU activation function.
#[inline]
pub fn selu(x: Float) -> Float {
	SELU_LAMBDA * if x < 0.0 { SELU_ALPHA * x.exp() - SELU_ALPHA } else { x }
}

/// Sigmoid activation function.
#[inline]
pub fn sigmoid(x: Float) -> Float {
	1.0 / (1.0 + (-x).exp())
}

/// Tanh activation function.
#[inline]
pub fn tanh(x: Float) -> Float {
	x.tanh()
}

/// Absolute activation function.
#[inline]
pub fn abs(x: Float) -> Float {
	x.abs()
}

/// Quadratic activation function.
#[inline]
pub fn quadratic(x: Float) -> Float {
	x * x
}

/// Cubic activation function.
#[inline]
pub fn cubic(x: Float) -> Float {
	x * x * x
}

/// Clipped linear activation function.
#[inline]
pub fn clip_linear(x: Float) -> Float {
	x.min(1.0).max(-1.0)
}

/// Gaussian activation function.
#[inline]
pub fn gaussian(x: Float) -> Float {
	(-x * x).exp()
}

/// Softplus activation function.
#[inline]
pub fn softplus(x: Float) -> Float {
	x.exp().ln_1p()
}

/// Softmax activation function.
#[inline]
pub fn softmax(arr: &mut [Float]) {
	let norm: Float = arr.iter().map(|x| x.exp()).sum();
	for val in arr.iter_mut() {
		*val = val.exp() / norm;
	}
}
