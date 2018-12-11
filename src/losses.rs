//! Loss computations.

use crate::Float;

/// Calculate the error between predictions and targets:
/// Mean squared error (for regression)
/// Potentially ignores different vector lenghts!
#[must_use]
pub fn mse(preds: &[Vec<Float>], targets: &[Vec<Float>]) -> Float {
	let mut avg_error = 0.0;
	for (pred, target) in preds.iter().zip(targets.iter()) {
		let mut metric = 0.0;
		for (yp, yt) in pred.iter().zip(target.iter()) {
			let error = *yt - *yp;
			metric += error * error;
		}
		metric /= target.len() as Float;
		avg_error += metric;
	}
	avg_error /= targets.len() as Float;
	avg_error
}

/// Calculate the error between predictions and targets:
/// Root mean squared error (for regression)
/// Potentially ignores different vector lenghts!
#[must_use]
pub fn rmse(preds: &[Vec<Float>], targets: &[Vec<Float>]) -> Float {
	let mut avg_error = 0.0;
	for (pred, target) in preds.iter().zip(targets.iter()) {
		let mut metric = 0.0;
		for (yp, yt) in pred.iter().zip(target.iter()) {
			let error = *yt - *yp;
			metric += error * error;
		}
		metric /= target.len() as Float;
		avg_error += metric.sqrt();
	}
	avg_error /= targets.len() as Float;
	avg_error
}

/// Calculate the error between predictions and targets:
/// Mean absolute error (for regression)
/// Potentially ignores different vector lenghts!
#[must_use]
pub fn mae(preds: &[Vec<Float>], targets: &[Vec<Float>]) -> Float {
	let mut avg_error = 0.0;
	for (pred, target) in preds.iter().zip(targets.iter()) {
		let mut metric = 0.0;
		for (yp, yt) in pred.iter().zip(target.iter()) {
			let error = *yt - *yp;
			metric += error.abs();
		}
		metric /= target.len() as Float;
		avg_error += metric;
	}
	avg_error /= targets.len() as Float;
	avg_error
}

/// Calculate the error between predictions and targets:
/// Mean absolute percentage error (better don't use if target has 0 values)
/// (for regression) Potentially ignores different vector lenghts!
#[must_use]
pub fn mape(preds: &[Vec<Float>], targets: &[Vec<Float>]) -> Float {
	let mut avg_error = 0.0;
	for (pred, target) in preds.iter().zip(targets.iter()) {
		let mut metric = 0.0;
		for (yp, yt) in pred.iter().zip(target.iter()) {
			let error = (*yt - *yp) / *yt;
			metric += error.abs();
		}
		metric *= 100.0 / target.len() as Float;
		avg_error += metric;
	}
	avg_error /= targets.len() as Float;
	avg_error
}

/// Calculate the error between predictions and targets:
/// logcosh (for regression)
/// Potentially ignores different vector lenghts!
#[must_use]
pub fn logcosh(preds: &[Vec<Float>], targets: &[Vec<Float>]) -> Float {
	let mut avg_error = 0.0;
	for (pred, target) in preds.iter().zip(targets.iter()) {
		let mut metric = 0.0;
		for (yp, yt) in pred.iter().zip(target.iter()) {
			let error = *yt - *yp;
			metric += error.cosh().ln();
		}
		metric /= target.len() as Float;
		avg_error += metric;
	}
	avg_error /= targets.len() as Float;
	avg_error
}

/// Calculate the error between predictions and targets:
/// binary cross-entropy (be sure to use 0, 1 classifiers+labels) (for
/// classification) Potentially ignores different vector lenghts!
#[must_use]
pub fn binary_crossentropy(preds: &[Vec<Float>], targets: &[Vec<Float>]) -> Float {
	let mut avg_error = 0.0;
	for (pred, target) in preds.iter().zip(targets.iter()) {
		let mut metric = 0.0;
		for (yp, yt) in pred.iter().zip(target.iter()) {
			let error = (*yt).mul_add(yp.ln(), (1.0 - *yt) * (1.0 - *yp).ln());
			metric += -error;
		}
		metric /= target.len() as Float;
		avg_error += metric;
	}
	avg_error /= targets.len() as Float;
	avg_error
}

/// Calculate the error between predictions and targets:
/// categorical cross-entropy (be sure to use 0, 1 classifiers+labels) (for
/// classification) Potentially ignores different vector lenghts!
#[must_use]
pub fn categorical_crossentropy(preds: &[Vec<Float>], targets: &[Vec<Float>]) -> Float {
	let mut avg_error = 0.0;
	for (pred, target) in preds.iter().zip(targets.iter()) {
		let mut metric = 0.0;
		for (yp, yt) in pred.iter().zip(target.iter()) {
			let error = *yt * (*yp).ln();
			metric += -error;
		}
		avg_error += metric;
	}
	avg_error /= targets.len() as Float;
	avg_error
}

/// Calculate the error between predictions and targets:
/// hinge loss (be sure to use 1, -1 classifiers+labels) (for classification)
/// Potentially ignores different vector lenghts!
#[must_use]
pub fn hingeloss(preds: &[Vec<Float>], targets: &[Vec<Float>]) -> Float {
	let mut avg_error = 0.0;
	for (pred, target) in preds.iter().zip(targets.iter()) {
		let mut metric = 0.0;
		for (yp, yt) in pred.iter().zip(target.iter()) {
			let error = 1.0 - *yt * *yp;
			metric += error.max(0.0);
		}
		metric /= target.len() as Float;
		avg_error += metric;
	}
	avg_error /= targets.len() as Float;
	avg_error
}
