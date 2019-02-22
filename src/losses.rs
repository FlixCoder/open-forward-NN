/// Calculate the error between predictions and targets:
/// Mean squared error (for regression)
/// Potentially ignores different vector lenghts!
pub fn mse(preds:&[Vec<f64>], targets:&[Vec<f64>]) -> f64
{
    let mut avg_error = 0.0;
    for (pred, target) in preds.iter().zip(targets.iter())
    {
        let mut metric = 0.0;
        for (yp, yt) in pred.iter().zip(target.iter())
        {
            let error = *yt - *yp;
            metric += error * error;
        }
        metric /= target.len() as f64;
        avg_error += metric;
    }
    avg_error /= targets.len() as f64;
    avg_error
}

/// Calculate the error between predictions and targets:
/// Root mean squared error (for regression)
/// Potentially ignores different vector lenghts!
pub fn rmse(preds:&[Vec<f64>], targets:&[Vec<f64>]) -> f64
{
    let mut avg_error = 0.0;
    for (pred, target) in preds.iter().zip(targets.iter())
    {
        let mut metric = 0.0;
        for (yp, yt) in pred.iter().zip(target.iter())
        {
            let error = *yt - *yp;
            metric += error * error;
        }
        metric /= target.len() as f64;
        avg_error += metric.sqrt();
    }
    avg_error /= targets.len() as f64;
    avg_error
}

/// Calculate the error between predictions and targets:
/// Mean absolute error (for regression)
/// Potentially ignores different vector lenghts!
pub fn mae(preds:&[Vec<f64>], targets:&[Vec<f64>]) -> f64
{
    let mut avg_error = 0.0;
    for (pred, target) in preds.iter().zip(targets.iter())
    {
        let mut metric = 0.0;
        for (yp, yt) in pred.iter().zip(target.iter())
        {
            let error = *yt - *yp;
            metric += error.abs();
        }
        metric /= target.len() as f64;
        avg_error += metric;
    }
    avg_error /= targets.len() as f64;
    avg_error
}

/// Calculate the error between predictions and targets:
/// Mean absolute percentage error (better don't use if target has 0 values) (for regression)
/// Potentially ignores different vector lenghts!
pub fn mape(preds:&[Vec<f64>], targets:&[Vec<f64>]) -> f64
{
    let mut avg_error = 0.0;
    for (pred, target) in preds.iter().zip(targets.iter())
    {
        let mut metric = 0.0;
        for (yp, yt) in pred.iter().zip(target.iter())
        {
            let error = (*yt - *yp) / *yt;
            metric += error.abs();
        }
        metric *= 100.0 / target.len() as f64;
        avg_error += metric;
    }
    avg_error /= targets.len() as f64;
    avg_error
}

/// Calculate the error between predictions and targets:
/// logcosh (for regression)
/// Potentially ignores different vector lenghts!
pub fn logcosh(preds:&[Vec<f64>], targets:&[Vec<f64>]) -> f64
{
    let mut avg_error = 0.0;
    for (pred, target) in preds.iter().zip(targets.iter())
    {
        let mut metric = 0.0;
        for (yp, yt) in pred.iter().zip(target.iter())
        {
            let error = *yt - *yp;
            metric += error.cosh().ln();
        }
        metric /= target.len() as f64;
        avg_error += metric;
    }
    avg_error /= targets.len() as f64;
    avg_error
}

/// Calculate the error between predictions and targets:
/// binary cross-entropy (be sure to use 0, 1 classifiers+labels) (for classification)
/// Potentially ignores different vector lenghts!
pub fn binary_crossentropy(preds:&[Vec<f64>], targets:&[Vec<f64>]) -> f64
{
    let mut avg_error = 0.0;
    for (pred, target) in preds.iter().zip(targets.iter())
    {
        let mut metric = 0.0;
        for (yp, yt) in pred.iter().zip(target.iter())
        {
            let error = *yt * yp.ln() + (1.0 - *yt) * (1.0 - *yp).ln();
            metric += -error;
        }
        metric /= target.len() as f64;
        avg_error += metric;
    }
    avg_error /= targets.len() as f64;
    avg_error
}

/// Calculate the error between predictions and targets:
/// categorical cross-entropy (be sure to use 0, 1 classifiers+labels) (for classification)
/// Potentially ignores different vector lenghts!
pub fn categorical_crossentropy(preds:&[Vec<f64>], targets:&[Vec<f64>]) -> f64
{
    let mut avg_error = 0.0;
    for (pred, target) in preds.iter().zip(targets.iter())
    {
        let mut metric = 0.0;
        for (yp, yt) in pred.iter().zip(target.iter())
        {
            let error = *yt * (*yp).ln();
            metric += -error;
        }
        avg_error += metric;
    }
    avg_error /= targets.len() as f64;
    avg_error
}

/// Calculate the error between predictions and targets:
/// hinge loss (be sure to use 1, -1 classifiers+labels) (for classification)
/// Potentially ignores different vector lenghts!
pub fn hingeloss(preds:&[Vec<f64>], targets:&[Vec<f64>]) -> f64
{
    let mut avg_error = 0.0;
    for (pred, target) in preds.iter().zip(targets.iter())
    {
        let mut metric = 0.0;
        for (yp, yt) in pred.iter().zip(target.iter())
        {
            let error = 1.0 - *yt * *yp;
            metric += error.max(0.0);
        }
        metric /= target.len() as f64;
        avg_error += metric;
    }
    avg_error /= targets.len() as f64;
    avg_error
}
