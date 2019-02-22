//! @author = FlixCoder
//!
//! Architecture influenced by Keras: Sequential models

#[macro_use]
extern crate serde_derive;

extern crate serde;
extern crate serde_json;
extern crate rand;

use std::io::prelude::*;
use std::fs::File;
use rand::Rng;
use rand::distributions::{Normal, Distribution};

//TODO:
//add (batch) normalization? (using running average)
//try new softmax without exp?
//multiplication node layer? (try some impossible stuff for backpropagation)
//add convolutional and pooling layers?

pub mod losses;


//SELU factors for a Normal(0, 1) data distribution from https://arxiv.org/pdf/1706.02515.pdf
const SELU_LAMBDA:f64 = 1.0507;
const SELU_ALPHA:f64 = 1.6733;
//SELU factors for a Normal(0, 2) data distribution from https://arxiv.org/pdf/1706.02515.pdf
//const SELU_LAMBDA:f64 = 1.06071;
//const SELU_ALPHA:f64 = 1.97126;


/// Define the available types of layers
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum Layer
{
    //Activation functions
    /// linear activation
    Linear,
    /// rectified linear unit
    ReLU,
    /// leaky rectified linear unit (factor = factor to apply for x < 0)
    LReLU(f64),
    /// parametric (leaky) rectified linear unit (factor = factor to apply for x < 0)
    PReLU(f64),
    /// exponential linear unit (alpha = 1)
    ELU,
    /// parametric exponential linear unit (factors a and b)
    PELU(f64, f64),
    /// scaled exponential linear unit (self-normalizing). parameters are adapted to var=1 data
    SELU,
    /// sigmoid
    Sigmoid,
    /// tanh
    Tanh,
    /// quadratic
    Quadratic,
    /// cubic
    Cubic,
    /// clipped linear activation [-1, 1]
    ClipLinear,
    /// gaussian
    Gaussian,
    /// soft plus
    SoftPlus,
    /// soft max
    SoftMax,
    
    //Regularization / Normalization / Utility
    /// Apply dropout to the previous layer (d = percent of neurons to drop)
    Dropout(f64),
    
    //Neuron-layers
    /// Dense layer (params = weights of the layer, be sure to have the correct dimensions! include bias as first parameter)
    Dense(Vec<Vec<f64>>),
}

/// Definition of usable initializers in Sequential.add_layer_dense
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Initializer
{
    /// Glorot/Xavier initialization
    Glorot,
    /// He initialization
    He,
    /// initialize with a constant value
    Const(f64),
}


/// Implementation of the neural network / sequential models of layers
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Sequential
{
    num_inputs:usize,
    layers:Vec<Layer>,
    num_outputs:usize,
}

impl Sequential
{
    /// Create a new instance of a sequential model
    /// num_inputs = the number of inputs to the model
    pub fn new(num_inputs:usize) -> Sequential
    {
        Sequential { num_inputs: num_inputs, layers: Vec::new(), num_outputs: num_inputs }
    }
    
    /// Returns the requested input dimension
    pub fn get_num_inputs(&self) -> usize
    {
        self.num_inputs
    }
    
    /// Get the layers (as ref)
    pub fn get_layers(&self) -> &Vec<Layer>
    {
        &self.layers
    }
    
    /// Get the layers (as mut)
    pub fn get_layers_mut(&mut self) -> &mut Vec<Layer>
    {
        &mut self.layers
    }
    
    /// Return the flat parameters of the layers (including LReLU factors).
    /// Used for evolution-strategies
    pub fn get_params(&self) -> Vec<f64>
    {
        let mut params = Vec::new();
        for layer in self.layers.iter()
        {
            match layer
            {
                //Activation functions
                //Layer::LReLU(factor) => params.push(*factor),
                Layer::PReLU(factor) => params.push(*factor),
                Layer::PELU(a, b) => { params.push(*a); params.push(*b);  },
                //Regularization / Normalization / Utility
                //Layer::Dropout(d) => params.push(*d),
                //Neuron-layers
                Layer::Dense(weights) =>
                {
                    for vec in weights.iter()
                    {
                        for val in vec.iter()
                        {
                            params.push(*val);
                        }
                    }
                },
                //rest does not have params (that have to/may be changed)
                _ => (),
            }
        }
        params
    }
    
    /// Set the layers' parameters (including LReLU factors) by a flat input.
    /// Used for evolution-strategies.
    /// Panics if params' size does not fit the layers
    pub fn set_params(&mut self, params:&[f64]) -> &mut Self
    {
        let mut iter = params.iter();
        for layer in self.layers.iter_mut()
        {
            match layer
            {
                //Activation functions
                //Layer::LReLU(factor) => *factor = *iter.next().expect("Vector params is not big enough!"),
                Layer::PReLU(factor) => *factor = *iter.next().expect("Vector params is not big enough!"),
                Layer::PELU(a, b) => { *a = *iter.next().expect("Vector params is not big enough!");
                                        *b = *iter.next().expect("Vector params is not big enough!"); },
                //Regularization / Normalization / Utility
                //Layer::Dropout(d) => *d = *iter.next().expect("Vector params is not big enough!"),
                //Neuron-layers
                Layer::Dense(weights) =>
                {
                    for vec in weights.iter_mut()
                    {
                        for val in vec.iter_mut()
                        {
                            *val = *iter.next().expect("Vector params is not big enough!");
                        }
                    }
                },
                //rest does not have params (that have to/may be changed)
                _ => (),
            }
        }
        self
    }
    
    /// Add a layer to the sequential model. Be sure to have appropriate parameters inside the layer, they are not checked!
    /// You can use specific add_layer_<layer> methods to get simple, correct creation of layers with parameters.
    pub fn add_layer(&mut self, layer:Layer) -> &mut Self
    {
        match &layer
        {
            Layer::Dense(weights) => self.num_outputs = weights.len(),
            _ => (),
        }
        self.layers.push(layer);
        self
    }
    
    /// Add a LReLU layer:
    /// factor = factor to apply to x < 0
    pub fn add_layer_lrelu(&mut self, factor:f64) -> &mut Self
    {
        let layer = Layer::LReLU(factor);
        self.layers.push(layer);
        self
    }
    
    /// Add a PReLU layer:
    /// factor = factor to apply to x < 0
    pub fn add_layer_prelu(&mut self, factor:f64) -> &mut Self
    {
        let layer = Layer::PReLU(factor);
        self.layers.push(layer);
        self
    }
    
    /// Add a PELU layer:
    /// a and b are the specific factors
    pub fn add_layer_pelu(&mut self, a:f64, b:f64) -> &mut Self
    {
        let layer = Layer::PELU(a, b);
        self.layers.push(layer);
        self
    }
    
    /// Add a Dropout layer:
    /// d = fraction of nodes to drop
    pub fn add_layer_dropout(&mut self, d:f64) -> &mut Self
    {
        if d < 0.0 || d >= 1.0
        {
            panic!("Inappropriate dropout parameter!");
        }
        
        let layer = Layer::Dropout(d);
        self.layers.push(layer);
        self
    }
    
    /// Add a Dense layer:
    /// neurons = number of neurons/units in the layer
    /// init = initializer to use
    pub fn add_layer_dense(&mut self, neurons:usize, init:Initializer) -> &mut Self
    {
        let weights = match init
        {
            Initializer::Glorot => gen_glorot(self.num_outputs, neurons),
            Initializer::He => gen_he(self.num_outputs, neurons),
            Initializer::Const(val) => vec![vec![val; self.num_outputs+1]; neurons],
        };
        self.num_outputs = neurons;
        let layer = Layer::Dense(weights);
        self.layers.push(layer);
        self
    }
    
    /// Do a forward pass through the model
    pub fn run(&self, input:&Vec<f64>) -> Vec<f64>
    {
        if input.len() != self.num_inputs
        {
            panic!("Incorrect input size!");
        }
        
        let mut result = input.clone();
        for layer in self.layers.iter()
        {
            match layer
            {
                //Activation functions
                Layer::Linear => result.iter_mut().for_each(|x| { *x = linear(*x); }),
                Layer::ReLU => result.iter_mut().for_each(|x| { *x = relu(*x); }),
                Layer::LReLU(factor) => result.iter_mut().for_each(|x| { *x = lrelu(*x, *factor); }),
                Layer::PReLU(factor) => result.iter_mut().for_each(|x| { *x = lrelu(*x, *factor); }),
                Layer::ELU => result.iter_mut().for_each(|x| { *x = elu(*x); }),
                Layer::PELU(a, b) => result.iter_mut().for_each(|x| { *x = pelu(*x, *a, *b); }),
                Layer::SELU => result.iter_mut().for_each(|x| { *x =selu(*x); }),
                Layer::Sigmoid => result.iter_mut().for_each(|x| { *x = sigmoid(*x); }),
                Layer::Tanh => result.iter_mut().for_each(|x| { *x = tanh(*x); }),
                Layer::Quadratic => result.iter_mut().for_each(|x| { *x = quadratic(*x); }),
                Layer::Cubic => result.iter_mut().for_each(|x| { *x = cubic(*x); }),
                Layer::ClipLinear => result.iter_mut().for_each(|x| { *x = clip_linear(*x); }),
                Layer::Gaussian => result.iter_mut().for_each(|x| { *x = gaussian(*x); }),
                Layer::SoftPlus => result.iter_mut().for_each(|x| { *x = softplus(*x); }),
                Layer::SoftMax => softmax(&mut result),
                
                //Regularization / Normalization / Utility
                Layer::Dropout(d) => apply_dropout(&mut result, *d),
                
                //Neuron-layers
                Layer::Dense(weights) => result = modified_matrix_dotprod(&weights, &result),
            }
        }
        result
    }
    
    /// Predict values (forward pass) for a vector of input data (Vec<input>):
    pub fn predict(&self, inputs:&Vec<Vec<f64>>) -> Vec<Vec<f64>>
    {
        let mut results = Vec::new();
        for input in inputs.iter()
        {
            let result = self.run(input);
            results.push(result);
        }
        results
    }
    
    /// Encodes the model as a JSON string.
    pub fn to_json(&self) -> String
    {
        serde_json::to_string(self).expect("Encoding JSON failed!")
    }

    /// Builds a new model from a JSON string.
    pub fn from_json(encoded:&str) -> Sequential
    {
        serde_json::from_str(encoded).expect("Decoding JSON failed!")
    }
    
    /// Saves the model to a file
    pub fn save(&self, file:&str) -> Result<(), std::io::Error>
    {
        let mut file = File::create(file)?;
        let json = self.to_json();
        file.write_all(json.as_bytes())?;
        Ok(())
    }
    
    /// Creates a model from a previously saved file
    pub fn load(file:&str) -> Result<Sequential, std::io::Error>
    {
        let mut file = File::open(file)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        Ok(Sequential::from_json(&json))
    }
    
    /// Calculate the error to a target set (Vec<(x, y)>):
    /// Mean squared error (for regression)
    /// Potentially ignores different vector lenghts!
    pub fn calc_mse(&self, target:&Vec<(Vec<f64>, Vec<f64>)>) -> f64
    {
        let mut avg_error = 0.0;
        for (x, y) in target.iter()
        {
            let pred = self.run(x);
            let mut metric = 0.0;
            for (yp, yt) in pred.iter().zip(y.iter())
            {
                let error = *yt - *yp;
                metric += error * error;
            }
            metric /= y.len() as f64;
            avg_error += metric;
        }
        avg_error /= target.len() as f64;
        avg_error
    }
    
    /// Calculate the error to a target set (Vec<(x, y)>):
    /// Root mean squared error (for regression)
    /// Potentially ignores different vector lenghts!
    pub fn calc_rmse(&self, target:&Vec<(Vec<f64>, Vec<f64>)>) -> f64
    {
        let mut avg_error = 0.0;
        for (x, y) in target.iter()
        {
            let pred = self.run(x);
            let mut metric = 0.0;
            for (yp, yt) in pred.iter().zip(y.iter())
            {
                let error = *yt - *yp;
                metric += error * error;
            }
            metric /= y.len() as f64;
            avg_error += metric.sqrt();
        }
        avg_error /= target.len() as f64;
        avg_error
    }
    
    /// Calculate the error to a target set (Vec<(x, y)>):
    /// Mean absolute error (for regression)
    /// Potentially ignores different vector lenghts!
    pub fn calc_mae(&self, target:&Vec<(Vec<f64>, Vec<f64>)>) -> f64
    {
        let mut avg_error = 0.0;
        for (x, y) in target.iter()
        {
            let pred = self.run(x);
            let mut metric = 0.0;
            for (yp, yt) in pred.iter().zip(y.iter())
            {
                let error = *yt - *yp;
                metric += error.abs();
            }
            metric /= y.len() as f64;
            avg_error += metric;
        }
        avg_error /= target.len() as f64;
        avg_error
    }
    
    /// Calculate the error to a target set (Vec<(x, y)>):
    /// Mean absolute percentage error (better don't use if target has 0 values) (for regression)
    /// Potentially ignores different vector lenghts!
    pub fn calc_mape(&self, target:&Vec<(Vec<f64>, Vec<f64>)>) -> f64
    {
        let mut avg_error = 0.0;
        for (x, y) in target.iter()
        {
            let pred = self.run(x);
            let mut metric = 0.0;
            for (yp, yt) in pred.iter().zip(y.iter())
            {
                let error = (*yt - *yp) / *yt;
                metric += error.abs();
            }
            metric *= 100.0 / y.len() as f64;
            avg_error += metric;
        }
        avg_error /= target.len() as f64;
        avg_error
    }
    
    /// Calculate the error to a target set (Vec<(x, y)>):
    /// logcosh (for regression)
    /// Potentially ignores different vector lenghts!
    pub fn calc_logcosh(&self, target:&Vec<(Vec<f64>, Vec<f64>)>) -> f64
    {
        let mut avg_error = 0.0;
        for (x, y) in target.iter()
        {
            let pred = self.run(x);
            let mut metric = 0.0;
            for (yp, yt) in pred.iter().zip(y.iter())
            {
                let error = *yt - *yp;
                metric += error.cosh().ln();
            }
            metric /= y.len() as f64;
            avg_error += metric;
        }
        avg_error /= target.len() as f64;
        avg_error
    }
    
    /// Calculate the error to a target set (Vec<(x, y)>):
    /// binary cross-entropy (be sure to use 0, 1 classifiers+labels) (for classification)
    /// Potentially ignores different vector lenghts!
    pub fn calc_binary_crossentropy(&self, target:&Vec<(Vec<f64>, Vec<f64>)>) -> f64
    {
        let mut avg_error = 0.0;
        for (x, y) in target.iter()
        {
            let pred = self.run(x);
            let mut metric = 0.0;
            for (yp, yt) in pred.iter().zip(y.iter())
            {
                let error = *yt * yp.ln() + (1.0 - *yt) * (1.0 - *yp).ln();
                metric += -error;
            }
            metric /= y.len() as f64;
            avg_error += metric;
        }
        avg_error /= target.len() as f64;
        avg_error
    }
    
    /// Calculate the error to a target set (Vec<(x, y)>):
    /// categorical cross-entropy (be sure to use 0, 1 classifiers+labels) (for classification)
    /// Potentially ignores different vector lenghts!
    pub fn calc_categorical_crossentropy(&self, target:&Vec<(Vec<f64>, Vec<f64>)>) -> f64
    {
        let mut avg_error = 0.0;
        for (x, y) in target.iter()
        {
            let pred = self.run(x);
            let mut metric = 0.0;
            for (yp, yt) in pred.iter().zip(y.iter())
            {
                let error = *yt * (*yp).ln();
                metric += -error;
            }
            avg_error += metric;
        }
        avg_error /= target.len() as f64;
        avg_error
    }
    
    /// Calculate the error to a target set (Vec<(x, y)>):
    /// hinge loss (be sure to use 1, -1 classifiers+labels) (for classification)
    /// Potentially ignores different vector lenghts!
    pub fn calc_hingeloss(&self, target:&Vec<(Vec<f64>, Vec<f64>)>) -> f64
    {
        let mut avg_error = 0.0;
        for (x, y) in target.iter()
        {
            let pred = self.run(x);
            let mut metric = 0.0;
            for (yp, yt) in pred.iter().zip(y.iter())
            {
                let error = 1.0 - *yt * *yp;
                metric += error.max(0.0);
            }
            metric /= y.len() as f64;
            avg_error += metric;
        }
        avg_error /= target.len() as f64;
        avg_error
    }
}



//helper functions
/// Generate a vector of random numbers with 0 mean and std std, normally distributed.
fn gen_rnd_vec(n:usize, std:f64) -> Vec<f64>
{
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, std);
    normal.sample_iter(&mut rng).take(n).collect()
}

/// Generate parameters based on Glorot initialization
fn gen_glorot(n_in:usize, n_out:usize) -> Vec<Vec<f64>>
{
    let std = (2.0 / (n_in + n_out) as f64).sqrt();
    let mut weights = Vec::new();
    for _ in 0..n_out
    {
        weights.push(gen_rnd_vec(n_in + 1, std));
    }
    weights
}

/// Generate parameters based on He initialization
fn gen_he(n_in:usize, n_out:usize) -> Vec<Vec<f64>>
{
    let std = (2.0 / n_in as f64).sqrt();
    let mut weights = Vec::new();
    for _ in 0..n_out
    {
        weights.push(gen_rnd_vec(n_in + 1, std));
    }
    weights
}

/// Apply dropout to a layer. d = fraction of nodes to be dropped
fn apply_dropout(layer:&mut Vec<f64>, d:f64)
{
    if d == 0.0
    { //allow zero dropout to allow later change, but do nothing here
        return;
    }
    // set nodes to zero
    let num = (d * layer.len() as f64) as usize;
    let mut rng = rand::thread_rng();
    for _ in 0..num
    {
        let i = rng.gen::<usize>() % layer.len();
        layer[i] = 0.0;
    }
    //divide other nodes by probability to adapt variance
    layer.iter_mut().for_each(|x| { *x /= d; })
}

/// Calculate layer results with bias from weight
/// If weights matrix is empty, result will be empty (indicating zero nodes)
fn modified_matrix_dotprod(weights:&Vec<Vec<f64>>, values:&Vec<f64>) -> Vec<f64>
{
    let mut result = Vec::new();
    for node in weights.iter()
    {
        let mut iter = node.iter();
        let mut sum = *iter.next().expect("Empty weights! (Bias)");
        for (weight, value) in iter.zip(values.iter()) //panics if weights do not have the correct shape
        {
            sum += weight * value;
        }
        result.push(sum);
    }
    result
}


//activiation functions
fn linear(x:f64) -> f64
{
    x
}

fn relu(x:f64) -> f64
{
    x.max(0.0)
}

fn lrelu(x:f64, factor:f64) -> f64
{
    if x < 0.0
    {
        factor * x
    }
    else
    {
        x
    }
}

fn elu(x:f64) -> f64
{
    if x < 0.0
    {
        x.exp()
    }
    else
    {
        x
    }
}

fn pelu(x:f64, a:f64, b:f64) -> f64
{
    if x < 0.0
    {
        a * (x / b).exp() - a
    }
    else
    {
        (a / b) * x
    }
}

fn selu(x:f64) -> f64
{
    SELU_LAMBDA * if x < 0.0
    {
        SELU_ALPHA * x.exp() - SELU_ALPHA
    }
    else
    {
        x
    }
}

fn sigmoid(x:f64) -> f64
{
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x:f64) -> f64
{
    x.tanh()
}

fn quadratic(x:f64) -> f64
{
    x * x
}

fn cubic(x:f64) -> f64
{
    x * x * x
}

fn clip_linear(x:f64) -> f64
{
    x.min(1.0).max(-1.0)
}

fn gaussian(x:f64) -> f64
{
    (-x * x).exp()
}

fn softplus(x:f64) -> f64
{
    (1.0 + x.exp()).ln()
}

fn softmax(arr:&mut Vec<f64>)
{
    let norm:f64 = arr.iter().map(|x| x.exp()).sum();
    for val in arr.iter_mut()
    {
        *val = val.exp() / norm;
    }
}
