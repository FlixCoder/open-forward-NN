//! Simple categorical example.
#![allow(clippy::missing_docs_in_private_items, clippy::print_stdout, clippy::unwrap_used)]

use esopt::*;
use ofnn::{Float, *};

fn main() {
	//training data: XOR
	let x = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
	let y = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];

	//NN model
	let mut model = Sequential::new(2); //input size = 2
	model
		.add_layer_dense(3, Initializer::He) //add hidden dense layer with 3 neurons, init with He
		.add_layer_prelu(0.05) //add lrelu activation with initial factor 0.3
		.add_layer_dense(2, Initializer::Glorot) //add output dense layer with only 1 output, init with Glorot
		.add_layer(Layer::SoftMax); //add sigmoid activation

	//create the evaluator
	let eval = NNEvaluator::new(model.clone(), x.clone(), y);

	//evolutionary optimizer (for more details about it, see the git repository of
	// it)
	let mut adam = Adam::new();
	adam.set_lr(0.25).set_lambda(0.01);
	let mut opt = ES::new(adam, eval); //learning rate, weight decay
	opt.set_params(model.get_params()).set_std(0.1).set_samples(50);

	//training: track the optimizer's results
	for i in 0..5 {
		let n = 5;
		let res = opt.optimize(n); //optimize for n steps
		println!("After {} iteratios:", (i + 1) * n);
		println!("Loss: {}", -res.0); //negative score
		println!("Gradnorm: {}", res.1);
		println!();
	}

	//display and save results
	model.set_params(opt.get_params());
	println!("PReLU factor: {:?}", model.get_layers()[1]);
	let pred = model.predict(&x);
	println!("Prediction on {:?}: {}", x[0], pred[0][1]);
	println!("Prediction on {:?}: {}", x[1], pred[1][1]);
	println!("Prediction on {:?}: {}", x[2], pred[2][1]);
	println!("Prediction on {:?}: {}", x[3], pred[3][1]);
}

#[derive(Clone)]
struct NNEvaluator {
	model: Sequential,
	x: Vec<Vec<Float>>,
	y: Vec<Vec<Float>>,
}

impl NNEvaluator {
	pub fn new(model: Sequential, x: Vec<Vec<Float>>, y: Vec<Vec<Float>>) -> NNEvaluator {
		NNEvaluator { model, x, y }
	}
}

impl Evaluator for NNEvaluator {
	fn eval_test(&self, params: &[Float]) -> Float {
		let mut local = self.model.clone();
		local.set_params(params);
		let pred = local.predict(&self.x);
		let mut score = -losses::categorical_crossentropy(&pred, &self.y);
		if score.is_nan() {
			// set error to 0 when error is nan (happens, when model is already perfect)
			score = 0.0;
		} //returning NaN destroys all parameters!
		score
	}

	fn eval_train(&self, params: &[Float], _: usize) -> Float {
		self.eval_test(params)
	}
}
