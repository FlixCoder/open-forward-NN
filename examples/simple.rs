//! Simple example.
#![allow(clippy::missing_docs_in_private_items, clippy::print_stdout, clippy::unwrap_used)]

use esopt::*;
use ofnn::{Float, *};

fn main() {
	//training data: XOR
	let target = vec![
		(vec![0.0, 0.0], vec![0.0]),
		(vec![0.0, 1.0], vec![1.0]),
		(vec![1.0, 0.0], vec![1.0]),
		(vec![1.0, 1.0], vec![0.0]),
	];

	//NN model
	let mut model = Sequential::load("test.nn").unwrap_or_else(|_| {
		//else construct it
		let mut model = Sequential::new(2); //input size = 2
		model
			.add_layer_dense(3, Initializer::He) //add hidden dense layer with 3 neurons, init with He
			.add_layer_prelu(0.05) //add lrelu activation with initial factor 0.3
			.add_layer_dense(1, Initializer::Glorot) //add output dense layer with only 1 output, init with Glorot
			.add_layer(Layer::Sigmoid); //add sigmoid activation
		model
	});

	//create the evaluator
	let eval = NNEvaluator::new(model.clone(), target.clone());

	//evolutionary optimizer (for more details about it, see the git repository of
	// it)
	let mut opt = ES::new_with_adam(eval, 0.25, 0.01); //learning rate, weight decay
												   //opt.get_opt_mut().set_beta2(0.99); //set adam's beta2
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
	model.save("test.nn").ok();
	println!("PReLU factor: {:?}", model.get_layers()[1]);
	println!("Prediction on {:?}: {}", target[0].0, model.run(&target[0].0)[0]);
	println!("Prediction on {:?}: {}", target[1].0, model.run(&target[1].0)[0]);
	println!("Prediction on {:?}: {}", target[2].0, model.run(&target[2].0)[0]);
	println!("Prediction on {:?}: {}", target[3].0, model.run(&target[3].0)[0]);

	//clean up
	std::fs::remove_file("test.nn").ok();
}

#[derive(Clone)]
struct NNEvaluator {
	model: Sequential,
	target: Vec<(Vec<Float>, Vec<Float>)>,
}

impl NNEvaluator {
	pub fn new(model: Sequential, target: Vec<(Vec<Float>, Vec<Float>)>) -> NNEvaluator {
		NNEvaluator { model, target }
	}
}

impl Evaluator for NNEvaluator {
	fn eval_test(&self, params: &[Float]) -> Float {
		let mut local = self.model.clone();
		local.set_params(params);
		let mut score = -local.calc_binary_crossentropy(&self.target);
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
