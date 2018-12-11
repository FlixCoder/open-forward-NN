//! CIFAR10 example.
#![allow(clippy::missing_docs_in_private_items, clippy::print_stdout, clippy::unwrap_used)]

use std::{fs::File, io::prelude::*, path::Path, time::Instant};

use esopt::*;
use ofnn::{Float, *};
use rand::prelude::*;

const BATCHSIZE: usize = 32; //number of items to form a batch inside evaluation

const NOISE_STD: Float = 0.025; //standard deviation of noise to mutate parameters and generate meta population
const POPULATION: usize = 500; //number of double-sided samples forming the pseudo/meta population

//TODO:
//find optimizer error, why worse than backprop?

fn main() {
	std::fs::create_dir_all("./model").ok();
	//NN model
	let mut model = Sequential::load("./model/cifar10.nn").unwrap_or_else(|_| {
		//else construct it
		let mut model = Sequential::new(3072);
		model
			.add_layer_dense(384, Initializer::Const(0.0)) //Initializer::Glorot ?
			.add_layer(Layer::SELU)
			.add_layer_dense(10, Initializer::Const(0.0)) //Initializer::He ?
			.add_layer(Layer::SoftMax);
		model
	});

	//create the evaluator with training data
	let eval = CIFAR10Evaluator::new(model.clone(), "cifar-10-binary/", true);

	//create or load optimizer
	let mut opt = Lookahead::<RAdam>::load("./model/optimizer.json")
		.unwrap_or_else(|_| Lookahead::new(RAdam::new()));
	opt.set_k(10);
	opt.get_opt_mut().set_lr(0.005).set_lambda(0.0); //0.0 or 0.001
	let iterations = opt.get_t();

	//evolutionary optimizer (for more details about it, see the git repository of
	// it)
	let mut opt = ES::new(opt, eval);
	opt.set_params(model.get_params()).set_std(NOISE_STD).set_samples(POPULATION);

	//show initial scores
	println!("Initial results on test set:");
	let mut tester = CIFAR10Evaluator::new(model.clone(), "cifar-10-binary/", false); //test data
	tester.print_metrics();

	//training: track the optimizer's results
	println!("Beginning training..");
	println!();
	let time = Instant::now();
	for i in 0..10 {
		//10 times
		//optimize for n steps
		let n = 100;
		let res = opt.optimize_ranked_par(n);

		//save results
		model.set_params(opt.get_params());
		model.save("./model/cifar10.nn").ok();
		opt.get_opt().save("./model/optimizer.json").ok();

		//display progress
		println!("After {} iteratios:", iterations + (i + 1) * n);
		println!("Score: {}", res.0);
		println!("Gradnorm: {}", res.1);
		println!();
	}
	let elapsed = time.elapsed();
	let sec = (elapsed.as_secs() as f64) + (f64::from(elapsed.subsec_nanos()) / 1_000_000_000.0);
	println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
	println!();

	//save trained model and estimate and display results
	model.set_params(opt.get_params());
	model.save("./model/cifar10.nn").ok();
	opt.get_opt().save("./model/optimizer.json").ok();

	println!("Final results on test data:");
	tester.set_model(model.clone());
	tester.print_metrics();

	//clean up
	//std::fs::remove_file("./model/cifar10.nn").ok();
	//std::fs::remove_file("./model/optimizer.json").ok();
}

/// Function to pass to map() for pixel normalization
fn image_normalize(x: u8) -> Float {
	(Float::from(x) / 255.0 - 0.5) * 2.0 * 1.6 //map [0, 255] to [-1.6, 1.6]
}

/// Loads a CIFAR10 file and returns normalized, categorized data in a
/// Result<(X, Y)>
#[allow(clippy::type_complexity)]
fn load_cifar10(filename: &Path) -> std::io::Result<(Vec<Vec<Float>>, Vec<Vec<Float>>)> {
	let mut file = File::open(filename)?;
	let mut x = Vec::new();
	let mut y = Vec::new();
	let mut buffer = [0_u8; 3073];

	for _i in 0..10000 {
		file.read_exact(&mut buffer)?;
		y.push(to_categorical(10, buffer[0]));
		let data: Vec<Float> = buffer[1..].iter().copied().map(image_normalize).collect();
		x.push(data);
	}

	Ok((x, y))
}

/// Translates integer labels into categorical label arrays.
fn to_categorical(classes: u8, label: u8) -> Vec<Float> {
	let mut vec = vec![0.0; classes as usize];
	vec[label as usize] = 1.0;
	vec
}

/// Return argmax of vector
fn argmax(vec: &[Float]) -> usize {
	let mut argmax = 0;
	let mut max = std::f64::NEG_INFINITY as Float;
	for (i, val) in vec.iter().enumerate() {
		if *val >= max {
			max = *val;
			argmax = i;
		}
	}
	argmax
}

#[derive(Clone)]
struct CIFAR10Evaluator {
	model: Sequential,
	data: (Vec<Vec<Float>>, Vec<Vec<Float>>),
	seed: u64,
}

impl CIFAR10Evaluator {
	pub fn new(model: Sequential, folder: &str, train: bool) -> CIFAR10Evaluator {
		let path = Path::new(folder);
		let mut data;
		if train {
			data = load_cifar10(&path.join("data_batch_1.bin")).unwrap();
			let mut tmp = load_cifar10(&path.join("data_batch_2.bin")).unwrap();
			data.0.append(&mut tmp.0);
			data.1.append(&mut tmp.1);
			let mut tmp = load_cifar10(&path.join("data_batch_3.bin")).unwrap();
			data.0.append(&mut tmp.0);
			data.1.append(&mut tmp.1);
			let mut tmp = load_cifar10(&path.join("data_batch_4.bin")).unwrap();
			data.0.append(&mut tmp.0);
			data.1.append(&mut tmp.1);
			let mut tmp = load_cifar10(&path.join("data_batch_5.bin")).unwrap();
			data.0.append(&mut tmp.0);
			data.1.append(&mut tmp.1);
		} else {
			data = load_cifar10(&path.join("test_batch.bin")).unwrap();
		}
		let seed = thread_rng().next_u64() % (std::u64::MAX - 50000); //prevent overflow when adding the index/iterations
		CIFAR10Evaluator { model, data, seed }
	}

	pub fn set_model(&mut self, model: Sequential) {
		self.model = model;
	}

	pub fn print_metrics(&mut self) {
		//compute predicitions for whole data
		let pred = self.model.predict(&self.data.0);

		//calculate metrics
		let loss = losses::categorical_crossentropy(&pred, &self.data.1);
		let mut acc = 0.0;
		for (p, t) in pred.iter().zip(self.data.1.iter()) {
			let select = argmax(p);
			if (t[select] - 1.0).abs() < Float::EPSILON {
				acc += 1.0;
			}
		}
		acc *= 100.0 / pred.len() as Float;

		//display results
		println!("Loss: {}", loss);
		println!("Accuracy: {:6.3}%", acc);
	}
}

impl Evaluator for CIFAR10Evaluator {
	//make the model repeat numbers from two iterations ago
	fn eval_train(&self, params: &[Float], index: usize) -> Float {
		let mut local = self.model.clone();
		local.set_params(params);

		//every parameter pertubation uses the same training data, but every iteration
		// uses different
		let mut rng = SmallRng::seed_from_u64(self.seed + index as u64);
		let start = rng.gen::<usize>() % (self.data.0.len() - BATCHSIZE); //not really uniform, but suffices
		let end = start + BATCHSIZE;

		let pred = local.predict(&self.data.0[start..end]);
		let loss = -losses::categorical_crossentropy(&pred, &self.data.1[start..end]);
		let reg =
			-0.1 * params.iter().fold(0.0, |acc, e| acc + e.abs().sqrt()) / (params.len() as Float); //factor * L0.5 regularization
		loss + reg
	}

	fn eval_test(&self, params: &[Float]) -> Float {
		//fast stochastic loss estimation like in training, but with the same data for
		// all steps to track changes self.eval_train(params, 49999) //use index greater
		// than can be used during training to possibly yield seperate test data
		// (constant)

		//slower, but complete training set metrics
		let mut local = self.model.clone();
		local.set_params(params);

		//compute predicitions for whole data
		let pred = local.predict(&self.data.0);

		//calculate metrics
		let loss = losses::categorical_crossentropy(&pred, &self.data.1);
		let mut acc = 0.0;
		for (p, t) in pred.iter().zip(self.data.1.iter()) {
			let select = argmax(p);
			if (t[select] - 1.0).abs() < Float::EPSILON {
				acc += 1.0;
			}
		}
		acc *= 100.0 / pred.len() as Float;

		//return accurracy.loss (is used for return value only)
		acc.round() + loss / 10.0
	}
}
