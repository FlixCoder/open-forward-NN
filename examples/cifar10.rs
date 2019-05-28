extern crate esopt;
extern crate ofnn;
extern crate rand;

use ofnn::*;
use esopt::*;
use ofnn::Float;
use rand::prelude::*;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

const BATCHSIZE:usize = 32; //number of items to form a batch inside evaluation

const NOISE_STD:Float = 0.025; //standard deviation of noise to mutate parameters and generate meta population
const POPULATION:usize = 500; //number of double-sided samples forming the psueod/meta population

//TODO:
//try L0.5 regularization (avg sqrt)
//try sequential batches?
//random/zero/convolution-like init?


fn main()
{
    std::fs::create_dir("./model").ok();
    //NN model
    let loaded = Sequential::load("./model/cifar10.nn");
    let mut model = if loaded.is_ok()
        { //try loaded model first
            loaded.unwrap()
        }
        else
        { //else construct it
            let mut model = Sequential::new(3072);
            model.add_layer_dense(384, Initializer::Const(0.0)) //better than random init (maybe due to convolution)
                .add_layer(Layer::SELU)
                .add_layer_dense(10, Initializer::Const(0.0)) //better than random init (maybe due to convolution)
                .add_layer(Layer::SoftMax);
            model
        };
    
    //create the evaluator with training data
    let eval = CIFAR10Evaluator::new(model.clone(), "cifar-10-binary/", true);
    
    //create or load optimizer
    let loaded = Adamax::load("./model/optimizer.json");
    let mut opt = if loaded.is_ok()
        {
            loaded.unwrap()
        }
        else
        {
            Adamax::new()
        };
    opt.set_lr(0.0025)
        //.set_lambda(0.001)
        .set_beta1(0.95)
        .set_beta2(0.999);
    let iterations = opt.get_t();
    
    //evolutionary optimizer (for more details about it, see the git repository of it)
    let mut opt = ES::new(opt, eval);
    opt.set_params(model.get_params())
        .set_std(NOISE_STD)
        .set_samples(POPULATION);
    
    //show initial scores
    println!("Initial results on test set:");
    let mut tester = CIFAR10Evaluator::new(model.clone(), "cifar-10-binary/", false); //test data
    tester.print_metrics();
    
    //training: track the optimizer's results
    println!("Beginning training..");
    println!("");
    let time = Instant::now();
    for i in 0..10
    { //10 times
        //optimize for n steps
        let n = 100;
        let res = opt.optimize_par(n); //opt.optimize_ranked_par(n);
        
        //save results
        model.set_params(opt.get_params());
        model.save("./model/cifar10.nn").ok();
        opt.get_opt().save("./model/optimizer.json").ok();
        
        //display progress
        println!("After {} iteratios:", iterations + (i+1) * n);
        println!("Score: {}", res.0);
        println!("Gradnorm: {}", res.1);
        println!("");
    }
    let elapsed = time.elapsed();
    let sec = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64 / 1000_000_000.0);
    println!("Time: {} min {:.3} s", (sec / 60.0).floor(), sec % 60.0);
    println!("");
    
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
fn image_normalize(x:&u8) -> Float
{
    (*x as Float / 255.0 - 0.5) * 2.0 * 1.6 //map [0, 255] to [-1.6, 1.6]
}

/// Loads a CIFAR10 file and returns normalized, categorized data in a Result<(X, Y)>
fn load_cifar10(filename:&Path) -> std::io::Result<(Vec<Vec<Float>>, Vec<Vec<Float>>)>
{
    let mut file = File::open(filename)?;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut buffer = [0u8; 3073];
    
    for _i in 0..10000
    {
        file.read_exact(&mut buffer)?;
        y.push(to_categorical(10, buffer[0]));
        let data:Vec<Float> = buffer[1..].iter().map(image_normalize).collect();
        x.push(data);
    }
    
    Ok((x, y))
}

/// Translates integer labels into categorical label arrays.
fn to_categorical(classes:u8, label:u8) -> Vec<Float>
{
    let mut vec = vec![0.0; classes as usize];
    vec[label as usize] = 1.0;
    vec
}

/// Return argmax of vector
fn argmax(vec:&[Float]) -> usize
{
    let mut argmax = 0;
    let mut max = std::f64::NEG_INFINITY as Float;
    for (i, val) in vec.iter().enumerate()
    {
        if *val >= max
        {
            max = *val;
            argmax = i;
        }
    }
    argmax
}


#[derive(Clone)]
struct CIFAR10Evaluator
{
    model:Sequential,
    data:(Vec<Vec<Float>>, Vec<Vec<Float>>),
    seed:u64,
}

impl CIFAR10Evaluator
{
    pub fn new(model:Sequential, folder:&str, train:bool) -> CIFAR10Evaluator
    {
        let path = Path::new(folder);
        let mut data;
        if train
        {
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
        }
        else
        {
            data = load_cifar10(&path.join("test_batch.bin")).unwrap();
        }
        let seed = thread_rng().next_u64() % (std::u64::MAX - 50000); //prevent overflow when adding the index/iterations
        CIFAR10Evaluator { model: model, data: data, seed: seed }
    }
    
    pub fn set_model(&mut self, model:Sequential)
    {
        self.model = model;
    }
    
    pub fn print_metrics(&mut self)
    {
        //compute predicitions for whole data
        let pred = self.model.predict(&self.data.0);
        
        //calculate metrics
        let loss = losses::categorical_crossentropy(&pred, &self.data.1);
        let mut acc = 0.0;
        for (p, t) in pred.iter().zip(self.data.1.iter())
        {
            let select = argmax(p);
            if t[select] == 1.0
            {
                acc += 1.0;
            }
        }
        acc *= 100.0 / pred.len() as Float;
        
        //display results
        println!("Loss: {}", loss);
        println!("Accuracy: {:6.3}%", acc);
    }
}

impl Evaluator for CIFAR10Evaluator
{
    //make the model repeat numbers from two iterations ago
    fn eval_train(&self, params:&[Float], index:usize) -> Float
    {
        let mut local = self.model.clone();
        local.set_params(params);
        
        //every parameter pertubation uses the same training data, but every iteration uses different
        let mut rng = SmallRng::seed_from_u64(self.seed + index as u64);
        let start = rng.gen::<usize>() % (self.data.0.len() - BATCHSIZE); //not really uniform, but suffices
        let end = start + BATCHSIZE;
        
        let pred = local.predict(&self.data.0[start..end]);
        -losses::categorical_crossentropy(&pred, &self.data.1[start..end])
    }
    
    fn eval_test(&self, params:&[Float]) -> Float
    {
        //fast stochastic loss estimation like in training, but with the same data for all steps to track changes
        //self.eval_train(params, 49999) //use index greater than can be used during training to possibly yield seperate test data (constant)
        
        //slower, but complete training set metrics
        let mut local = self.model.clone();
        local.set_params(params);
        
        //compute predicitions for whole data
        let pred = local.predict(&self.data.0);
        
        //calculate metrics
        let loss = losses::categorical_crossentropy(&pred, &self.data.1);
        let mut acc = 0.0;
        for (p, t) in pred.iter().zip(self.data.1.iter())
        {
            let select = argmax(p);
            if t[select] == 1.0
            {
                acc += 1.0;
            }
        }
        acc *= 100.0 / pred.len() as Float;
        
        //return accurracy.loss (is used for return value only)
        acc.round() + loss / 10.0
    }
}
