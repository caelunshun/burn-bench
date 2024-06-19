use burn::backend::libtorch::LibTorchDevice;
use burn::backend::LibTorch;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{activation, bf16};
use mimalloc::MiMalloc;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::time::Instant;
use std::{array, hint, thread};

#[global_allocator]
static ALLOCATOR: MiMalloc = MiMalloc;

const INPUT_SIZE: usize = 16;
const OUTPUT_SIZE: usize = 10;

#[derive(Debug, Module)]
struct Model<B: Backend> {
    ln: Vec<Linear<B>>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            ln: vec![
                LinearConfig::new(INPUT_SIZE, 2048).init(device),
                LinearConfig::new(2048, 2048).init(device),
                LinearConfig::new(2048, 2048).init(device),
                LinearConfig::new(2048, 2048).init(device),
                LinearConfig::new(2048, 2048).init(device),
                LinearConfig::new(2048, 2048).init(device),
                LinearConfig::new(2048, 2048).init(device),
                LinearConfig::new(2048, 128).init(device),
                LinearConfig::new(128, OUTPUT_SIZE).init(device),
            ],
        }
    }

    pub fn forward(&self, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        for (i, ln) in self.ln.iter().enumerate() {
            x = ln.forward(x);
            if i != self.ln.len() - 1 {
                x = activation::relu(x);
            }
        }
        x
    }
}

fn main() {
    tch::maybe_init_cuda();

    println!("f32: {}", matmul::<LibTorch>());
    println!("bf16: {}", matmul::<LibTorch<bf16>>());

    //let data = Vec::leak(generate_data());

    //do_benchmark::<LibTorch>("f32_gpu", &LibTorchDevice::Cuda(0), data);
    //do_benchmark::<LibTorch<bf16>>("bf16_gpu", &LibTorchDevice::Cuda(0), data);
    //do_benchmark::<LibTorch>("f32_cpu", &LibTorchDevice::Cpu, data);
    //do_benchmark::<LibTorch<bf16>>("bf16_cpu", &LibTorchDevice::Cpu, data);
}

fn do_benchmark<B: Backend<Device = LibTorchDevice>>(
    name: &str,
    device: &B::Device,
    data: &'static [([f32; INPUT_SIZE], [f32; OUTPUT_SIZE])],
) {
    // warm up
    //do_iteration::<B>(device, data);

    let time = time(|| do_iteration::<B>(device, data));
    println!("{name}: {time}");
}

fn time(f: impl FnOnce()) -> String {
    let start = Instant::now();
    f();
    format!("{:.2?}", start.elapsed())
}

fn do_iteration<B: Backend<Device = LibTorchDevice>>(
    device: &B::Device,
    data: &'static [([f32; INPUT_SIZE], [f32; OUTPUT_SIZE])],
) {
    let model = Model::<B>::new(device);
    let batch_size = 4096;

    let (sender, receiver) = flume::bounded(128);

    let device2 = device.clone();
    thread::spawn(move || {
        for batch in data.chunks(batch_size) {
            let inputs = batch
                .iter()
                .map(|(x, _)| Tensor::from_floats(*x, &LibTorchDevice::Cpu))
                .collect::<Vec<_>>();
            let inputs = Tensor::stack(inputs, 0).fork(&device2);
            let targets = batch
                .iter()
                .map(|(_, y)| Tensor::from_floats(*y, &LibTorchDevice::Cpu))
                .collect::<Vec<_>>();
            let targets = Tensor::stack(targets, 0).fork(&device2);
            sender.send((inputs, targets)).unwrap();
        }
    });

    thread::scope(|s| {
        for _ in 0..4 {
            let model = model.clone();
            let receiver = receiver.clone();
            s.spawn(move || {
                let mut total_loss = 0.0f64;
                for (inputs, targets) in receiver {
                    let logits = model.forward(inputs);
                    let loss = MseLoss::new().forward(logits, targets, Reduction::Mean);
                    total_loss += loss.into_data().convert::<f64>().value[0];
                }
                hint::black_box(total_loss);
            });
        }
    });
}

fn generate_data() -> Vec<([f32; INPUT_SIZE], [f32; OUTPUT_SIZE])> {
    const NUM_ENTRIES: usize = 8 * 1024 * 1024;
    (0..NUM_ENTRIES)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            (
                array::from_fn(|_| StandardNormal.sample(&mut rng)),
                array::from_fn(|_| StandardNormal.sample(&mut rng)),
            )
        })
        .collect()
}

const DIM: usize = 16_384;

fn matmul<B: Backend<Device=LibTorchDevice>>() -> String {
    let device = LibTorchDevice::Cuda(0);
    let a = random_matrix();
    let b= random_matrix();
    let a= Tensor::<B, 2>::from_floats(Data::new(a, Shape::new([DIM, DIM])), &device );
    let b = Tensor::from_floats(Data::new(b, Shape::new([DIM, DIM])), &device );

    let start = Instant::now();
    const ITERS: usize = 32;
    let mut d = Tensor::zeros([DIM, DIM], &device);
    for _ in 0..ITERS {
        let c = a.clone().matmul(b.clone());
        d = d + c;
    }

    hint::black_box(d.into_data());

    let elapsed = start.elapsed() / ITERS as u32;
    format!("{:.2?} per iter", elapsed)
}

fn random_matrix() -> Vec<f32> {
    (0..crate::DIM * crate::DIM)
        .into_par_iter()
        .map(|_| StandardNormal.sample(&mut rand::thread_rng()))
        .collect::<Vec<_>>()
}
