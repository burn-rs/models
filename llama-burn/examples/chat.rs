use std::time::Instant;

use burn::tensor::{backend::Backend, Device};
use clap::Parser;
use llama_burn::{
    llama::{Llama, LlamaConfig},
    sampling::{Sampler, TopP},
    tokenizer::Tokenizer,
};

#[cfg(feature = "llama3")]
use clap::ValueEnum;

const DEFAULT_PROMPT: &str = "How many helicopters can a human eat in one sitting?";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Config {
    /// Top-p probability threshold.
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Temperature value for controlling randomness in sampling.
    #[arg(long, default_value_t = 0.6)]
    temperature: f64,

    /// Maximum sequence length for input text.
    #[arg(long, default_value_t = 128)]
    max_seq_len: usize,

    /// The number of new tokens to generate (i.e., the number of generation steps to take).
    #[arg(long, short = 'n', default_value_t = 65)]
    sample_len: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// The input prompt.
    #[arg(short, long, default_value_t = String::from(DEFAULT_PROMPT))]
    prompt: String,

    /// The Llama 3 model version.
    #[cfg(feature = "llama3")]
    #[arg(long, default_value = "llama-3.2-1b-instruct")]
    model_version: Llama3,
}

#[cfg(feature = "llama3")]
#[derive(Clone, Debug, ValueEnum)]
/// Llama-3 model variants to load.
enum Llama3 {
    /// Llama-3-8B-Instruct.
    #[value(name = "llama-3-8b-instruct")]
    V3Instruct,
    /// Llama-3.1-8B-Instruct.
    #[value(name = "llama-3.1-8b-instruct")]
    V31Instruct,
    /// Llama-3.2-1B-Instruct.
    #[value(name = "llama-3.2-1b-instruct")]
    V321bInstruct,
    /// Llama-3.2-3B-Instruct.
    #[value(name = "llama-3.2-3b-instruct")]
    V323bInstruct,
}

pub fn generate<B: Backend, T: Tokenizer>(
    llama: &mut Llama<B, T>,
    prompt: &str,
    sample_len: usize,
    temperature: f64,
    sampler: &mut Sampler,
) {
    let now = Instant::now();
    let generated = llama.generate(prompt, sample_len, temperature, sampler);
    let elapsed = now.elapsed().as_secs();

    println!("> {}\n", generated.text);
    println!(
        "{} tokens generated ({:.4} tokens/s)\n",
        generated.tokens,
        generated.tokens as f64 / generated.time
    );

    println!(
        "Generation completed in {}m{}s",
        (elapsed / 60),
        elapsed % 60
    );
}

pub fn chat<B: Backend>(args: Config, device: Device<B>) {
    let mut prompt = args.prompt;

    // Sampling strategy
    let mut sampler = if args.temperature > 0.0 {
        Sampler::TopP(TopP::new(args.top_p, args.seed))
    } else {
        Sampler::Argmax
    };

    #[cfg(feature = "tiny")]
    {
        // TinyLlama-1.1B Chat v1.0
        let mut llama = LlamaConfig::tiny_llama_pretrained::<B>(args.max_seq_len, &device).unwrap();
        println!("Processing prompt: {}", prompt);

        // Prompt formatting for chat model
        prompt = format!(
            "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        );

        generate(
            &mut llama,
            &prompt,
            args.sample_len,
            args.temperature,
            &mut sampler,
        );
    }

    #[cfg(feature = "llama3")]
    {
        // Llama-3-8B-Instruct or Llama-3.1-8B-Instruct
        let mut llama = match args.model_version {
            Llama3::V3Instruct => {
                LlamaConfig::llama3_8b_pretrained::<B>(args.max_seq_len, &device).unwrap()
            }
            Llama3::V31Instruct => {
                LlamaConfig::llama3_1_8b_pretrained::<B>(args.max_seq_len, &device).unwrap()
            }
            Llama3::V321bInstruct => {
                LlamaConfig::llama3_2_1b_pretrained::<B>(args.max_seq_len, &device).unwrap()
            }
            Llama3::V323bInstruct => {
                LlamaConfig::llama3_2_3b_pretrained::<B>(args.max_seq_len, &device).unwrap()
            }
        };
        println!("Processing prompt: {}", prompt);

        // Prompt formatting for chat model
        prompt = format!(
            "<|start_header_id|>system<|end_header_id|>\n\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );

        generate(
            &mut llama,
            &prompt,
            args.sample_len,
            args.temperature,
            &mut sampler,
        );
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use super::*;
    use burn::{
        backend::{libtorch::LibTorchDevice, LibTorch},
        tensor::f16,
    };

    pub fn run(args: Config) {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Mps;

        chat::<LibTorch<f16>>(args, device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use super::*;
    use burn::backend::{libtorch::LibTorchDevice, LibTorch};

    pub fn run(args: Config) {
        let device = LibTorchDevice::Cpu;

        chat::<LibTorch>(args, device);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run(args: Config) {
        let device = WgpuDevice::default();

        chat::<Wgpu>(args, device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::{
        backend::{cuda_jit::CudaDevice, CudaJit},
        tensor::f16,
    };

    pub fn run(args: Config) {
        let device = CudaDevice::default();

        chat::<CudaJit<f16, i32>>(args, device);
    }
}

pub fn main() {
    // Parse arguments
    let args = Config::parse();

    #[cfg(feature = "tch-gpu")]
    tch_gpu::run(args);
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run(args);
    #[cfg(feature = "wgpu")]
    wgpu::run(args);
    #[cfg(feature = "cuda")]
    cuda::run(args);
}
