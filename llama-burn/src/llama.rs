use std::time::Instant;

use burn::{
    config::Config,
    module::Module,
    nn::{RotaryEncoding, RotaryEncodingConfig},
    record::{FileRecorder, HalfPrecisionSettings, RecorderError},
    tensor::{
        activation::softmax, backend::Backend, Device, ElementConversion, Int, Shape, Tensor,
        TensorData,
    },
};

#[cfg(feature = "import")]
use {
    crate::transformer::TransformerRecord,
    burn::record::Recorder,
    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
};

use crate::{
    sampling::Sampler,
    tokenizer::Tokenizer,
    transformer::{KeyValueCache, Transformer, TransformerConfig},
};

#[cfg(feature = "pretrained")]
use crate::pretrained::{self, ModelMeta};
#[cfg(feature = "tiny")]
use crate::tokenizer::SentiencePieceTokenizer;
#[cfg(feature = "llama3")]
use crate::tokenizer::Tiktoken;

#[derive(Config, Debug)]
pub struct LlamaConfig {
    /// The size of the model.
    #[config(default = "4096")]
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of transformer blocks.
    #[config(default = "32")]
    pub num_hidden_layers: usize,
    /// The number of attention heads.
    #[config(default = "32")]
    pub num_attention_heads: usize,
    /// The number of key-value heads.
    pub num_key_value_heads: Option<usize>,
    /// The vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon
    #[config(default = "1e-5")]
    pub norm_eps: f64,
    /// Rotary positional encoding (RoPE).
    #[config(default = "RopeConfig::new(10000.0)")]
    pub rope: RopeConfig,
    /// Maximum sequence length for input text.
    #[config(default = "128")]
    pub max_seq_len: usize,
    /// The tokenizer path.
    pub tokenizer: String,
}

/// Rotary positional encoding (RoPE)
#[derive(Config, Debug)]
pub struct RopeConfig {
    pub theta: f32,
    #[config(default = "None")]
    pub scaled: Option<RopeFrequencyScaling>,
}

/// RoPE frequency scaling.
#[derive(Config, Debug)]
pub struct RopeFrequencyScaling {
    #[config(default = "8.")]
    pub scale_factor: f32,
    #[config(default = "1.")]
    pub low_freq_factor: f32,
    #[config(default = "4.")]
    pub high_freq_factor: f32,
    #[config(default = "8192.")]
    pub old_context_len: f32,
}

impl LlamaConfig {
    /// Llama-3.2-3B configuration.
    pub fn llama3_2_3b(tokenizer_path: &str) -> Self {
        // hidden_size = 8192; vocab_size = 128256
        Self::new(8192, 128256, tokenizer_path.to_string())
            .with_d_model(3072)
            .with_num_hidden_layers(28)
            .with_num_attention_heads(24)
            .with_num_key_value_heads(Some(8))
            .with_rope(
                RopeConfig::new(500000.0)
                    .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.))),
            )
    }

    /// Llama-3.2-1B configuration.
    pub fn llama3_2_1b(tokenizer_path: &str) -> Self {
        // hidden_size = 8192; vocab_size = 128256
        Self::new(8192, 128256, tokenizer_path.to_string())
            .with_d_model(2048)
            .with_num_hidden_layers(16)
            .with_num_key_value_heads(Some(8))
            .with_rope(
                RopeConfig::new(500000.0)
                    .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.))),
            )
    }

    /// Llama-3.1-8B configuration.
    pub fn llama3_1_8b(tokenizer_path: &str) -> Self {
        // hidden_size = 14336; vocab_size = 128256
        Self::new(14336, 128256, tokenizer_path.to_string())
            .with_num_key_value_heads(Some(8))
            .with_rope(RopeConfig::new(500000.0).with_scaled(Some(RopeFrequencyScaling::new())))
    }

    /// Llama-3-8B configuration.
    pub fn llama3_8b(tokenizer_path: &str) -> Self {
        // hidden_size = 14336; vocab_size = 128256
        Self::new(14336, 128256, tokenizer_path.to_string())
            .with_num_key_value_heads(Some(8))
            .with_rope(RopeConfig::new(500000.0))
    }

    /// TinyLlama-1.1B Chat v1.0 configuration.
    pub fn tiny_llama(tokenizer_path: &str) -> Self {
        // hidden_size = 5632; vocab_size = 32000
        Self::new(5632, 32000, tokenizer_path.to_string())
            .with_d_model(2048)
            .with_num_hidden_layers(22)
            .with_num_key_value_heads(Some(4))
            .with_rope(RopeConfig::new(10000.0))
    }

    /// Load pre-trained Llama-3.2-3B model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    #[cfg(feature = "llama3")]
    pub fn load_llama3_2_3b<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::llama3_2_3b(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<B, Tiktoken>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained Llama-3.2-3B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    #[cfg(all(feature = "llama3", feature = "pretrained"))]
    pub fn llama3_2_3b_pretrained<B: Backend>(
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        // Llama-3.2 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        let model = pretrained::Llama::Llama323bInstruct.pretrained();
        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_llama3_2_3b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }

    /// Load pre-trained Llama-3.2-1B model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    #[cfg(feature = "llama3")]
    pub fn load_llama3_2_1b<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::llama3_2_1b(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<B, Tiktoken>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained Llama-3.2-3B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    #[cfg(all(feature = "llama3", feature = "pretrained"))]
    pub fn llama3_2_1b_pretrained<B: Backend>(
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        // Llama-3.2 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        let model = pretrained::Llama::Llama321bInstruct.pretrained();
        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_llama3_2_1b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }

    /// Load pre-trained Llama-3.1-8B model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    #[cfg(feature = "llama3")]
    pub fn load_llama3_1_8b<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::llama3_1_8b(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<B, Tiktoken>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained Llama-3.1-8B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    #[cfg(all(feature = "llama3", feature = "pretrained"))]
    pub fn llama3_1_8b_pretrained<B: Backend>(
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        // Llama-3.1 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        let model = pretrained::Llama::Llama31Instruct.pretrained();
        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_llama3_1_8b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }

    /// Load pre-trained Llama-3-8B model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    #[cfg(feature = "llama3")]
    pub fn load_llama3_8b<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::llama3_8b(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<B, Tiktoken>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained Llama-3-8B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    #[cfg(all(feature = "llama3", feature = "pretrained"))]
    pub fn llama3_8b_pretrained<B: Backend>(
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, Tiktoken>, String> {
        // Llama-3 models support context length up to 8K tokens.
        check_context_length(max_seq_len, 8 * 1024);

        // Download checkpoint and tokenizer
        let model = pretrained::Llama::Llama3Instruct.pretrained();
        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_llama3_8b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }

    /// Load pre-trained TinyLlama-1.1B Chat v1.0 model with [SentenciePiece](https://github.com/google/sentencepiece) tokenizer.
    #[cfg(feature = "tiny")]
    pub fn load_tiny_llama<B: Backend>(
        checkpoint: &str,
        tokenizer_path: &str,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, SentiencePieceTokenizer>, String> {
        use burn::record::NamedMpkFileRecorder;

        let llama = Self::tiny_llama(tokenizer_path)
            .with_max_seq_len(max_seq_len)
            .init::<B, SentiencePieceTokenizer>(device)?;

        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        let llama = llama
            .load(checkpoint, &recorder)
            .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

        Ok(llama)
    }

    /// Load pre-trained TinyLlama-1.1B Chat v1.0 model with [SentenciePiece](https://github.com/google/sentencepiece) tokenizer.
    #[cfg(all(feature = "tiny", feature = "pretrained"))]
    pub fn tiny_llama_pretrained<B: Backend>(
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Llama<B, SentiencePieceTokenizer>, String> {
        // TinyLlama models support context length up to 2K tokens.
        check_context_length(max_seq_len, 2 * 1024);

        // Download checkpoint and tokenizer
        let model = pretrained::Llama::TinyLlama.pretrained();
        let checkpoint = model
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        Self::load_tiny_llama(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )
    }

    /// Initialize a new [Llama](Llama) module.
    pub fn init<B: Backend, T: Tokenizer>(
        &self,
        device: &Device<B>,
    ) -> Result<Llama<B, T>, String> {
        let tokenizer = T::new(&self.tokenizer)?;
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let model = TransformerConfig::new(
            self.vocab_size,
            self.num_hidden_layers,
            self.d_model,
            self.hidden_size,
            self.num_attention_heads,
            num_key_value_heads,
        )
        .with_max_seq_len(self.max_seq_len)
        .with_norm_eps(self.norm_eps)
        .init(device);

        let cache = (0..self.num_hidden_layers)
            .map(|_| KeyValueCache::new(self.max_seq_len))
            .collect::<Vec<_>>();

        let rope = RotaryEncodingConfig::new(
            self.max_seq_len * 2,
            self.d_model / self.num_attention_heads,
        )
        .with_theta(self.rope.theta);

        let rope = if let Some(scaling) = &self.rope.scaled {
            let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);
            rope.init_with_frequency_scaling(freq_scaling_fn, device)
        } else {
            rope.init(device)
        };

        Ok(Llama {
            tokenizer,
            model,
            cache,
            rope,
            device: device.clone(),
        })
    }

    /// Load pre-trained Llama checkpoint.
    #[cfg(feature = "import")]
    pub fn load_pretrained<B: Backend, T: Tokenizer>(
        &self,
        checkpoint: &str,
        device: &Device<B>,
    ) -> Result<Llama<B, T>, String> {
        let mut llama = self.init(device)?;

        // Load weights from torch state_dict
        let mut load_args = LoadArgs::new(checkpoint.into());

        if !cfg!(feature = "tiny") {
            load_args = load_args
                // Map layers.[i].feed_forward.w1.* -> layers.[i].feed_forward.swiglu.linear_inner.*
                .with_key_remap(
                    "(layers\\.[0-9]+\\.feed_forward)\\.w1\\.(.+)",
                    "$1.swiglu.linear_inner.$2",
                )
                // Map layers.[i].feed_forward.w3.* -> layers.[i].feed_forward.swiglu.linear_outer.*
                .with_key_remap(
                    "(layers\\.[0-9]+\\.feed_forward)\\.w3\\.(.+)",
                    "$1.swiglu.linear_outer.$2",
                )
                // Map norm.weight -> norm.gamma for all layers
                .with_key_remap("(.*)norm\\.weight", "${1}norm.gamma");
        } else {
            load_args = load_args
                // Map lm_head.* -> output.*
                .with_key_remap("lm_head\\.(.+)", "output.$1")
                // Remove model. prefix
                .with_key_remap("model\\.(.+)", "$1")
                // Map embed_tokens.* -> tok_embeddings.*
                .with_key_remap("embed_tokens\\.(.+)", "tok_embeddings.$1")
                // Map layers.[i].input_layernorm.* -> layers.[i].attention_norm.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.input_layernorm\\.(.+)",
                    "$1.attention_norm.$2",
                )
                // Map layers.[i].post_attention_layernorm.* -> layers.[i].ffn_norm.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.post_attention_layernorm\\.(.+)",
                    "$1.ffn_norm.$2",
                )
                // Map layers.[i].mlp.down_proj.* -> layers.[i].feed_forward.w2.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.mlp\\.down_proj\\.(.+)",
                    "$1.feed_forward.w2.$2",
                )
                // Map layers.[i].mlp.gate_proj.* -> layers.[i].feed_forward.swiglu.linear_inner.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.mlp\\.gate_proj\\.(.+)",
                    "$1.feed_forward.swiglu.linear_inner.$2",
                )
                // Map layers.[i].mlp.up_proj.* -> layers.[i].feed_forward.swiglu.linear_outer.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.mlp\\.up_proj\\.(.+)",
                    "$1.feed_forward.swiglu.linear_outer.$2",
                )
                // Map layers.[i].self_attn.k_proj.* -> layers.[i].attention.wk.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.self_attn\\.k_proj\\.(.+)",
                    "$1.attention.wk.$2",
                )
                // Map layers.[i].self_attn.o_proj.* -> layers.[i].attention.wo.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.self_attn\\.o_proj\\.(.+)",
                    "$1.attention.wo.$2",
                )
                // Map layers.[i].self_attn.q_proj.* -> layers.[i].attention.wq.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.self_attn\\.q_proj\\.(.+)",
                    "$1.attention.wq.$2",
                )
                // Map layers.[i].self_attn.v_proj.* -> layers.[i].attention.wv.*
                .with_key_remap(
                    "(layers\\.[0-9]+)\\.self_attn\\.v_proj\\.(.+)",
                    "$1.attention.wv.$2",
                )
                // Map norm.weight -> norm.gamma for all layers
                .with_key_remap("(.*)norm\\.weight", "${1}norm.gamma");
        }
        println!("Loading record...");
        let now = Instant::now();
        let mut record: TransformerRecord<B> = PyTorchFileRecorder::<HalfPrecisionSettings>::new()
            .load(load_args, device)
            .map_err(|e| e.to_string())?;
        let elapsed = now.elapsed().as_secs();
        println!("Loaded in {}s", elapsed);

        if cfg!(feature = "tiny") {
            // TinyLlama weights from HuggingFace use a different rotary positional encoding
            // which requires weight permutation:
            // https://github.com/huggingface/transformers/issues/25199#issuecomment-1687720247
            // https://github.com/jzhang38/TinyLlama/issues/24
            let n_heads = self.num_attention_heads;
            let n_kv_heads = self.num_key_value_heads.unwrap_or(n_heads);
            let wk_dim = self.d_model * n_kv_heads / n_heads;
            let permute = |w: Tensor<B, 2>, n_heads: usize, dim1: usize, dim2: usize| {
                let w = w // [2048, 256]
                    .reshape([dim1, n_heads, 2, dim2 / n_heads / 2]) // [2048, 4, 2, 32]
                    .swap_dims(2, 3) // [2048, 4, 32, 2]
                    .reshape([dim1, dim2]);
                w
            };

            record.layers = record
                .layers
                .into_iter()
                .map(|mut layer| {
                    layer.attention.wq.weight = layer
                        .attention
                        .wq
                        .weight
                        .map(|w| permute(w, n_heads, self.d_model, self.d_model));
                    layer.attention.wk.weight = layer
                        .attention
                        .wk
                        .weight
                        .map(|w| permute(w, n_kv_heads, self.d_model, wk_dim));
                    layer
                })
                .collect::<Vec<_>>();
        }

        llama.model = llama.model.load_record(record);
        println!("Llama record loaded");

        Ok(llama)
    }
}

fn check_context_length(max_seq_len: usize, max_context_len: usize) {
    assert!(
        max_seq_len <= max_context_len,
        "Maximum sequence length must not exceed {max_context_len}"
    );
}

/// Generated text sample output.
pub struct GenerationOutput {
    /// The generated text.
    pub text: String,
    /// The number of generated tokens.
    pub tokens: usize,
    /// The time it took to produce the output tokens (generation + decoding).
    pub time: f64,
}

/// Meta Llama large language model and tokenizer.
pub struct Llama<B: Backend, T: Tokenizer> {
    /// The tokenizer.
    pub tokenizer: T,
    /// Llama decoder-only transformer.
    pub model: Transformer<B>,
    /// Key-value cache for each transformer block.
    pub cache: Vec<KeyValueCache<B>>,
    /// Rotary positional encoding (RoPE).
    pub rope: RotaryEncoding<B>,
    pub device: Device<B>,
}

impl<B: Backend, T: Tokenizer> Llama<B, T> {
    /// Generate text sample based on the provided prompt.
    ///
    /// # Arguments
    /// - `prompt`: The prompt string to use for generating the samples.
    /// - `sample_len`: The number of new tokens to generate (i.e., the number of generation steps to take).
    /// - `temperature`: Temperature value for controlling randomness in sampling (scales logits by `1 / temperature`).
    ///                  High values result in more random sampling.
    /// - `sampler`: The sampling strategy to use when selecting the next token based on the predicted probabilities.
    ///
    /// # Returns
    /// The generated text along with some other metadata (see [GenerationOutput]).
    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        temperature: f64,
        sampler: &mut Sampler,
    ) -> GenerationOutput {
        let mut tokens = self.tokenize(prompt);
        let prompt_len = tokens.dims()[0];
        let stop_tokens = Tensor::from_ints(self.tokenizer.stop_ids().as_slice(), &self.device);

        let mut num_tokens: usize = 0;
        let mut input_pos = Tensor::<B, 1, Int>::arange(0..tokens.dims()[0] as i64, &self.device);
        let now = Instant::now();
        for _ in 0..sample_len {
            let x = tokens.clone().select(0, input_pos.clone()).reshape([1, -1]);
            let logits = self.model.forward(x, &mut self.cache, &self.rope);

            let [batch_size, seq_len, _vocab_size] = logits.dims();
            let mut next_token_logits = logits
                .slice([0..batch_size, seq_len - 1..seq_len])
                .squeeze(1); // [batch_size=1, vocab_size]

            if temperature > 0.0 {
                next_token_logits = temperature_scaled_softmax(next_token_logits, temperature);
            };

            let next_token = sampler.sample(next_token_logits).squeeze(0);

            // Stop when any of the valid stop tokens is encountered
            if stop_tokens
                .clone()
                .equal(next_token.clone())
                .any()
                .into_scalar()
            {
                break;
            }

            // Concatenate the new generated token
            tokens = Tensor::cat(vec![tokens, next_token], 0);
            num_tokens += 1;

            // Advance
            let t = input_pos.dims()[0];
            input_pos = input_pos.slice([t - 1..t]) + 1;
        }

        let tokens = tokens.into_data().as_slice::<B::IntElem>().unwrap()[prompt_len..]
            .iter()
            .map(|t| t.elem::<u32>())
            .collect::<Vec<_>>();

        let generated = self.tokenizer.decode(tokens);
        let elapsed = now.elapsed().as_secs_f64();

        GenerationOutput {
            text: generated,
            tokens: num_tokens,
            time: elapsed,
        }
    }

    /// Encode a string into a tensor of tokens.
    fn tokenize(&self, text: &str) -> Tensor<B, 1, Int> {
        let bos = !cfg!(feature = "tiny"); // TinyLlama Chat doesn't prepend BOS token with the chat format
        let tokens = self.tokenizer.encode(text, bos, false);

        let shape = Shape::new([tokens.len()]);
        Tensor::<B, 1, Int>::from_data(TensorData::new(tokens, shape), &self.device)
    }

    /// Save Llama model to file using the specified recorder.
    pub fn save<R: FileRecorder<B>>(
        self,
        file_path: &str,
        recorder: &R,
    ) -> Result<(), RecorderError> {
        println!("Saving record...");
        let now = Instant::now();
        self.model.save_file(file_path, recorder)?;
        let elapsed = now.elapsed().as_secs();
        println!("Saved in {}s", elapsed);

        Ok(())
    }

    /// Load Llama model from file using the specified recorder.
    pub fn load<R: FileRecorder<B>>(
        mut self,
        file_path: &str,
        recorder: &R,
    ) -> Result<Self, RecorderError> {
        println!("Loading record...");
        let now = Instant::now();
        self.model = self.model.load_file(file_path, recorder, &self.device)?;
        let elapsed = now.elapsed().as_secs();
        println!("Loaded in {}s", elapsed);

        Ok(self)
    }
}

impl RopeFrequencyScaling {
    /// Applies frequency scaling by parts following Llama 3.1's scheme.
    ///
    /// Adapted from: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L45
    pub fn freq_scaling_by_parts<B: Backend>(&self, freqs: Tensor<B, 1>) -> Tensor<B, 1> {
        let low_freq_wavelen = self.old_context_len / self.low_freq_factor;
        let high_freq_wavelen = self.old_context_len / self.high_freq_factor;

        let wavelen = freqs.clone().recip().mul_scalar(2. * core::f32::consts::PI);

        // if wavelen >= high_freq_wavelen
        let cond = wavelen.clone().greater_equal_elem(high_freq_wavelen);
        let smooth = wavelen
            .clone()
            .recip()
            .mul_scalar(self.old_context_len)
            .sub_scalar(self.low_freq_factor)
            .div_scalar(self.high_freq_factor - self.low_freq_factor);
        // (1 - smooth) * freq / scale_factor + smooth * freq
        let new_freqs = smooth
            .clone()
            .neg()
            .add_scalar(1.)
            .mul(freqs.clone().div_scalar(self.scale_factor))
            .add(smooth.clone().mul(freqs.clone()));
        let new_freqs = freqs.clone().mask_where(cond, new_freqs);

        // if wavelen > low_freq_wavelen
        let cond = wavelen.clone().greater_elem(low_freq_wavelen);
        let new_freqs = new_freqs.mask_where(cond, freqs.clone().div_scalar(self.scale_factor));

        // if wavelen < high_freq_wavelen
        let cond = wavelen.lower_elem(high_freq_wavelen);
        let new_freqs = new_freqs.mask_where(cond, freqs);

        new_freqs
    }
}

pub(crate) fn temperature_scaled_softmax<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
) -> Tensor<B, 2> {
    softmax(logits / temperature, 1)
}

#[cfg(test)]
#[cfg(any(feature = "cuda", feature = "tch-gpu"))]
mod tests {
    use super::*;
    use crate::tests::*;

    use burn::tensor::TensorData;

    #[test]
    fn test_temperature_softmax() {
        let tensor = TestTensor::<2>::from([[21.3125, 19.859375, 19.0625, 18.75, 18.171875]]);

        let output = crate::llama::temperature_scaled_softmax(tensor, 0.6);
        let expected = TensorData::from([[
            0.8691406,
            0.07836914,
            0.020767212,
            0.0124053955,
            0.0047035217,
        ]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn test_transformer_block() {
        let device = Default::default();

        let max_seq_len = 16;
        let block = crate::transformer::TransformerBlockConfig::new(
            /*n_layers=*/ 1, /*d_model=*/ 4, /*hidden_size=*/ 16,
            /*n_heads=*/ 2, /*n_kv_heads=*/ 1, /*norm_eps=*/ 0.00001,
        )
        .init::<TestBackend>(&device);
        let mut cache = crate::transformer::KeyValueCache::new(max_seq_len);

        let rope = RopeConfig::new(500000.0)
            .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.)));
        let scaling = rope.scaled.unwrap();
        let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);

        let rope = RotaryEncodingConfig::new(max_seq_len * 2, 4 / 2)
            .with_theta(rope.theta)
            .init_with_frequency_scaling(freq_scaling_fn, &device);

        // input: [batch_size, seq_len, d_model]
        let input = TestTensor::<3>::from([[
            [0.0026, 0.003, -0.006, 0.006],
            [0.001, 0.0008, 0.0015, -0.016],
        ]]);
        let output = block.forward(input, &mut cache, &rope);
        let expected = TensorData::from([[
            [-0.04269409, 0.020523071, -0.0791626, 0.12731934],
            [-0.091674805, -0.013809204, 0.03152466, -0.058776855],
        ]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn test_rope() {
        let device = Default::default();

        let max_seq_len = 16;
        let rope = RopeConfig::new(500000.0)
            .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.)));
        let scaling = rope.scaled.unwrap();
        let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);

        let rope = RotaryEncodingConfig::new(max_seq_len * 2, 4 / 2)
            .with_theta(rope.theta)
            .init_with_frequency_scaling(freq_scaling_fn, &device);

        let input = TestTensor::<4>::from([[
            [[-0.60253906, -0.035308838], [0.41357422, 0.15100098]],
            [[-0.044677734, -0.094177246], [0.60546875, 0.2442627]],
        ]]);

        let output = rope.apply(input, 0);
        let expected = TensorData::from([[
            [[-0.60253906, -0.035308838], [0.09643555, 0.42944336]],
            [[-0.044677734, -0.094177246], [0.12194824, 0.64160156]],
        ]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}
