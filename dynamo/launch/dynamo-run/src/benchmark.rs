// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Duration;

use clap::error::ErrorKind::InvalidValue;
use clap::{ArgGroup, Error, Parser};
use inference_benchmarker::{run as benchmark_run, RunConfiguration, TokenizeOptions};
use reqwest::Url;

/// Benchmark subcommand for running inference benchmarks
#[derive(Parser, Debug, Clone)]
#[command(group(ArgGroup::new("group_profile").multiple(true)))]
#[command(group(ArgGroup::new("group_manual").multiple(true).conflicts_with("group_profile")))]
pub struct Benchmark {
    /// The name of the tokenizer to use
    #[arg(short, long, env)]
    pub tokenizer_name: String,

    /// The name of the model to use. If not provided, the same name as the tokenizer will be used.
    #[arg(long, env)]
    pub model_name: Option<String>,

    /// The maximum number of virtual users to use
    #[arg(default_value = "128", short, long, env, group = "group_manual")]
    pub max_vus: u64,

    /// The duration of each benchmark step
    #[arg(default_value = "120s", short, long, env, group = "group_manual")]
    #[arg(value_parser = parse_duration)]
    pub duration: Duration,

    /// A list of rates of requests to send per second (only valid for the ConstantArrivalRate benchmark).
    #[arg(short, long, env)]
    pub rates: Option<Vec<f64>>,

    /// The number of rates to sweep through (only valid for the "sweep" benchmark)
    /// The rates will be linearly spaced up to the detected maximum rate
    #[arg(default_value = "10", long, env)]
    pub num_rates: u64,

    /// A benchmark profile to use
    #[arg(long, env, group = "group_profile")]
    pub profile: Option<String>,

    /// The kind of benchmark to run (throughput, sweep, csweep, rate)
    #[arg(default_value = "sweep", short, long, env, group = "group_manual")]
    pub benchmark_kind: String,

    /// The duration of the prewarm step ran before the benchmark to warm up the backend (JIT, caches, etc.)
    #[arg(default_value = "30s", short, long, env, group = "group_manual")]
    #[arg(value_parser = parse_duration)]
    pub warmup: Duration,

    /// The URL of the backend to benchmark. Must be compatible with OpenAI Message API
    #[arg(default_value = "http://localhost:8000", short, long, env)]
    pub url: Url,

    /// The api key send to the [`url`] as Header "Authorization: Bearer {API_KEY}".
    #[arg(default_value = "", short, long, env)]
    pub api_key: String,

    /// Constraints for prompt length.
    /// No value means use the input prompt as defined in input dataset.
    /// We sample the number of tokens to generate from a normal distribution.
    /// Specified as a comma-separated list of key=value pairs.
    /// * num_tokens: target number of prompt tokens
    /// * min_tokens: minimum number of prompt tokens
    /// * max_tokens: maximum number of prompt tokens
    /// * variance: variance in the number of prompt tokens
    ///
    /// Example: num_tokens=200,max_tokens=210,min_tokens=190,variance=10
    #[arg(
        long,
        env,
        value_parser(parse_tokenizer_options),
        group = "group_manual"
    )]
    pub prompt_options: Option<TokenizeOptions>,

    /// Constraints for the generated text.
    /// We sample the number of tokens to generate from a normal distribution.
    /// Specified as a comma-separated list of key=value pairs.
    /// * num_tokens: target number of generated tokens
    /// * min_tokens: minimum number of generated tokens
    /// * max_tokens: maximum number of generated tokens
    /// * variance: variance in the number of generated tokens
    ///
    /// Example: num_tokens=200,max_tokens=210,min_tokens=190,variance=10
    #[arg(
        long,
        env,
        value_parser(parse_tokenizer_options),
        group = "group_manual"
    )]
    pub decode_options: Option<TokenizeOptions>,

    /// Hugging Face dataset to use for prompt generation
    #[arg(
        default_value = "hlarcher/inference-benchmarker",
        long,
        env,
        group = "group_manual"
    )]
    pub dataset: String,

    /// File to use in the Dataset
    #[arg(
        default_value = "share_gpt_filtered_small.json",
        long,
        env,
        group = "group_manual"
    )]
    pub dataset_file: String,

    /// Extra metadata to include in the benchmark results file, comma-separated key-value pairs.
    /// It can be, for example, used to include information about the configuration of the
    /// benched server.
    /// Example: --extra-meta "key1=value1,key2=value2"
    #[arg(long, env, value_parser(parse_key_val))]
    pub extra_meta: Option<HashMap<String, String>>,

    /// A run identifier to use for the benchmark. This is used to identify the benchmark in the
    /// results file.
    #[arg(long, env)]
    pub run_id: Option<String>,
}

fn parse_duration(s: &str) -> Result<Duration, Error> {
    humantime::parse_duration(s).map_err(|_| Error::new(InvalidValue))
}

fn parse_key_val(s: &str) -> Result<HashMap<String, String>, Error> {
    let mut key_val_map = HashMap::new();
    let items = s.split(",").collect::<Vec<&str>>();
    for item in items.iter() {
        let key_value = item.split("=").collect::<Vec<&str>>();
        if key_value.len() % 2 != 0 {
            return Err(Error::new(InvalidValue));
        }
        for i in 0..key_value.len() / 2 {
            key_val_map.insert(
                key_value[i * 2].to_string(),
                key_value[i * 2 + 1].to_string(),
            );
        }
    }
    Ok(key_val_map)
}

fn parse_tokenizer_options(s: &str) -> Result<TokenizeOptions, Error> {
    let mut tokenizer_options = TokenizeOptions::new();
    let items = s.split(",").collect::<Vec<&str>>();
    for item in items.iter() {
        let key_value = item.split("=").collect::<Vec<&str>>();
        if key_value.len() != 2 {
            return Err(Error::new(InvalidValue));
        }
        match key_value[0] {
            "num_tokens" => {
                tokenizer_options.num_tokens = Some(key_value[1].parse::<u64>().unwrap())
            }
            "min_tokens" => tokenizer_options.min_tokens = key_value[1].parse::<u64>().unwrap(),
            "max_tokens" => tokenizer_options.max_tokens = key_value[1].parse::<u64>().unwrap(),
            "variance" => tokenizer_options.variance = key_value[1].parse::<u64>().unwrap(),
            _ => return Err(Error::new(InvalidValue)),
        }
    }
    if tokenizer_options.num_tokens.is_some()
        && (tokenizer_options.num_tokens.unwrap() == 0
            || tokenizer_options.min_tokens == 0
            || tokenizer_options.max_tokens == 0)
    {
        return Err(Error::new(InvalidValue));
    }
    if tokenizer_options.min_tokens > tokenizer_options.max_tokens {
        return Err(Error::new(InvalidValue));
    }
    Ok(tokenizer_options)
}

impl Benchmark {
    pub async fn run(self) -> anyhow::Result<()> {
        use tokio::sync::broadcast;

        let git_sha = option_env!("VERGEN_GIT_SHA").unwrap_or("unknown");
        println!(
            "Text Generation Inference Benchmark {} ({})",
            env!("CARGO_PKG_VERSION"),
            git_sha
        );

        let (stop_sender, _) = broadcast::channel(1);
        // handle ctrl-c
        let stop_sender_clone = stop_sender.clone();
        tokio::spawn(async move {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to listen for ctrl-c");
            tracing::debug!("Received stop signal, stopping benchmark");
            stop_sender_clone
                .send(())
                .expect("Failed to send stop signal");
        });

        // get HF token
        let token_env_key = "HF_TOKEN".to_string();
        let cache = hf_hub::Cache::from_env();
        let hf_token = match std::env::var(token_env_key).ok() {
            Some(token) => Some(token),
            None => cache.token(),
        };
        let model_name = self
            .model_name
            .clone()
            .unwrap_or(self.tokenizer_name.clone());
        let run_id = self
            .run_id
            .unwrap_or(uuid::Uuid::new_v4().to_string()[..7].to_string());
        let run_config = RunConfiguration {
            url: self.url,
            api_key: self.api_key,
            profile: self.profile.clone(),
            tokenizer_name: self.tokenizer_name.clone(),
            max_vus: self.max_vus,
            duration: self.duration,
            rates: self.rates,
            num_rates: self.num_rates,
            benchmark_kind: self.benchmark_kind.clone(),
            warmup_duration: self.warmup,
            prompt_options: self.prompt_options.clone(),
            decode_options: self.decode_options.clone(),
            dataset: self.dataset.clone(),
            dataset_file: self.dataset_file.clone(),
            extra_metadata: self.extra_meta.clone(),
            hf_token,
            model_name,
            run_id,
        };
        benchmark_run(run_config, stop_sender).await
    }
}

