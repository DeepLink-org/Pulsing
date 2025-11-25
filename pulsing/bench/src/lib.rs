use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use chrono::Local;
use clap::error::ErrorKind::InvalidValue;
use clap::{ArgGroup, Error, Parser};
use log::{debug, error, info, warn, Level, LevelFilter};
use reqwest::Url;
use tokenizers::{FromPretrainedParameters, Tokenizer};
use tokio::sync::broadcast::{self, Sender};
use tokio::sync::Mutex;
use writers::BenchmarkReportWriter;

pub use crate::benchmark::{BenchmarkConfig, BenchmarkKind};
use crate::benchmark::{Event, MessageEvent};
pub use crate::console::run_console;
pub use crate::profiles::apply_profile;
use crate::requests::OpenAITextGenerationBackend;
pub use crate::requests::TokenizeOptions;

mod benchmark;
mod console;
mod executors;
mod profiles;
mod requests;
mod results;
mod scheduler;
mod table;
mod writers;

pub struct RunConfiguration {
    pub url: Url,
    pub api_key: String,
    pub tokenizer_name: String,
    pub profile: Option<String>,
    pub max_vus: u64,
    pub duration: std::time::Duration,
    pub rates: Option<Vec<f64>>,
    pub num_rates: u64,
    pub benchmark_kind: String,
    pub warmup_duration: std::time::Duration,
    pub prompt_options: Option<TokenizeOptions>,
    pub decode_options: Option<TokenizeOptions>,
    pub dataset: String,
    pub dataset_file: String,
    pub hf_token: Option<String>,
    pub extra_metadata: Option<HashMap<String, String>>,
    pub model_name: String,
    pub run_id: String,
}

pub async fn run(mut run_config: RunConfiguration, stop_sender: Sender<()>) -> anyhow::Result<()> {
    info!("Starting benchmark");
    // set process system limits
    sysinfo::set_open_files_limit(0);
    // apply profile if needed
    run_config = match run_config.profile.clone() {
        None => run_config,
        Some(profile) => match apply_profile(profile.as_str(), run_config) {
            Ok(config) => {
                info!("Profile applied: {}", profile);
                config
            }
            Err(e) => {
                error!("Failed to apply profile: {:?}", e);
                return Err(e);
            }
        },
    };
    // initialize tokenizer
    let params = FromPretrainedParameters {
        token: run_config.hf_token.clone(),
        ..Default::default()
    };
    let tokenizer =
        match Tokenizer::from_pretrained(run_config.tokenizer_name.clone(), Some(params)) {
            Ok(tokenizer) => tokenizer,
            Err(e) => {
                return Err(anyhow::anyhow!("Error loading tokenizer: {e}"));
            }
        };
    let tokenizer = Arc::new(tokenizer);
    let backend = OpenAITextGenerationBackend::try_new(
        run_config.api_key,
        run_config.url,
        run_config.model_name.clone(),
        tokenizer,
        run_config.duration,
    )?;

    let config = BenchmarkConfig {
        max_vus: run_config.max_vus,
        duration: run_config.duration,
        benchmark_kind: match run_config.benchmark_kind.to_lowercase().as_str() {
            "throughput" => BenchmarkKind::Throughput,
            "sweep" => BenchmarkKind::Sweep,
            "csweep" => BenchmarkKind::ConcurrencySweep,
            "rate" => BenchmarkKind::Rate,
            _ => BenchmarkKind::Sweep,
        },
        warmup_duration: run_config.warmup_duration,
        rates: run_config.rates,
        num_rates: run_config.num_rates,
        prompt_options: run_config.prompt_options.clone(),
        decode_options: run_config.decode_options.clone(),
        tokenizer: run_config.tokenizer_name.clone(),
        model_name: run_config.model_name.clone(),
        profile: run_config.profile.clone(),
        extra_metadata: run_config.extra_metadata.clone(),
        run_id: run_config.run_id.clone(),
    };
    config.validate()?;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    // Always use console interface, send logs to file for debugging
    let target = Box::new(File::create("log.txt").expect("Can't create file"));
    env_logger::Builder::new()
        .target(env_logger::Target::Pipe(target))
        .filter(Some("inference_benchmarker"), LevelFilter::Debug)
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {} {}:{}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                record.level(),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        })
        .init();
    let config_clone = config.clone();
    let mut stop_receiver = stop_sender.subscribe();
    let stop_sender_clone = stop_sender.clone();
    let ui_thread = tokio::spawn(async move {
        tokio::select! {
            _ = stop_receiver.recv() => {
                debug!("Received stop signal, stopping benchmark");
            }
            _ = async{
                run_console(config_clone, rx, stop_sender_clone).await;
            } => {}
        }
    });

    // download prompts dataset
    info!("Downloading dataset");
    let _ = tx.send(Event::Message(MessageEvent {
        message: "Downloading dataset".to_string(),
        timestamp: chrono::Utc::now(),
        level: Level::Info,
    }));
    let filepath = requests::ConversationTextRequestGenerator::download_dataset(
        run_config.dataset,
        run_config.dataset_file,
        run_config.hf_token.clone(),
    )
    .expect("Can't download dataset");
    let requests = requests::ConversationTextRequestGenerator::load(
        filepath,
        run_config.tokenizer_name.clone(),
        run_config.prompt_options,
        run_config.decode_options,
        run_config.hf_token,
    )?;

    let mut benchmark = benchmark::Benchmark::new(
        config.clone(),
        Box::new(backend),
        Arc::from(Mutex::from(requests)),
        tx.clone(),
        stop_sender.clone(),
    );
    let mut stop_receiver = stop_sender.subscribe();
    tokio::select! {
        report = benchmark.run() => {
            match report {
                Ok(_) => {
                    let report = benchmark.get_report();
                    let path = format!("results/{}_{}.json",run_config.tokenizer_name.replace("/","_").replace(".","_"), chrono::Utc::now().format("%Y-%m-%d-%H-%M-%S"));
                    let path=Path::new(&path);
                    let writer=BenchmarkReportWriter::try_new(config.clone(), report)?;
                    writer.json(path).await?;
                    info!("Report saved to {:?}",path);
                    let _ = tx.send(Event::BenchmarkReportEnd(format!("{:?}", path)));
                },
                Err(e) => {
                    error!("Error running benchmark: {:?}", e.to_string());
                    let _ = tx.send(Event::BenchmarkError(e.to_string()));
                }
            };
        }
        _ = stop_receiver.recv() => {
            debug!("Received stop signal, stopping benchmark");
        }
    }
    info!("Benchmark finished");
    // Always keep the console interface running
    ui_thread.await?;

    // No need to revert terminal since we're not using TUI anymore

    let report = benchmark.get_report();
    match BenchmarkReportWriter::try_new(config.clone(), report) {
        Ok(writer) => {
            writer.stdout().await?;
        }
        Err(_) => {
            warn!("No results to report.");
        }
    };

    Ok(())
}

#[derive(Parser, Debug)]
#[clap(
    author,
    version,
    about,
    long_about = None,
    group(ArgGroup::new("group_profile").multiple(true)),
    group(ArgGroup::new("group_manual").multiple(true).conflicts_with("group_profile"))
)]
pub struct Args {
    /// The name of the tokenizer to use
    #[clap(short, long, env)]
    pub tokenizer_name: String,

    /// The name of the model to use. If not provided, the same name as the tokenizer will be used.
    #[clap(long, env)]
    pub model_name: Option<String>,

    /// The maximum number of virtual users to use
    #[clap(default_value = "128", short, long, env, group = "group_manual")]
    pub max_vus: u64,
    /// The duration of each benchmark step
    #[clap(default_value = "120s", short, long, env, group = "group_manual")]
    #[arg(value_parser = parse_duration)]
    pub duration: std::time::Duration,
    /// A list of rates of requests to send per second (only valid for the ConstantArrivalRate benchmark).
    #[clap(short, long, env)]
    pub rates: Option<Vec<f64>>,
    /// The number of rates to sweep through (only valid for the "sweep" benchmark)
    /// The rates will be linearly spaced up to the detected maximum rate
    #[clap(default_value = "10", long, env)]
    pub num_rates: u64,
    /// A benchmark profile to use
    #[clap(long, env, group = "group_profile")]
    pub profile: Option<String>,
    /// The kind of benchmark to run (throughput, sweep, csweep, rate)
    #[clap(default_value = "sweep", short, long, env, group = "group_manual")]
    pub benchmark_kind: String,
    /// The duration of the prewarm step ran before the benchmark to warm up the backend (JIT, caches, etc.)
    #[clap(default_value = "30s", short, long, env, group = "group_manual")]
    #[arg(value_parser = parse_duration)]
    pub warmup: std::time::Duration,
    /// The URL of the backend to benchmark. Must be compatible with OpenAI Message API
    #[clap(default_value = "http://localhost:8000", short, long, env)]
    pub url: Url,

    /// The api key send to the [`url`] as Header "Authorization: Bearer {API_KEY}".
    #[clap(default_value = "", short, long, env)]
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
    #[clap(
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
    #[clap(
        long,
        env,
        value_parser(parse_tokenizer_options),
        group = "group_manual"
    )]
    pub decode_options: Option<TokenizeOptions>,
    /// Hugging Face dataset to use for prompt generation
    #[clap(
        default_value = "hlarcher/inference-benchmarker",
        long,
        env,
        group = "group_manual"
    )]
    pub dataset: String,
    /// File to use in the Dataset
    #[clap(
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
    #[clap(long, env, value_parser(parse_key_val))]
    pub extra_meta: Option<HashMap<String, String>>,
    // A run identifier to use for the benchmark. This is used to identify the benchmark in the
    // results file.
    #[clap(long, env)]
    pub run_id: Option<String>,
}

fn parse_duration(s: &str) -> Result<std::time::Duration, Error> {
    humantime::parse_duration(s).map_err(|_| Error::new(InvalidValue))
}

fn parse_key_val(s: &str) -> Result<HashMap<String, String>, Error> {
    let mut key_val_map = HashMap::new();
    let items = s.split(',').collect::<Vec<&str>>();
    for item in items.iter() {
        let key_value = item.split('=').collect::<Vec<&str>>();
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

#[tokio::main]
pub async fn benchmark_main(args: Vec<String>) {
    let args = Args::parse_from(args);
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
        debug!("Received stop signal, stopping benchmark");
        stop_sender_clone
            .send(())
            .expect("Failed to send stop signal");
    });

    let stop_sender_clone = stop_sender.clone();
    // get HF token
    let token_env_key = "HF_TOKEN".to_string();
    let cache = hf_hub::Cache::from_env();
    let hf_token = match std::env::var(token_env_key).ok() {
        Some(token) => Some(token),
        None => cache.token(),
    };
    let model_name = args
        .model_name
        .clone()
        .unwrap_or(args.tokenizer_name.clone());
    let run_id = args
        .run_id
        .unwrap_or(uuid::Uuid::new_v4().to_string()[..7].to_string());
    let run_config = RunConfiguration {
        url: args.url,
        api_key: args.api_key,
        profile: args.profile.clone(),
        tokenizer_name: args.tokenizer_name.clone(),
        max_vus: args.max_vus,
        duration: args.duration,
        rates: args.rates,
        num_rates: args.num_rates,
        benchmark_kind: args.benchmark_kind.clone(),
        warmup_duration: args.warmup,
        prompt_options: args.prompt_options.clone(),
        decode_options: args.decode_options.clone(),
        dataset: args.dataset.clone(),
        dataset_file: args.dataset_file.clone(),
        extra_metadata: args.extra_meta.clone(),
        hf_token,
        model_name,
        run_id,
    };
    let main_thread = tokio::spawn(async move {
        match run(run_config, stop_sender_clone).await {
            Ok(_) => {}
            Err(e) => {
                error!("Fatal: {:?}", e);
                println!("Fatal: {:?}", e)
            }
        };
    });
    let _ = main_thread.await;
}
