//! HTTP/2 Connection Pool 2x 自动扩容示例
//!
//! 演示当并发请求数超过 `max_connections_per_host` 时,连接池会自动 2x 扩容,
//! 而不是抛出 "Connection pool exhausted" 错误。
//!
//! Run: cargo run --example http2_pool_expand -p pulsing-actor

use bytes::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::{TokioExecutor, TokioIo};
use pulsing_actor::transport::http2::{ConnectionPool, Http2Config, PoolConfig};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;

const INITIAL_CAP: usize = 2;
const BURST_SIZE: usize = 8;
const HOLD_DURATION: Duration = Duration::from_millis(300);

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,pulsing_actor::transport::http2=warn".into()),
        )
        .init();

    let addr = spawn_h2c_server().await?;
    println!("h2c server listening on {addr}");

    let pool_config = PoolConfig {
        max_connections_per_host: INITIAL_CAP,
        // 关闭 cleanup 触发条件,让所有连接在 burst 期间稳稳存活
        max_connection_age: None,
        max_idle_time: None,
        max_requests_per_connection: None,
        ..Default::default()
    };
    let pool = Arc::new(ConnectionPool::with_config(
        Http2Config::default(),
        pool_config,
    ));

    println!(
        "\n=== burst test ===\n  initial cap : {INITIAL_CAP}\n  burst size  : {BURST_SIZE}\n  hold time   : {HOLD_DURATION:?}\n  expected    : pool grows 2 -> 4 -> 8, all {BURST_SIZE} requests succeed\n"
    );

    let mut handles = Vec::with_capacity(BURST_SIZE);
    for i in 0..BURST_SIZE {
        let pool = pool.clone();
        handles.push(tokio::spawn(async move {
            let started = Instant::now();
            let outcome = pool.get_connection(addr).await;
            let elapsed = started.elapsed();
            match outcome {
                Ok(guard) => {
                    println!("  [task {i:>2}] acquired in {elapsed:>8.2?}");
                    tokio::time::sleep(HOLD_DURATION).await;
                    drop(guard);
                    Ok(())
                }
                Err(e) => {
                    eprintln!("  [task {i:>2}] FAILED after {elapsed:>8.2?}: {e}");
                    Err(e)
                }
            }
        }));
    }

    let mut ok = 0usize;
    let mut err = 0usize;
    for h in handles {
        match h.await? {
            Ok(()) => ok += 1,
            Err(_) => err += 1,
        }
    }

    println!("\nresults: {ok} succeeded, {err} failed");

    let info = pool.pool_info().await;
    println!("\npool_info:\n{}", serde_json::to_string_pretty(&info)?);

    let expansions = info["stats"]["pool_expansions"].as_u64().unwrap_or(0);
    let final_cap = info["hosts"]
        .as_array()
        .and_then(|a| a.first())
        .and_then(|h| h["total_connections"].as_u64())
        .unwrap_or(0);

    println!("\n--- verdict ---");
    if err == 0 && expansions >= 1 && final_cap as usize >= BURST_SIZE {
        println!(
            "PASS: pool expanded {expansions} time(s); all {BURST_SIZE} requests succeeded; \
             {final_cap} connections live (>= burst size)"
        );
        Ok(())
    } else {
        Err(format!(
            "FAIL: succeeded={ok}, failed={err}, expansions={expansions}, live_conns={final_cap}"
        )
        .into())
    }
}

async fn spawn_h2c_server() -> std::io::Result<SocketAddr> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let addr = listener.local_addr()?;

    tokio::spawn(async move {
        loop {
            let (stream, _peer) = match listener.accept().await {
                Ok(a) => a,
                Err(_) => break,
            };
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let service = service_fn(|_req: Request<hyper::body::Incoming>| async {
                    Ok::<_, Infallible>(Response::new(Full::new(Bytes::from_static(b"ok"))))
                });
                let _ = hyper::server::conn::http2::Builder::new(TokioExecutor::new())
                    .serve_connection(io, service)
                    .await;
            });
        }
    });

    Ok(addr)
}
