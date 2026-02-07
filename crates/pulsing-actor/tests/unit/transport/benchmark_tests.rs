//! HTTP/2 performance and benchmark tests

use crate::common::fixtures::TestHandler;
use pulsing_actor::transport::{Http2Client, Http2Config, Http2Server};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn test_http2_throughput_benchmark() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::high_throughput(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();
    let client = Arc::new(Http2Client::new(Http2Config::high_throughput()));

    let request_count = 1000;
    let start = std::time::Instant::now();

    for i in 0..request_count {
        let _ = client
            .ask(
                addr,
                "/actors/bench",
                "Bench",
                format!("req-{}", i).into_bytes(),
            )
            .await
            .unwrap();
    }

    let elapsed = start.elapsed();
    let rps = request_count as f64 / elapsed.as_secs_f64();

    println!(
        "HTTP/2 Sequential Throughput: {} requests in {:?} ({:.0} req/s)",
        request_count, elapsed, rps
    );

    assert!(rps > 100.0, "Throughput too low: {} req/s", rps);

    cancel.cancel();
}

#[tokio::test]
async fn test_http2_concurrent_throughput_benchmark() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::high_throughput(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();
    let client = Arc::new(Http2Client::new(Http2Config::high_throughput()));

    let request_count = 1000;
    let concurrency = 50;
    let start = std::time::Instant::now();

    let mut handles = Vec::new();
    for i in 0..request_count {
        let client = client.clone();
        let handle = tokio::spawn(async move {
            client
                .ask(
                    addr,
                    "/actors/bench",
                    "Bench",
                    format!("req-{}", i).into_bytes(),
                )
                .await
        });
        handles.push(handle);

        if handles.len() >= concurrency {
            let results: Vec<_> = futures::future::join_all(handles.drain(..)).await;
            for r in results {
                r.unwrap().unwrap();
            }
        }
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    for r in results {
        r.unwrap().unwrap();
    }

    let elapsed = start.elapsed();
    let rps = request_count as f64 / elapsed.as_secs_f64();

    println!(
        "HTTP/2 Concurrent Throughput ({} concurrency): {} requests in {:?} ({:.0} req/s)",
        concurrency, request_count, elapsed, rps
    );

    assert!(rps > 100.0, "Concurrent throughput too low: {} req/s", rps);

    cancel.cancel();
}

#[tokio::test]
async fn test_http2_latency_benchmark() {
    let handler = Arc::new(TestHandler::new());
    let cancel = CancellationToken::new();

    let server = Http2Server::new(
        "127.0.0.1:0".parse().unwrap(),
        handler,
        Http2Config::low_latency(),
        cancel.clone(),
    )
    .await
    .unwrap();

    let addr = server.local_addr();
    let client = Http2Client::new(Http2Config::low_latency());

    let request_count = 100;
    let mut latencies = Vec::with_capacity(request_count);

    for i in 0..request_count {
        let start = std::time::Instant::now();
        let _ = client
            .ask(
                addr,
                "/actors/latency",
                "Ping",
                format!("req-{}", i).into_bytes(),
            )
            .await
            .unwrap();
        latencies.push(start.elapsed());
    }

    latencies.sort();
    let min = latencies.first().unwrap();
    let max = latencies.last().unwrap();
    let median = latencies[request_count / 2];
    let p99 = latencies[(request_count * 99) / 100];
    let avg: std::time::Duration =
        latencies.iter().sum::<std::time::Duration>() / request_count as u32;

    println!(
        "HTTP/2 Latency: min={:?}, avg={:?}, median={:?}, p99={:?}, max={:?}",
        min, avg, median, p99, max
    );

    assert!(
        p99 < std::time::Duration::from_millis(500),
        "P99 latency too high: {:?}",
        p99
    );

    cancel.cancel();
}
