//! HTTP Server 模式完整示例：在内存中启动一个 `HttpServer`，并用原始 TCP 连接调用 API。
//!
//! 运行方式：
//!   cargo run --release --example server_http
//!
//! 演示 API：
//! - GET  /stats
//! - POST /insert
//! - POST /search
//! - POST /batch_insert
//!
//! 为什么用原始 TCP 而非外部 HTTP 客户端库：
//! 示例应保持最小依赖，且能清晰展示 HTTP/1.0 请求-响应格式。

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::thread;
use std::time::Duration;
use vdb_rs::http_server::HttpServer;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;

fn http_request(addr: SocketAddr, request: &str) -> String {
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
        .expect("failed to connect to HTTP server");
    stream
        .write_all(request.as_bytes())
        .expect("failed to send request");

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("failed to read response");
    response
}

fn json_body(response: &str) -> &str {
    // HTTP/1.0 响应格式：header\r\n\r\nbody
    response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body.trim())
        .unwrap_or(response.trim())
}

fn main() {
    let dim = 128;
    let index = IvfRabitqIndex::new(dim);
    let mut server = HttpServer::new(index);
    let addr = server
        .bind("127.0.0.1:0")
        .expect("failed to bind HTTP server");

    // 在独立线程中运行事件循环；主线程发送请求。
    thread::spawn(move || {
        server.run().expect("HTTP server failed");
    });

    // 等待 libevent 完成 bind 并开始监听。
    thread::sleep(Duration::from_millis(100));

    // 1. 健康检查 / stats。
    let resp = http_request(addr, "GET /stats HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n");
    println!("[http] /stats response:\n{}\n", json_body(&resp));

    // 2. 单条插入。
    let vector: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
    let payload = serde_json::json!({"tag": "demo", "score": 0.95});
    let body = serde_json::json!({
        "vector": vector,
        "payload": payload,
    })
    .to_string();
    let resp = http_request(
        addr,
        &format!(
            "POST /insert HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
            body.len(),
            body
        ),
    );
    println!("[http] /insert response: {}\n", json_body(&resp));

    // 3. 向量搜索（生产推荐配置）。
    let query: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
    let body = serde_json::json!({
        "query": query,
        "k": 5,
        "nprobe": 50,
        "refine": true,
        "refine_k": 1000,
    })
    .to_string();
    let resp = http_request(
        addr,
        &format!(
            "POST /search HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
            body.len(),
            body
        ),
    );
    println!("[http] /search response: {}\n", json_body(&resp));

    // 4. SQL 过滤搜索。
    let body = serde_json::json!({
        "query": query,
        "k": 5,
        "nprobe": 0,
        "refine": true,
        "refine_k": 100,
        "sql_filter": "score >= 0.9",
    })
    .to_string();
    let resp = http_request(
        addr,
        &format!(
            "POST /search HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
            body.len(),
            body
        ),
    );
    println!("[http] SQL /search response: {}\n", json_body(&resp));

    // 5. 批量插入。
    let vectors: Vec<Vec<f32>> = (0..10)
        .map(|_| (0..dim).map(|_| rand::random::<f32>()).collect())
        .collect();
    let body = serde_json::json!({ "vectors": vectors }).to_string();
    let resp = http_request(
        addr,
        &format!(
            "POST /batch_insert HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
            body.len(),
            body
        ),
    );
    println!("[http] /batch_insert response: {}\n", json_body(&resp));

    println!("[http] example completed on {}", addr);
}
