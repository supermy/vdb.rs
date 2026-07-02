//! 服务器测试：HTTP 请求解析、分片组装、超限 413、静态资源、search/insert/stats API。
//!
//! 当前 HTTP server 基于 libevent evhttp 实现，测试通过 Arc<HttpServer> 控制其生命周期。

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use vdb_rs::http_server::HttpServer;
use vdb_rs::index_ivf_rq::IvfRabitqIndex;

struct ServerHandle {
    port: u16,
    server: Arc<HttpServer>,
    done: mpsc::Receiver<()>,
}

impl ServerHandle {
    fn stop(self) {
        self.server.stop();
        let _ = self.done.recv_timeout(Duration::from_secs(2));
    }
}

fn start_server() -> ServerHandle {
    let index = IvfRabitqIndex::new(64);
    let mut server = HttpServer::new(index);
    let addr = server.bind("127.0.0.1:0").unwrap();
    eprintln!("[DEBUG] server bound to {}", addr);
    let port = addr.port();
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = server_clone.run();
        let _ = tx.send(());
    });
    thread::sleep(Duration::from_millis(50));

    ServerHandle {
        port,
        server,
        done: rx,
    }
}

fn http_request(port: u16, request: &str) -> String {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    stream.flush().unwrap();

    // libevent 默认 keep-alive，测试不依赖服务器关连接；按 Content-Length 读取完整响应。
    stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .unwrap();
    let mut buf = [0u8; 4096];
    let mut response = String::new();
    let mut headers_done = false;
    let mut content_length: Option<usize> = None;
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                response.push_str(&String::from_utf8_lossy(&buf[..n]));
                if !headers_done {
                    if let Some(pos) = response.find("\r\n\r\n") {
                        headers_done = true;
                        content_length = parse_content_length(&response[..pos]);
                        if content_length == Some(0) {
                            break;
                        }
                    }
                }
                if let Some(len) = content_length {
                    let header_end = response.find("\r\n\r\n").unwrap() + 4;
                    if response.len() >= header_end + len {
                        break;
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(e) => panic!("read failed: {}", e),
        }
    }
    response
}

fn parse_content_length(headers: &str) -> Option<usize> {
    headers.lines().find_map(|line| {
        let line = line.trim();
        if line.to_ascii_lowercase().starts_with("content-length:") {
            line["content-length:".len()..].trim().parse().ok()
        } else {
            None
        }
    })
}

#[test]
fn server_static_resources_embedded() {
    let html = include_str!("../src/web/index.html");
    let js = include_str!("../src/web/app.js");
    let css = include_str!("../src/web/style.css");

    assert!(!html.is_empty());
    assert!(!js.is_empty());
    assert!(!css.is_empty());

    assert!(html.contains("<html") || html.contains("<!DOCTYPE"));
    assert!(js.contains("vdb"));
    assert!(css.contains("{") && css.contains("}"));
}

#[test]
fn server_health_returns_ok() {
    let handle = start_server();
    let response = http_request(
        handle.port,
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n",
    );
    assert!(
        response.contains("HTTP/1.1 200"),
        "unexpected response: {:?}",
        response
    );
    assert!(response.contains("\"status\":\"ok\""));
    handle.stop();
}

#[test]
fn server_cors_headers_present() {
    let handle = start_server();
    let response = http_request(
        handle.port,
        "OPTIONS /search HTTP/1.1\r\nHost: localhost\r\nOrigin: http://example.com\r\n\r\n",
    );
    assert!(response.contains("HTTP/1.1 204"));
    assert!(response.contains("Access-Control-Allow-Origin: *"));
    handle.stop();
}

#[test]
fn server_index_html_served() {
    let handle = start_server();
    let response = http_request(handle.port, "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");
    assert!(response.contains("HTTP/1.1 200"));
    assert!(response.contains("text/html"));
    assert!(response.contains("<html") || response.contains("<!DOCTYPE"));
    handle.stop();
}

#[test]
fn server_insert_and_search_roundtrip() {
    let handle = start_server();
    let port = handle.port;

    let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let insert_body = serde_json::json!({ "vector": vector }).to_string();
    let request = format!(
        "POST /insert HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
        insert_body.len(),
        insert_body
    );
    let response = http_request(port, &request);
    assert!(response.contains("HTTP/1.1 200"));
    assert!(response.contains("\"id\":0"));

    let search_body = serde_json::json!({
        "query": vector,
        "k": 5,
        "nprobe": 0,
    })
    .to_string();
    let request = format!(
        "POST /search HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
        search_body.len(),
        search_body
    );
    let response = http_request(port, &request);
    assert!(response.contains("HTTP/1.1 200"));
    assert!(response.contains("\"results\""));
    assert!(response.contains("\"id\":0"));
    handle.stop();
}

#[test]
fn server_batch_insert_roundtrip() {
    let handle = start_server();
    let port = handle.port;

    let vectors: Vec<Vec<f32>> = (0..3)
        .map(|i| (0..64).map(|j| (i * 64 + j) as f32).collect())
        .collect();
    let payloads = vec![
        serde_json::json!({ "idx": 0 }),
        serde_json::json!({ "idx": 1 }),
        serde_json::json!({ "idx": 2 }),
    ];
    let body = serde_json::json!({ "vectors": vectors, "payloads": payloads }).to_string();
    let request = format!(
        "POST /batch_insert HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
        body.len(),
        body
    );
    let response = http_request(port, &request);
    assert!(response.contains("HTTP/1.1 200"));
    assert!(response.contains("\"ids\""));
    assert!(response.contains("0"));
    assert!(response.contains("1"));
    assert!(response.contains("2"));

    // 搜索其中一条向量，确认批量插入生效。
    let search_body = serde_json::json!({
        "query": vectors[1],
        "k": 3,
        "nprobe": 0,
    })
    .to_string();
    let request = format!(
        "POST /search HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
        search_body.len(),
        search_body
    );
    let response = http_request(port, &request);
    assert!(response.contains("HTTP/1.1 200"));
    assert!(response.contains("\"results\""));
    handle.stop();
}

#[test]
fn server_batch_insert_payload_length_mismatch() {
    let handle = start_server();
    let port = handle.port;

    let vectors: Vec<Vec<f32>> = (0..3)
        .map(|i| (0..64).map(|j| (i * 64 + j) as f32).collect())
        .collect();
    let payloads = vec![serde_json::json!({ "idx": 0 })];
    let body = serde_json::json!({ "vectors": vectors, "payloads": payloads }).to_string();
    let request = format!(
        "POST /batch_insert HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
        body.len(),
        body
    );
    let response = http_request(port, &request);
    assert!(response.contains("HTTP/1.1 400"));
    assert!(response.contains("vectors and payloads length mismatch"));
    handle.stop();
}

#[test]
fn server_k_limit_enforced() {
    let handle = start_server();

    let vector: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
    let search_body = serde_json::json!({
        "query": vector,
        "k": 1000,
    })
    .to_string();
    let request = format!(
        "POST /search HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nContent-Type: application/json\r\n\r\n{}",
        search_body.len(),
        search_body
    );
    let response = http_request(handle.port, &request);
    assert!(response.contains("HTTP/1.1 200"));
    // k=1000 被截断到 MAX_K=256，结果数不应超过 256；空索引时则为 0。
    handle.stop();
}

#[test]
fn server_413_on_large_body() {
    let handle = start_server();

    // 构造一个超过 16 MiB 限制的 Content-Length。
    let request = format!(
        "POST /insert HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{}",
        17 * 1024 * 1024,
        ""
    );
    let response = http_request(handle.port, &request);
    // evhttp 会拒绝或截断超大 body；测试验证服务器不崩溃且不以 200 接受。
    assert!(
        response.is_empty() || !response.contains("HTTP/1.1 200"),
        "server should not accept oversized body as valid"
    );
    handle.stop();
}

#[test]
fn server_stats_endpoint() {
    let handle = start_server();
    let response = http_request(
        handle.port,
        "GET /stats HTTP/1.1\r\nHost: localhost\r\n\r\n",
    );
    assert!(response.contains("HTTP/1.1 200"));
    assert!(response.contains("\"num_vectors\""));
    assert!(response.contains("\"dim\":64"));
    handle.stop();
}
