//! NNG 二进制协议客户端示例。
//!
//! 运行方式：
//!   # 先启动 NNG 服务
//!   cargo run --release --bin vdb-nng-server
//!
//!   # 再运行客户端
//!   cargo run --release --example nng_client
//!
//! 协议格式：[4 bytes: message length][1 byte: command][payload]
//! 响应格式：[4 bytes: length][1 byte: response_code][data]，response_code 0x00 成功，0xFF 错误。

use std::env;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

const CMD_PING: u8 = 0x01;
const CMD_SEARCH: u8 = 0x02;
const CMD_BATCH_SEARCH: u8 = 0x03;
const CMD_INSERT: u8 = 0x04;

const RESP_OK: u8 = 0x00;

fn connect(addr: SocketAddr) -> TcpStream {
    TcpStream::connect_timeout(&addr, Duration::from_secs(2))
        .expect("failed to connect to NNG server; make sure `vdb-nng-server` is running")
}

fn send_command(stream: &mut TcpStream, cmd: u8, payload: &[u8]) -> Vec<u8> {
    let len = 1 + payload.len();
    let mut msg = Vec::with_capacity(4 + len);
    msg.extend_from_slice(&(len as u32).to_le_bytes());
    msg.push(cmd);
    msg.extend_from_slice(payload);
    stream.write_all(&msg).expect("failed to send command");

    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).expect("failed to read response length");
    let resp_len = u32::from_le_bytes(len_buf) as usize;
    if resp_len == 0 || resp_len > 16 * 1024 * 1024 {
        panic!("invalid response length: {}", resp_len);
    }

    let mut resp = vec![0u8; resp_len];
    stream.read_exact(&mut resp).expect("failed to read response body");
    resp
}

fn parse_results(data: &[u8]) -> Vec<(u64, f32)> {
    data.chunks_exact(12)
        .map(|chunk| {
            let id = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let dist = f32::from_le_bytes(chunk[8..12].try_into().unwrap());
            (id, dist)
        })
        .collect()
}

fn main() {
    let addr: SocketAddr = env::var("VDB_NNG_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:9090".to_string())
        .parse()
        .expect("invalid VDB_NNG_ADDR");
    let dim: usize = env::var("VDB_NNG_DIM")
        .unwrap_or_else(|_| "64".to_string())
        .parse()
        .expect("invalid VDB_NNG_DIM");

    let mut stream = connect(addr);

    // 1. PING。
    let resp = send_command(&mut stream, CMD_PING, &[]);
    assert_eq!(resp.len(), 1);
    assert_eq!(resp[0], RESP_OK);
    println!("[nng] PING ok");

    // 2. INSERT。

    let vector: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
    let mut payload = Vec::with_capacity(4 + dim * 4);
    payload.extend_from_slice(&(dim as u32).to_le_bytes());
    for v in &vector {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let resp = send_command(&mut stream, CMD_INSERT, &payload);
    assert_eq!(resp[0], RESP_OK);
    let id = u64::from_le_bytes(resp[1..9].try_into().unwrap());
    println!("[nng] INSERT ok, id={}", id);

    // 3. SEARCH。
    let k = 5u32;
    let nprobe = 50u32;
    let mut payload = Vec::with_capacity(12 + dim * 4);
    payload.extend_from_slice(&k.to_le_bytes());
    payload.extend_from_slice(&nprobe.to_le_bytes());
    payload.extend_from_slice(&(dim as u32).to_le_bytes());
    for v in &vector {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let resp = send_command(&mut stream, CMD_SEARCH, &payload);
    assert_eq!(resp[0], RESP_OK);
    let results = parse_results(&resp[1..]);
    println!("[nng] SEARCH ok, results={:?}", results);

    // 4. BATCH_SEARCH。
    let num_queries = 3u32;
    let mut payload = Vec::with_capacity(16 + num_queries as usize * dim * 4);
    payload.extend_from_slice(&k.to_le_bytes());
    payload.extend_from_slice(&nprobe.to_le_bytes());
    payload.extend_from_slice(&(dim as u32).to_le_bytes());
    payload.extend_from_slice(&num_queries.to_le_bytes());
    for _ in 0..num_queries {
        for _ in 0..dim {
            let v: f32 = rand::random();
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }
    let resp = send_command(&mut stream, CMD_BATCH_SEARCH, &payload);
    assert_eq!(resp[0], RESP_OK);
    let mut offset = 1usize;
    for q in 0..num_queries {
        let count = u32::from_le_bytes(resp[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let results = parse_results(&resp[offset..offset + count * 12]);
        offset += count * 12;
        println!("[nng] BATCH_SEARCH query {}: {:?}", q, results);
    }

    println!("[nng] example completed on {}", addr);
}
