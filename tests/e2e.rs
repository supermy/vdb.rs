//! 端到端测试：模拟真实客户端交互。
//!
//! 当前覆盖 NNG 二进制协议：PING、INSERT、SEARCH、BATCH_SEARCH、IMPORT_JSON、EXPORT_JSON。
//! HTTP server 仍为 placeholder，待实现后再补充 HTTP 端到端用例。

use std::io::{Read, Write};
use std::net::TcpStream;
use std::thread;
use std::time::Duration;

const RESP_OK: u8 = 0x00;
const RESP_ERR: u8 = 0xFF;
const CMD_PING: u8 = 0x01;
const CMD_SEARCH: u8 = 0x02;
const CMD_BATCH_SEARCH: u8 = 0x03;
const CMD_INSERT: u8 = 0x04;
const CMD_IMPORT_JSON: u8 = 0x05;
const CMD_EXPORT_JSON: u8 = 0x06;

fn write_message(stream: &mut TcpStream, cmd: u8, payload: &[u8]) -> std::io::Result<()> {
    let len = (1 + payload.len()) as u32;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&[cmd])?;
    stream.write_all(payload)?;
    stream.flush()
}

fn read_response(stream: &mut TcpStream) -> std::io::Result<(u8, Vec<u8>)> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok((buf[0], buf[1..].to_vec()))
}

fn start_test_server() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    thread::spawn(move || {
        use vdb_rs::index_ivf_rq::IvfRabitqIndex;
        let index = std::sync::Arc::new(std::sync::Mutex::new(IvfRabitqIndex::new(64)));
        for stream in listener.incoming() {
            let index = std::sync::Arc::clone(&index);
            thread::spawn(move || -> Result<(), String> {
                let mut stream = stream.unwrap();
                loop {
                    let mut len_buf = [0u8; 4];
                    if stream.read_exact(&mut len_buf).is_err() {
                        return Ok(());
                    }
                    let len = u32::from_le_bytes(len_buf) as usize;
                    if len == 0 || len > 16 * 1024 * 1024 {
                        return Ok(());
                    }
                    let mut msg = vec![0u8; len];
                    if stream.read_exact(&mut msg).is_err() {
                        return Ok(());
                    }
                    let cmd = msg[0];
                    let payload = &msg[1..];

                    let result: Result<Vec<u8>, String> = match cmd {
                        CMD_PING => Ok(Vec::new()),
                        CMD_INSERT => {
                            if payload.len() < 4 {
                                Err("short".to_string())
                            } else {
                                let dim =
                                    u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                                let bytes = &payload[4..];
                                let vector: Vec<f32> = bytes
                                    .chunks_exact(4)
                                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                                    .collect();
                                if vector.len() != dim {
                                    Err("dim mismatch".to_string())
                                } else {
                                    let id = index.lock().unwrap().add(&vector);
                                    Ok(id.to_le_bytes().to_vec())
                                }
                            }
                        }
                        CMD_SEARCH => {
                            if payload.len() < 12 {
                                Err("short".to_string())
                            } else {
                                let k =
                                    u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                                let nprobe =
                                    u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
                                let dim =
                                    u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
                                let vector: Vec<f32> = payload[12..]
                                    .chunks_exact(4)
                                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                                    .collect();
                                if vector.len() != dim {
                                    Err("dim mismatch".to_string())
                                } else {
                                    let results = index.lock().unwrap().search(&vector, k, nprobe);
                                    let mut out = Vec::with_capacity(results.len() * 12);
                                    for (id, dist) in results {
                                        out.extend_from_slice(&id.to_le_bytes());
                                        out.extend_from_slice(&dist.to_le_bytes());
                                    }
                                    Ok(out)
                                }
                            }
                        }
                        CMD_BATCH_SEARCH => {
                            if payload.len() < 16 {
                                Err("short".to_string())
                            } else {
                                let k =
                                    u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
                                let nprobe =
                                    u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
                                let dim =
                                    u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
                                let num_queries =
                                    u32::from_le_bytes(payload[12..16].try_into().unwrap())
                                        as usize;
                                let expected_bytes = num_queries * dim * 4;
                                if payload.len() < 16 + expected_bytes {
                                    Err("short queries".to_string())
                                } else {
                                    let idx = index.lock().unwrap();
                                    let mut out = Vec::new();
                                    out.extend_from_slice(&(num_queries as u32).to_le_bytes());
                                    for q in 0..num_queries {
                                        let start = 16 + q * dim * 4;
                                        let vector: Vec<f32> = payload[start..start + dim * 4]
                                            .chunks_exact(4)
                                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                                            .collect();
                                        let results = idx.search(&vector, k, nprobe);
                                        out.extend_from_slice(
                                            &(results.len() as u32).to_le_bytes(),
                                        );
                                        for (id, dist) in results {
                                            out.extend_from_slice(&id.to_le_bytes());
                                            out.extend_from_slice(&dist.to_le_bytes());
                                        }
                                    }
                                    Ok(out)
                                }
                            }
                        }
                        CMD_IMPORT_JSON => {
                            let json: serde_json::Value = match serde_json::from_slice(payload) {
                                Ok(v) => v,
                                Err(e) => return Err(e.to_string()),
                            };
                            let arr = match json.get("vectors").and_then(|v| v.as_array()) {
                                Some(a) => a,
                                None => return Err("missing vectors".to_string()),
                            };
                            let mut count: u32 = 0;
                            {
                                let mut idx = index.lock().unwrap();
                                for v in arr {
                                    let inner = match v.as_array() {
                                        Some(a) => a,
                                        None => return Err("not array".to_string()),
                                    };
                                    let mut vector = Vec::with_capacity(inner.len());
                                    for x in inner {
                                        match x.as_f64() {
                                            Some(f) => vector.push(f as f32),
                                            None => return Err("not f64".to_string()),
                                        }
                                    }
                                    idx.add(&vector);
                                    count += 1;
                                }
                            }
                            Ok(count.to_le_bytes().to_vec())
                        }
                        CMD_EXPORT_JSON => {
                            let idx = index.lock().unwrap();
                            let vectors: Vec<Option<&[f32]>> =
                                (0..idx.len() as u64).map(|id| idx.raw_vector(id)).collect();
                            serde_json::to_vec(&vectors).map_err(|e| e.to_string())
                        }
                        _ => Err("unsupported".to_string()),
                    };

                    match result {
                        Ok(data) => {
                            let _ = write_message(&mut stream, RESP_OK, &data);
                        }
                        Err(e) => {
                            let _ = write_message(&mut stream, RESP_ERR, e.as_bytes());
                        }
                    }
                }
            });
        }
    });

    port
}

#[test]
fn e2e_nng_ping() {
    let port = start_test_server();
    thread::sleep(Duration::from_millis(50));

    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
    write_message(&mut stream, CMD_PING, &[]).unwrap();
    let (code, data) = read_response(&mut stream).unwrap();
    assert_eq!(code, RESP_OK);
    assert!(data.is_empty());
}

#[test]
fn e2e_nng_insert_and_search() {
    let port = start_test_server();
    thread::sleep(Duration::from_millis(50));

    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();

    let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let mut payload = Vec::new();
    payload.extend_from_slice(&(64u32).to_le_bytes());
    payload.extend_from_slice(
        &vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    write_message(&mut stream, CMD_INSERT, &payload).unwrap();

    let (code, data) = read_response(&mut stream).unwrap();
    assert_eq!(code, RESP_OK);
    assert_eq!(u64::from_le_bytes(data[..8].try_into().unwrap()), 0);

    let mut search_payload = Vec::new();
    search_payload.extend_from_slice(&(10u32).to_le_bytes());
    search_payload.extend_from_slice(&(0u32).to_le_bytes());
    search_payload.extend_from_slice(&(64u32).to_le_bytes());
    search_payload.extend_from_slice(
        &vector
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    write_message(&mut stream, CMD_SEARCH, &search_payload).unwrap();

    let (code2, data2) = read_response(&mut stream).unwrap();
    assert_eq!(code2, RESP_OK);
    assert!(!data2.is_empty());
    let first_id = u64::from_le_bytes(data2[0..8].try_into().unwrap());
    assert_eq!(first_id, 0);
}

#[test]
fn e2e_nng_export_json() {
    let port = start_test_server();
    thread::sleep(Duration::from_millis(50));

    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
    write_message(&mut stream, CMD_EXPORT_JSON, &[]).unwrap();
    let (code, data) = read_response(&mut stream).unwrap();
    assert_eq!(code, RESP_OK);
    let exported: serde_json::Value = serde_json::from_slice(&data).unwrap();
    assert!(exported.is_array());
}

#[test]
fn e2e_nng_batch_search() {
    let port = start_test_server();
    thread::sleep(Duration::from_millis(50));

    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();

    let vectors: Vec<Vec<f32>> = (0..5)
        .map(|i| (0..64).map(|j| (i * 64 + j) as f32).collect())
        .collect();
    for v in &vectors {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(64u32).to_le_bytes());
        payload.extend_from_slice(&v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>());
        write_message(&mut stream, CMD_INSERT, &payload).unwrap();
        let (code, _) = read_response(&mut stream).unwrap();
        assert_eq!(code, RESP_OK);
    }

    let k = 3u32;
    let nprobe = 0u32;
    let dim = 64u32;
    let num_queries = 2u32;
    let mut batch_payload = Vec::new();
    batch_payload.extend_from_slice(&k.to_le_bytes());
    batch_payload.extend_from_slice(&nprobe.to_le_bytes());
    batch_payload.extend_from_slice(&dim.to_le_bytes());
    batch_payload.extend_from_slice(&num_queries.to_le_bytes());
    for v in &vectors[..2] {
        batch_payload
            .extend_from_slice(&v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>());
    }
    write_message(&mut stream, CMD_BATCH_SEARCH, &batch_payload).unwrap();

    let (code, data) = read_response(&mut stream).unwrap();
    assert_eq!(code, RESP_OK);
    assert!(!data.is_empty());
    let returned_queries = u32::from_le_bytes(data[0..4].try_into().unwrap());
    assert_eq!(returned_queries, num_queries);
}

#[test]
fn e2e_nng_import_json() {
    let port = start_test_server();
    thread::sleep(Duration::from_millis(50));

    let vectors: Vec<Vec<f32>> = (0..4)
        .map(|i| (0..64).map(|j| (i * 64 + j) as f32).collect())
        .collect();
    let json = serde_json::json!({ "vectors": vectors });
    let payload = serde_json::to_vec(&json).unwrap();

    let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port)).unwrap();
    write_message(&mut stream, CMD_IMPORT_JSON, &payload).unwrap();

    let (code, data) = read_response(&mut stream).unwrap();
    assert_eq!(code, RESP_OK);
    let count = u32::from_le_bytes(data[..4].try_into().unwrap());
    assert_eq!(count, 4);

    write_message(&mut stream, CMD_EXPORT_JSON, &[]).unwrap();
    let (code2, data2) = read_response(&mut stream).unwrap();
    assert_eq!(code2, RESP_OK);
    let exported: Vec<Option<Vec<f32>>> = serde_json::from_slice(&data2).unwrap();
    assert_eq!(exported.len(), 4);
}
