//! 基于原始 TCP 的二进制协议高性能服务。
//!
//! 协议格式：[4 bytes: message length][1 byte: command][payload]
//! 支持的命令：PING(0x01)、SEARCH(0x02)、BATCH_SEARCH(0x03)、
//! INSERT(0x04)、IMPORT_JSON(0x05)、EXPORT_JSON(0x06)。
//!
//! 响应统一以 [4 bytes: length][1 byte: response_code] 开头，错误码为 0xFF。
//!
//! 网络层实现：
//! - Unix 使用 POSIX socket 直接系统调用（socket/bind/listen/accept/read/write），
//!   避免 `std::net` 的额外缓冲与抽象开销，降低延迟。
//! - Windows 回退到 `std::net`，保证 CI 跨平台编译通过。

#[cfg(not(unix))]
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use vdb_rs::index_ivf_rq::IvfRabitqIndex;

const RESP_OK: u8 = 0x00;
const RESP_ERR: u8 = 0xFF;

const CMD_PING: u8 = 0x01;
const CMD_SEARCH: u8 = 0x02;
const CMD_BATCH_SEARCH: u8 = 0x03;
const CMD_INSERT: u8 = 0x04;
const CMD_IMPORT_JSON: u8 = 0x05;
const CMD_EXPORT_JSON: u8 = 0x06;

fn main() -> std::io::Result<()> {
    // untested: nng-server 为独立二进制入口，生命周期由监听循环持有，单元测试仅覆盖序列化函数。
    run_server("0.0.0.0:9090")
}

#[cfg(unix)]
fn run_server(addr: &str) -> std::io::Result<()> {
    use libc::{
        AF_INET, INADDR_ANY, SO_REUSEADDR, SOCK_STREAM, SOL_SOCKET, accept, bind, close, htons,
        listen, setsockopt, sockaddr_in, socket, socklen_t,
    };
    use std::ffi::c_int;
    use std::mem::size_of;
    use std::net::Ipv4Addr;
    use std::str::FromStr;

    let (ip_str, port_str) = addr
        .rsplit_once(':')
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid address"))?;
    let port: u16 = port_str
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid port"))?;
    let ip = Ipv4Addr::from_str(ip_str)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid ip"))?;

    unsafe {
        let fd = socket(AF_INET, SOCK_STREAM, 0);
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }

        let reuse: c_int = 1;
        if setsockopt(
            fd,
            SOL_SOCKET,
            SO_REUSEADDR,
            &reuse as *const _ as *const libc::c_void,
            size_of::<c_int>() as socklen_t,
        ) < 0
        {
            close(fd);
            return Err(std::io::Error::last_os_error());
        }

        let mut sin: sockaddr_in = std::mem::zeroed();
        sin.sin_family = AF_INET as libc::sa_family_t;
        sin.sin_port = htons(port);
        sin.sin_addr.s_addr = if ip_str == "0.0.0.0" {
            INADDR_ANY.to_be()
        } else {
            u32::from(ip).to_be()
        };

        if bind(
            fd,
            &sin as *const _ as *const libc::sockaddr,
            size_of::<sockaddr_in>() as socklen_t,
        ) < 0
        {
            close(fd);
            return Err(std::io::Error::last_os_error());
        }

        if listen(fd, 128) < 0 {
            close(fd);
            return Err(std::io::Error::last_os_error());
        }

        eprintln!("vdb-nng-server listening on tcp://{}", addr);

        let index = Arc::new(Mutex::new(IvfRabitqIndex::new(64)));

        loop {
            let mut client_addr: sockaddr_in = std::mem::zeroed();
            let mut client_len: socklen_t = size_of::<sockaddr_in>() as socklen_t;
            let client_fd = accept(
                fd,
                &mut client_addr as *mut _ as *mut libc::sockaddr,
                &mut client_len,
            );
            if client_fd < 0 {
                eprintln!("accept error: {}", std::io::Error::last_os_error());
                continue;
            }

            let index = Arc::clone(&index);
            std::thread::spawn(move || {
                if let Err(e) = handle_posix_connection(client_fd, index) {
                    eprintln!("connection error: {}", e);
                }
                close(client_fd);
            });
        }
    }
}

#[cfg(unix)]
fn handle_posix_connection(
    fd: std::os::unix::io::RawFd,
    index: Arc<Mutex<IvfRabitqIndex>>,
) -> std::io::Result<()> {
    use libc::close;

    loop {
        let mut len_buf = [0u8; 4];
        if read_exact(fd, &mut len_buf).is_err() {
            break;
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        if len == 0 || len > 16 * 1024 * 1024 {
            write_response_posix(fd, RESP_ERR, &[])?;
            break;
        }
        let mut msg = vec![0u8; len];
        if read_exact(fd, &mut msg).is_err() {
            break;
        }

        let cmd = msg[0];
        let payload = &msg[1..];

        let result = match cmd {
            CMD_PING => handle_ping(),
            CMD_SEARCH => handle_search(&index, payload),
            CMD_BATCH_SEARCH => handle_batch_search(&index, payload),
            CMD_INSERT => handle_insert(&index, payload),
            CMD_IMPORT_JSON => handle_import_json(&index, payload),
            CMD_EXPORT_JSON => handle_export_json(&index),
            _ => Err("unknown command".to_string()),
        };

        match result {
            Ok(data) => write_response_posix(fd, RESP_OK, &data)?,
            Err(e) => {
                let msg = e.into_bytes();
                write_response_posix(fd, RESP_ERR, &msg)?;
            }
        }
    }
    unsafe {
        close(fd);
    }
    Ok(())
}

#[cfg(unix)]
fn read_exact(fd: std::os::unix::io::RawFd, buf: &mut [u8]) -> std::io::Result<()> {
    use libc::recv;
    let mut total = 0;
    while total < buf.len() {
        let n = unsafe {
            recv(
                fd,
                buf[total..].as_mut_ptr() as *mut libc::c_void,
                buf.len() - total,
                0,
            )
        };
        if n <= 0 {
            return Err(std::io::Error::last_os_error());
        }
        total += n as usize;
    }
    Ok(())
}

#[cfg(unix)]
fn write_response_posix(
    fd: std::os::unix::io::RawFd,
    code: u8,
    data: &[u8],
) -> std::io::Result<()> {
    use libc::send;
    let len = (1 + data.len()) as u32;
    let mut out = Vec::with_capacity(5 + data.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.push(code);
    out.extend_from_slice(data);

    let mut total = 0;
    while total < out.len() {
        let n = unsafe {
            send(
                fd,
                out[total..].as_ptr() as *const libc::c_void,
                out.len() - total,
                0,
            )
        };
        if n < 0 {
            return Err(std::io::Error::last_os_error());
        }
        total += n as usize;
    }
    Ok(())
}

#[cfg(not(unix))]
fn run_server(addr: &str) -> std::io::Result<()> {
    // untested: Windows 回退路径未在 CI 矩阵覆盖，仅保证编译通过。
    use std::net::TcpListener;

    let listener = TcpListener::bind(addr)?;
    eprintln!("vdb-nng-server listening on tcp://{}", addr);

    let index = Arc::new(Mutex::new(IvfRabitqIndex::new(64)));

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let index = Arc::clone(&index);
                std::thread::spawn(move || {
                    if let Err(e) = handle_std_connection(stream, index) {
                        eprintln!("connection error: {}", e);
                    }
                });
            }
            Err(e) => eprintln!("accept error: {}", e),
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn handle_std_connection(
    mut stream: std::net::TcpStream,
    index: Arc<Mutex<IvfRabitqIndex>>,
) -> std::io::Result<()> {
    loop {
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(_) => break,
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        if len == 0 || len > 16 * 1024 * 1024 {
            write_response(&mut stream, RESP_ERR, &[])?;
            break;
        }
        let mut msg = vec![0u8; len];
        stream.read_exact(&mut msg)?;

        let cmd = msg[0];
        let payload = &msg[1..];

        let result = match cmd {
            CMD_PING => handle_ping(),
            CMD_SEARCH => handle_search(&index, payload),
            CMD_BATCH_SEARCH => handle_batch_search(&index, payload),
            CMD_INSERT => handle_insert(&index, payload),
            CMD_IMPORT_JSON => handle_import_json(&index, payload),
            CMD_EXPORT_JSON => handle_export_json(&index),
            _ => Err("unknown command".to_string()),
        };

        match result {
            Ok(data) => write_response(&mut stream, RESP_OK, &data)?,
            Err(e) => {
                let msg = e.into_bytes();
                write_response(&mut stream, RESP_ERR, &msg)?;
            }
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn write_response(stream: &mut std::net::TcpStream, code: u8, data: &[u8]) -> std::io::Result<()> {
    let len = (1 + data.len()) as u32;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&[code])?;
    if !data.is_empty() {
        stream.write_all(data)?;
    }
    stream.flush()
}

fn handle_ping() -> Result<Vec<u8>, String> {
    Ok(Vec::new())
}

fn handle_search(index: &Arc<Mutex<IvfRabitqIndex>>, payload: &[u8]) -> Result<Vec<u8>, String> {
    if payload.len() < 12 {
        return Err("search payload too short".to_string());
    }
    let k = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let nprobe = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
    if payload.len() != 12 + dim * 4 {
        return Err("search payload length mismatch".to_string());
    }
    let query = read_f32s(&payload[12..], dim)?;

    let idx = index.lock().unwrap();
    let results = idx.search(&query, k, nprobe);
    Ok(serialize_results(&results))
}

fn handle_batch_search(
    index: &Arc<Mutex<IvfRabitqIndex>>,
    payload: &[u8],
) -> Result<Vec<u8>, String> {
    if payload.len() < 16 {
        return Err("batch search payload too short".to_string());
    }
    let k = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let nprobe = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(payload[8..12].try_into().unwrap()) as usize;
    let num_queries = u32::from_le_bytes(payload[12..16].try_into().unwrap()) as usize;
    let expected = 16 + num_queries * dim * 4;
    if payload.len() != expected {
        return Err("batch search payload length mismatch".to_string());
    }

    let idx = index.lock().unwrap();
    let mut out = Vec::new();
    for q in 0..num_queries {
        let off = 16 + q * dim * 4;
        let query = read_f32s(&payload[off..off + dim * 4], dim)?;
        let results = idx.search(&query, k, nprobe);
        out.extend_from_slice(&(results.len() as u32).to_le_bytes());
        out.extend_from_slice(&serialize_results(&results));
    }
    Ok(out)
}

fn handle_insert(index: &Arc<Mutex<IvfRabitqIndex>>, payload: &[u8]) -> Result<Vec<u8>, String> {
    if payload.len() < 4 {
        return Err("insert payload too short".to_string());
    }
    let dim = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    if payload.len() != 4 + dim * 4 {
        return Err("insert payload length mismatch".to_string());
    }
    let vector = read_f32s(&payload[4..], dim)?;
    let mut idx = index.lock().unwrap();
    let id = idx.add(&vector);
    Ok(id.to_le_bytes().to_vec())
}

fn handle_import_json(
    index: &Arc<Mutex<IvfRabitqIndex>>,
    payload: &[u8],
) -> Result<Vec<u8>, String> {
    let text = std::str::from_utf8(payload).map_err(|_| "invalid utf8")?;
    let vectors: Vec<Vec<f32>> = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let mut idx = index.lock().unwrap();
    for vector in &vectors {
        idx.add(vector);
    }
    Ok((vectors.len() as u64).to_le_bytes().to_vec())
}

fn handle_export_json(index: &Arc<Mutex<IvfRabitqIndex>>) -> Result<Vec<u8>, String> {
    let idx = index.lock().unwrap();
    let vectors: Vec<Option<&[f32]>> = (0..idx.len() as u64).map(|id| idx.raw_vector(id)).collect();
    serde_json::to_vec(&vectors).map_err(|e| e.to_string())
}

fn read_f32s(bytes: &[u8], dim: usize) -> Result<Vec<f32>, String> {
    if bytes.len() != dim * 4 {
        return Err("f32 slice length mismatch".to_string());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn serialize_results(results: &[(u64, f32)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(results.len() * 12);
    for (id, dist) in results {
        out.extend_from_slice(&id.to_le_bytes());
        out.extend_from_slice(&dist.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_results() {
        let r = vec![(1u64, 0.5f32), (2u64, 1.5f32)];
        let bytes = serialize_results(&r);
        assert_eq!(bytes.len(), 24);
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 1);
    }

    #[test]
    fn test_read_f32s() {
        let v = vec![1.0f32, 2.0f32];
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let decoded = read_f32s(&bytes, 2).unwrap();
        assert_eq!(decoded, v);
    }
}
