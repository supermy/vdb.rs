//! HTTP 服务核心逻辑，基于 libevent evhttp 实现。
//!
//! 为什么用 libevent：
//! 1. 事件驱动单线程/多线程模型，避免 `std::net` 每个连接一个 OS 线程的开销。
//! 2. evhttp 已经处理 HTTP 解析、分块编码、keep-alive 等细节，减少自研解析器 bug。
//! 3. 通过 C FFI 调用系统库，符合项目“仅在 Arrow/Tantivy/SIMD 处例外使用 C/FFI”的约束扩展。
//!
//! 所有权说明：
//! - `HttpServer` 持有 `Arc<Mutex<IvfRabitqIndex>>` 与 libevent 句柄。
//! - 运行期间把 `Arc` 的原始指针作为 evhttp 回调参数；`run` 返回前该 Arc 一直有效。
//! - 事件循环 `event_base_dispatch` 阻塞当前线程，直到调用 `stop` 触发退出。

use crate::index_ivf_rq::{IvfRabitqIndex, Payload};
use crate::search::{SearchOptions, search};
use libc::{c_char, c_int, c_ushort, c_void, sockaddr, sockaddr_in, socklen_t};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::os::raw::c_long;
use std::ptr;
use std::sync::{Arc, Mutex};

const INDEX_HTML: &str = include_str!("web/index.html");
const APP_JS: &str = include_str!("web/app.js");
const STYLE_CSS: &str = include_str!("web/style.css");

const MAX_K: usize = 256;
/// HTTP 请求体大小上限（16 MiB），与 NNG 协议保持一致，防止超大 JSON 拖垮服务。
const MAX_BODY_SIZE: libc::ssize_t = 16 * 1024 * 1024;
/// HTTP 请求超时（秒）。evhttp 会在超时时自动关闭连接，避免慢客户端占用事件循环。
const HTTP_TIMEOUT_SECS: c_int = 30;

// libevent evhttp 请求方法常量。
const EVHTTP_REQ_GET: c_int = 1 << 0;
const EVHTTP_REQ_POST: c_int = 1 << 1;
const EVHTTP_REQ_OPTIONS: c_int = 1 << 5;

#[repr(C)]
struct event_base {
    _private: [u8; 0],
}
#[repr(C)]
struct evhttp {
    _private: [u8; 0],
}
#[repr(C)]
struct evhttp_request {
    _private: [u8; 0],
}
#[repr(C)]
struct evhttp_bound_socket {
    _private: [u8; 0],
}
#[repr(C)]
struct evbuffer {
    _private: [u8; 0],
}
#[repr(C)]
struct evkeyvalq {
    _private: [u8; 0],
}

#[repr(C)]
struct timeval {
    tv_sec: c_long,
    tv_usec: c_long,
}

unsafe extern "C" {
    fn event_base_new() -> *mut event_base;
    fn event_base_dispatch(base: *mut event_base) -> c_int;
    fn event_base_loopexit(base: *mut event_base, tv: *const timeval) -> c_int;
    fn event_base_free(base: *mut event_base);

    fn evhttp_new(base: *mut event_base) -> *mut evhttp;
    fn evhttp_free(http: *mut evhttp);
    fn evhttp_set_allowed_methods(http: *mut evhttp, methods: c_ushort);
    fn evhttp_bind_socket_with_handle(
        http: *mut evhttp,
        address: *const c_char,
        port: c_ushort,
    ) -> *mut evhttp_bound_socket;
    fn evhttp_bound_socket_get_fd(sock: *mut evhttp_bound_socket) -> c_int;
    fn evhttp_set_gencb(
        http: *mut evhttp,
        cb: Option<unsafe extern "C" fn(*mut evhttp_request, *mut c_void)>,
        arg: *mut c_void,
    );
    fn evhttp_set_max_body_size(http: *mut evhttp, max_body_size: libc::ssize_t);
    fn evhttp_set_timeout(http: *mut evhttp, timeout_in_secs: c_int);

    fn evhttp_request_get_uri(req: *mut evhttp_request) -> *const c_char;
    fn evhttp_request_get_command(req: *mut evhttp_request) -> c_int;
    fn evhttp_request_get_input_buffer(req: *mut evhttp_request) -> *mut evbuffer;
    fn evhttp_request_get_output_buffer(req: *mut evhttp_request) -> *mut evbuffer;
    fn evhttp_request_get_output_headers(req: *mut evhttp_request) -> *mut evkeyvalq;

    fn evhttp_add_header(
        headers: *mut evkeyvalq,
        key: *const c_char,
        value: *const c_char,
    ) -> c_int;
    fn evhttp_send_reply(
        req: *mut evhttp_request,
        code: c_int,
        reason: *const c_char,
        databuf: *mut evbuffer,
    );

    fn evbuffer_add(buffer: *mut evbuffer, data: *const c_void, len: usize) -> c_int;
    fn evbuffer_get_length(buffer: *const evbuffer) -> usize;
    fn evbuffer_pullup(buffer: *mut evbuffer, size: isize) -> *mut u8;

    fn getsockname(sockfd: c_int, addr: *mut sockaddr, addrlen: *mut socklen_t) -> c_int;
}

/// HTTP 服务器状态。
pub struct HttpServer {
    index: Arc<Mutex<IvfRabitqIndex>>,
    base: *mut event_base,
    http: *mut evhttp,
    bound: *mut evhttp_bound_socket,
}

impl HttpServer {
    pub fn new(index: IvfRabitqIndex) -> Self {
        unsafe {
            let base = event_base_new();
            assert!(!base.is_null(), "event_base_new failed");
            let http = evhttp_new(base);
            assert!(!http.is_null(), "evhttp_new failed");

            // 默认 evhttp 不处理 OPTIONS，需显式开启以支持 CORS 预检。
            let methods = (EVHTTP_REQ_GET | EVHTTP_REQ_POST | EVHTTP_REQ_OPTIONS) as c_ushort;
            evhttp_set_allowed_methods(http, methods);

            // 限制请求体大小与超时：在 C 层直接拒绝/断开超大或超慢请求，
            // 避免 Rust 回调中再分配大内存或长时间持有事件循环。
            evhttp_set_max_body_size(http, MAX_BODY_SIZE);
            evhttp_set_timeout(http, HTTP_TIMEOUT_SECS);

            Self {
                index: Arc::new(Mutex::new(index)),
                base,
                http,
                bound: ptr::null_mut(),
            }
        }
    }

    /// 绑定到指定地址。
    ///
    /// 为什么先创建 base/http 再 bind：libevent 要求 evhttp 对象存在后才能绑定监听端口。
    pub fn bind<A: std::net::ToSocketAddrs>(&mut self, addr: A) -> std::io::Result<SocketAddr> {
        let addr = addr.to_socket_addrs()?.next().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "empty address")
        })?;

        let c_addr = CString::new(addr.ip().to_string()).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid address string")
        })?;

        unsafe {
            let bound = evhttp_bind_socket_with_handle(self.http, c_addr.as_ptr(), addr.port());
            if bound.is_null() {
                // untested: 端口冲突场景由部署环境触发，单元测试使用随机端口避免冲突。
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AddrInUse,
                    "evhttp_bind_socket_with_handle failed",
                ));
            }
            self.bound = bound;

            // 当传入端口为 0 时，需通过 getsockname 获取内核分配的真实端口。
            let actual_addr = if addr.port() == 0 {
                get_bound_socket_addr(self.bound)?
            } else {
                addr
            };

            log::info!("vdb-server listening on {}", actual_addr);
            Ok(actual_addr)
        }
    }

    /// 启动事件循环。
    ///
    /// 把 `Arc<Mutex<IvfRabitqIndex>>` 的原始指针传给回调；
    /// `run` 返回前该 Arc 一直存活，因此指针有效。
    pub fn run(&self) -> std::io::Result<()> {
        unsafe {
            // 在堆上分配一份 Arc 克隆，把指向该 Arc 的指针传给 C 回调。
            // 这样回调拿到的是 *const Arc<Mutex<IvfRabitqIndex>>，
            // 而不是 Arc::into_raw 返回的 *const Mutex；后者会被错误解引用为 Arc。
            let index_arc = Box::new(Arc::clone(&self.index));
            let index_ptr = Box::into_raw(index_arc) as *mut c_void;
            evhttp_set_gencb(self.http, Some(http_generic_callback), index_ptr);

            let ret = event_base_dispatch(self.base);
            // 回收堆上的 Arc，避免泄漏。
            let _ = Box::from_raw(index_ptr as *mut Arc<Mutex<IvfRabitqIndex>>);
            if ret == 0 {
                Ok(())
            } else {
                // untested: event_base_dispatch 失败通常由 libevent 内部错误引起，难以在测试中构造。
                Err(std::io::Error::other("event_base_dispatch failed"))
            }
        }
    }

    /// 停止事件循环。
    ///
    /// 通过 event_base_loopexit 让 dispatch 尽快退出；
    /// 可在另一个线程调用。
    pub fn stop(&self) {
        unsafe {
            let tv = timeval {
                tv_sec: 0,
                tv_usec: 0,
            };
            event_base_loopexit(self.base, &tv);
        }
    }
}

// HttpServer 内部持有 libevent 的 C 指针；
// 显式标记 Send/Sync 的理由：
// - `base`/`http` 仅在 `run` 中由单个事件循环线程使用，`run` 本身不跨线程调用。
// - `bound` 在 bind 后只读。
// - `stop` 使用 libevent 的 `event_base_loopexit`，该函数设计为可从其他线程安全调用，
//   内部通过唤醒机制通知事件循环退出。
// - `index` 本身是 `Arc<Mutex<...>>`，已保证线程安全。
// 因此外部可以将 HttpServer 移动/共享给运行事件循环的线程与调用 stop 的线程。
unsafe impl Send for HttpServer {}
unsafe impl Sync for HttpServer {}

impl Drop for HttpServer {
    fn drop(&mut self) {
        unsafe {
            if !self.http.is_null() {
                evhttp_free(self.http);
            }
            if !self.base.is_null() {
                event_base_free(self.base);
            }
        }
    }
}

unsafe extern "C" fn http_generic_callback(req: *mut evhttp_request, arg: *mut c_void) {
    unsafe {
        if req.is_null() {
            return;
        }
        let index = &*(arg as *const Arc<Mutex<IvfRabitqIndex>>);

        let method = evhttp_request_get_command(req);
        let uri = CStr::from_ptr(evhttp_request_get_uri(req))
            .to_string_lossy()
            .into_owned();

        // 读取 body。
        let body = read_request_body(req);
        let body_str = String::from_utf8_lossy(&body);

        // 执行路由与业务处理。
        let (status, content_type, body_out) = match method {
            m if m == EVHTTP_REQ_OPTIONS => (204, "text/plain".to_string(), Vec::new()),
            m if m == EVHTTP_REQ_GET && uri == "/health" => {
                log::debug!("handling /health request");
                let json = serde_json::json!({ "status": "ok" });
                let bytes = serde_json::to_vec(&json).unwrap_or_default();
                (200, "application/json".to_string(), bytes)
            }
            m if m == EVHTTP_REQ_GET && uri == "/" => {
                (200, "text/html".to_string(), INDEX_HTML.as_bytes().to_vec())
            }
            m if m == EVHTTP_REQ_GET && uri == "/app.js" => (
                200,
                "application/javascript".to_string(),
                APP_JS.as_bytes().to_vec(),
            ),
            m if m == EVHTTP_REQ_GET && uri == "/style.css" => {
                (200, "text/css".to_string(), STYLE_CSS.as_bytes().to_vec())
            }
            m if m == EVHTTP_REQ_POST && uri == "/search" => handle_search(index, &body_str),
            m if m == EVHTTP_REQ_POST && uri == "/insert" => handle_insert(index, &body_str),
            m if m == EVHTTP_REQ_POST && uri == "/batch_insert" => {
                handle_batch_insert(index, &body_str)
            }
            m if m == EVHTTP_REQ_GET && uri == "/stats" => handle_stats(index),
            _ => {
                let json = serde_json::json!({ "error": "not found" });
                let bytes = serde_json::to_vec(&json).unwrap_or_default();
                (404, "application/json".to_string(), bytes)
            }
        };

        // 添加响应头并发送。
        {
            let headers = evhttp_request_get_output_headers(req);
            let _ = evhttp_add_header(
                headers,
                CString::new("Content-Type").unwrap().as_ptr(),
                CString::new(content_type.as_str()).unwrap().as_ptr(),
            );
            let _ = evhttp_add_header(
                headers,
                CString::new("Access-Control-Allow-Origin")
                    .unwrap()
                    .as_ptr(),
                CString::new("*").unwrap().as_ptr(),
            );
            let _ = evhttp_add_header(
                headers,
                CString::new("Access-Control-Allow-Methods")
                    .unwrap()
                    .as_ptr(),
                CString::new("GET, POST, OPTIONS").unwrap().as_ptr(),
            );
            let _ = evhttp_add_header(
                headers,
                CString::new("Access-Control-Allow-Headers")
                    .unwrap()
                    .as_ptr(),
                CString::new("Content-Type").unwrap().as_ptr(),
            );
            let out_buf = evhttp_request_get_output_buffer(req);
            if !body_out.is_empty() {
                evbuffer_add(out_buf, body_out.as_ptr() as *const c_void, body_out.len());
            }

            let reason = status_reason(status);
            let c_reason = CString::new(reason).unwrap();
            evhttp_send_reply(req, status as c_int, c_reason.as_ptr(), out_buf);
        }
    }
}

unsafe fn read_request_body(req: *mut evhttp_request) -> Vec<u8> {
    unsafe {
        let buf = evhttp_request_get_input_buffer(req);
        if buf.is_null() {
            return Vec::new();
        }
        let len = evbuffer_get_length(buf);
        if len == 0 {
            return Vec::new();
        }
        //  defense-in-depth：即使 evhttp_set_max_body_size 已拒绝超大 body，
        //  回调中仍二次检查，防止未来 libevent 版本或配置差异导致大内存分配。
        if len > MAX_BODY_SIZE as usize {
            return Vec::new();
        }
        let data = evbuffer_pullup(buf, len as isize);
        if data.is_null() {
            return Vec::new();
        }
        std::slice::from_raw_parts(data, len).to_vec()
    }
}

/// 通过 getsockname 获取 libevent 绑定 socket 的真实地址。
unsafe fn get_bound_socket_addr(bound: *mut evhttp_bound_socket) -> std::io::Result<SocketAddr> {
    unsafe {
        let fd = evhttp_bound_socket_get_fd(bound);
        if fd < 0 {
            return Err(std::io::Error::other("evhttp_bound_socket_get_fd failed"));
        }
        let mut addr: sockaddr_in = std::mem::zeroed();
        let mut len: socklen_t = std::mem::size_of::<sockaddr_in>() as socklen_t;
        let ret = getsockname(fd, &mut addr as *mut _ as *mut sockaddr, &mut len);
        if ret != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let port = u16::from_be(addr.sin_port);
        let ip = Ipv4Addr::from(u32::from_be(addr.sin_addr.s_addr));
        Ok(SocketAddr::V4(SocketAddrV4::new(ip, port)))
    }
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    query: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    nprobe: usize,
    #[serde(default = "default_refine")]
    refine: bool,
    #[serde(default = "default_refine_k")]
    refine_k: usize,
    #[serde(default)]
    sql_filter: Option<String>,
}

fn default_k() -> usize {
    10
}

fn default_refine() -> bool {
    true
}

fn default_refine_k() -> usize {
    100
}

fn handle_search(index: &Arc<Mutex<IvfRabitqIndex>>, body: &str) -> (u16, String, Vec<u8>) {
    let req: SearchRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(400, &serde_json::json!({ "error": e.to_string() }));
        }
    };

    let k = req.k.min(MAX_K);
    if k == 0 {
        return json_response(400, &serde_json::json!({ "error": "k must be > 0" }));
    }

    let options = SearchOptions {
        k,
        nprobe: req.nprobe,
        refine: req.refine,
        refine_k: req.refine_k,
        fastscan: true,
        query_bits: 0,
        sq8_refine: false,
        sql_filter: req.sql_filter,
    };

    let idx = index.lock().unwrap();
    let results = search(&idx, &req.query, &options, None);
    let results_json: Vec<serde_json::Value> = results
        .into_iter()
        .map(|(id, dist)| serde_json::json!({"id": id, "distance": dist}))
        .collect();
    json_response(200, &serde_json::json!({ "results": results_json }))
}

#[derive(Debug, Deserialize)]
struct InsertRequest {
    vector: Vec<f32>,
    #[serde(default)]
    payload: Option<Payload>,
}

fn handle_insert(index: &Arc<Mutex<IvfRabitqIndex>>, body: &str) -> (u16, String, Vec<u8>) {
    let req: InsertRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(400, &serde_json::json!({ "error": e.to_string() }));
        }
    };

    let mut idx = index.lock().unwrap();
    let id = match req.payload {
        Some(p) => idx.add_with_payload(&req.vector, p),
        None => idx.add(&req.vector),
    };
    json_response(200, &serde_json::json!({ "id": id }))
}

#[derive(Debug, Deserialize)]
struct BatchInsertRequest {
    vectors: Vec<Vec<f32>>,
}

fn handle_batch_insert(index: &Arc<Mutex<IvfRabitqIndex>>, body: &str) -> (u16, String, Vec<u8>) {
    let req: BatchInsertRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => {
            return json_response(400, &serde_json::json!({ "error": e.to_string() }));
        }
    };

    let mut idx = index.lock().unwrap();
    let mut ids = Vec::with_capacity(req.vectors.len());
    for vector in req.vectors {
        ids.push(idx.add(&vector));
    }
    json_response(200, &serde_json::json!({ "ids": ids }))
}

#[derive(Debug, Serialize)]
struct StatsResponse {
    num_vectors: usize,
    num_partitions: usize,
    dim: usize,
}

fn handle_stats(index: &Arc<Mutex<IvfRabitqIndex>>) -> (u16, String, Vec<u8>) {
    let idx = index.lock().unwrap();
    let stats = StatsResponse {
        num_vectors: idx.len(),
        num_partitions: idx.num_partitions(),
        dim: idx.config().dim,
    };
    json_response(200, &serde_json::to_value(&stats).unwrap_or_default())
}

fn json_response(status: u16, value: &serde_json::Value) -> (u16, String, Vec<u8>) {
    let body = serde_json::to_vec(value).unwrap_or_default();
    (status, "application/json".to_string(), body)
}
