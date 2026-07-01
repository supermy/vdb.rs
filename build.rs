//! 构建脚本：为 libevent 提供跨平台链接配置。
//!
//! 为什么需要：http_server.rs 通过 C FFI 调用 libevent evhttp，
//! 需要告知 cargo 链接 `event` 库；不同平台发现方式不同。

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("windows") {
        // Windows：尝试通过 vcpkg 发现 libevent。
        // 若未安装 vcpkg 或 libevent，编译会提示，保持显式失败优于隐式链接错误。
        if let Ok(lib) = vcpkg::find_package("libevent") {
            for path in lib.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
            println!("cargo:rustc-link-lib=event");
        } else {
            // 兜底：依赖环境已设置好库路径。
            println!("cargo:rustc-link-lib=event");
        }
    } else {
        // Unix/macOS：优先使用 pkg-config；失败则直接按系统默认路径链接。
        match pkg_config::probe_library("libevent") {
            Ok(_) => {}
            Err(_) => {
                println!("cargo:rustc-link-lib=event");
            }
        }
    }
}
