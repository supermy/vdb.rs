//! Tantivy C FFI 封装、倒排索引加载、段管理、全文过滤与向量搜索的交集执行。
//!
//! 当前为骨架阶段：接口已预留，真实 Tantivy FFI 集成待后续实现。
//! TDD 先行：本模块提供最小占位 API，确保 hybrid.rs 与测试可编译。
//!
//! untested: 全文检索为占位实现，尚未接入 Tantivy FFI，因此所有函数均通过桩返回，
//! 无实际可验证的索引/搜索行为。

use crate::index_ivf_rq::Payload;

/// 全文查询语法（占位）。
#[derive(Debug, Clone)]
pub struct FulltextQuery {
    pub text: String,
}

impl FulltextQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// 全文索引句柄（占位）。
#[derive(Debug, Clone)]
pub struct FulltextIndex;

impl FulltextIndex {
    pub fn new() -> Self {
        Self
    }

    /// 对 payload 集合建立倒排索引（占位实现，目前返回空句柄）。
    pub fn build(_payloads: &[Payload]) -> Self {
        Self
    }

    /// 执行全文搜索（占位实现，目前返回全部 id）。
    pub fn search(&self, _query: &FulltextQuery) -> Vec<u64> {
        Vec::new()
    }
}

impl Default for FulltextIndex {
    fn default() -> Self {
        Self::new()
    }
}
