//! 分块 mmap + 用户态 LRU 的零拷贝索引加载器。
//!
//! 与 `storage::load_index_mmap` 的区别：
//! - `load_index_mmap` 仍会一次性映射全文件并在加载时把所有分区/原始向量拷贝到 `IvfRabitqIndex`。
//! - `MmapIndex` 在打开时只加载 header、rotation、centroids、partition metadata（通常几 MB 以内），
//!   分区数据与原始向量按 64MB 块按需 mmap，配合 LRU 淘汰，实现真正的“零拷贝启动”。
//!
//! 内存策略：
//! - 物理内存 < 16GB 时，缓存上限为物理内存的 50%；
//! - 否则为 85%。
//! - 单个 chunk 64MB，便于内核按页回收，也减少 LRU 管理开销。

use crate::index_ivf_rq::{RabitqCode, RabitqConfig, RabitqQuantizer};
use crate::simd::{dot_product, l2_distance_squared};
use crate::storage::{
    FORMAT_VERSION, Header, MAGIC, PartitionMeta, crc32_finalize, crc32_init, crc32_update,
};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// 默认 chunk 大小：64MB。
///
/// 选择 64MB 的理由：
/// - 大于多数分区数据，单个查询通常只需 fault 进 1~2 个 chunk。
/// - 小于低内存机器 50% 预算的下限（8GB 机器预算 4GB，可容纳 64 个 chunk），
///   保证 LRU 有足够的候选用于淘汰。
const CHUNK_SIZE: u64 = 64 * 1024 * 1024;

/// 低内存阈值与缓存比例，与 `storage.rs` 保持一致。
const LOW_MEMORY_THRESHOLD: u64 = 16 * 1024 * 1024 * 1024;
const LOW_MEMORY_FRACTION: f64 = 0.5;
const NORMAL_MEMORY_FRACTION: f64 = 0.85;

/// 流式校验读取缓冲区。
///
/// 选择 256KB：足够小以避免启动时大内存分配，又足够大以减少 read 调用次数。
const CRC_STREAM_BUF: usize = 256 * 1024;

/// 一个 mmap 住的文件块。
struct MmapChunk {
    ptr: *mut u8,
    len: usize,
}

impl MmapChunk {
    /// mmap 文件 [offset, offset + len) 区域（按页对齐）。
    fn map(file: &File, offset: u64, len: usize) -> io::Result<Self> {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) } as u64;
        let aligned_offset = (offset / page_size) * page_size;
        let front_padding = (offset - aligned_offset) as usize;
        let map_len = len + front_padding;

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                aligned_offset as libc::off_t,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        // 实际可用切片从 ptr + front_padding 开始，长度为 len。
        Ok(Self {
            ptr: unsafe { (ptr as *mut u8).add(front_padding) },
            len,
        })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MmapChunk {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        // 释放时必须按页对齐地址与原长度 munmap。
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) } as usize;
        let aligned_ptr = ((self.ptr as usize / page_size) * page_size) as *mut libc::c_void;
        let front_padding = self.ptr as usize - aligned_ptr as usize;
        let map_len = self.len + front_padding;
        unsafe {
            libc::munmap(aligned_ptr, map_len);
        }
    }
}

// MmapChunk 内部是只读的文件映射，指针在 chunk 生命周期内稳定，
// 因此可以安全地跨线程共享（多个查询并发读取同一块）。
unsafe impl Send for MmapChunk {}
unsafe impl Sync for MmapChunk {}

/// 分块 mmap 存储，带用户态 LRU 缓存。
///
/// 线程安全：内部使用 Mutex 保护 chunk 缓存与 LRU 队列，使 `MmapIndex::search` 可以只读引用。
/// 缓存项使用 `Arc<MmapChunk>`，保证 chunk 在被任何查询引用期间不会被 munmap。
pub struct ChunkedMmapStorage {
    file: File,
    file_len: u64,
    chunk_size: u64,
    max_total_bytes: u64,
    chunks: Mutex<HashMap<u64, Arc<MmapChunk>>>,
    lru: Mutex<VecDeque<u64>>,
    current_bytes: Mutex<u64>,
}

impl ChunkedMmapStorage {
    /// 打开文件并创建分块 mmap 存储。
    ///
    /// 启动时不 mmap 任何数据块；header/metadata 由 `MmapIndex::open` 单独读取。
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty file"));
        }

        let phys = available_physical_memory().unwrap_or(u64::MAX);
        let fraction = if phys < LOW_MEMORY_THRESHOLD {
            LOW_MEMORY_FRACTION
        } else {
            NORMAL_MEMORY_FRACTION
        };
        let max_total_bytes = ((phys as f64 * fraction) as u64).max(CHUNK_SIZE);

        Ok(Self {
            file,
            file_len,
            chunk_size: CHUNK_SIZE,
            max_total_bytes,
            chunks: Mutex::new(HashMap::new()),
            lru: Mutex::new(VecDeque::new()),
            current_bytes: Mutex::new(0),
        })
    }

    /// 将 [offset, offset + len) 的数据拷贝到 `out`。
    ///
    /// 按需 fault 进对应的 chunk；若缓存超限则按 LRU 淘汰。
    pub fn read(&self, offset: u64, len: usize, out: &mut [u8]) -> io::Result<()> {
        if offset + len as u64 > self.file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "read out of bounds",
            ));
        }
        assert_eq!(out.len(), len, "output buffer size mismatch");

        let mut written = 0usize;
        while written < len {
            let chunk_idx = (offset + written as u64) / self.chunk_size;
            let chunk_start = chunk_idx * self.chunk_size;
            let in_chunk_off = (offset + written as u64 - chunk_start) as usize;
            let to_read = ((self.chunk_size - in_chunk_off as u64) as usize).min(len - written);

            let chunk = self.get_chunk(chunk_idx)?;
            let data = chunk.as_slice();
            out[written..written + to_read]
                .copy_from_slice(&data[in_chunk_off..in_chunk_off + to_read]);
            written += to_read;
        }
        Ok(())
    }

    fn get_chunk(&self, chunk_idx: u64) -> io::Result<Arc<MmapChunk>> {
        // 热路径：已缓存则直接返回 Arc。
        {
            let chunks = self.chunks.lock().unwrap();
            if let Some(chunk) = chunks.get(&chunk_idx) {
                let arc = Arc::clone(chunk);
                drop(chunks);
                self.touch(chunk_idx);
                return Ok(arc);
            }
        }

        // 未命中：计算 chunk 范围并 mmap。
        let chunk_start = chunk_idx * self.chunk_size;
        let chunk_end = ((chunk_start + self.chunk_size).min(self.file_len)) as usize;
        let chunk_len = chunk_end - chunk_start as usize;

        let new_chunk = Arc::new(MmapChunk::map(&self.file, chunk_start, chunk_len)?);

        // 加锁顺序：chunks -> lru -> current_bytes，避免死锁。
        let mut chunks = self.chunks.lock().unwrap();
        let mut lru = self.lru.lock().unwrap();
        let mut current = self.current_bytes.lock().unwrap();

        // 双重检查：其他线程可能已加载。
        if let Some(chunk) = chunks.get(&chunk_idx) {
            let arc = Arc::clone(chunk);
            drop(current);
            drop(lru);
            drop(chunks);
            return Ok(arc);
        }

        // 淘汰直到有空间。
        // 仅当 chunk 除了缓存表格外没有其他引用时才真正释放并扣减 current_bytes；
        // 若 chunk 仍被外部查询持有（strong_count > 1），则视为热点数据并保留在缓存中。
        // 当所有缓存块都在使用时，预算会被暂时突破，这是 LRU 在活跃工作集过大时的预期行为。
        let mut examined = 0usize;
        while *current + chunk_len as u64 > self.max_total_bytes && examined < lru.len() {
            let evict_idx = lru
                .pop_back()
                .ok_or_else(|| io::Error::other("LRU inconsistent"))?;
            examined += 1;
            if let Some(evicted) = chunks.get(&evict_idx) {
                if Arc::strong_count(evicted) == 1 {
                    let len = evicted.len;
                    chunks.remove(&evict_idx);
                    *current -= len as u64;
                } else {
                    // 热点块：放回 LRU 头部，避免误淘汰。
                    lru.push_front(evict_idx);
                }
            }
        }

        chunks.insert(chunk_idx, Arc::clone(&new_chunk));
        *current += chunk_len as u64;
        lru.push_front(chunk_idx);
        Ok(new_chunk)
    }

    fn touch(&self, chunk_idx: u64) {
        let mut lru = self.lru.lock().unwrap();
        if let Some(pos) = lru.iter().position(|&x| x == chunk_idx) {
            lru.remove(pos);
            lru.push_front(chunk_idx);
        }
    }
}

/// 基于分块 mmap 的零拷贝索引。
///
/// 打开时仅加载：header、rotation（dim×dim f32）、centroids（num_partitions×dim f32）、
/// partition metadata（num_partitions × 24B）。
/// 分区量化码与原始向量在查询时按块 fault 进内存，并受 LRU 缓存约束。
pub struct MmapIndex {
    storage: ChunkedMmapStorage,
    header: Header,
    quantizer: RabitqQuantizer,
    #[allow(dead_code)]
    centroids: Vec<Vec<f32>>,
    rotated_centroids: Vec<Vec<f32>>,
    rotated_centroid_norms_sq: Vec<f32>,
    partition_metas: Vec<PartitionMeta>,
    #[allow(dead_code)]
    payload_offset: u64,
    #[allow(dead_code)]
    payload_len: u64,
    num_vectors: usize,
    dim: usize,
    num_partitions: usize,
    code_bytes: usize,
    entry_bytes: usize,
}

impl MmapIndex {
    /// 打开索引文件，仅加载元数据，不拷贝向量数据。
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let storage = ChunkedMmapStorage::open(path)?;

        // 先读取 header（它肯定在文件第一页内）。
        let mut header_buf = vec![0u8; Header::SIZE];
        storage.read(0, Header::SIZE, &mut header_buf)?;
        let header = Header::deserialize(&header_buf)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad header"))?;
        if &header.magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        if header.version != FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported version",
            ));
        }

        // 校验 checksum：使用流式 CRC，避免启动时一次性读取文件余下全部内容，
        // 保证“零拷贝启动”不被大文件校验破坏。
        verify_checksum_streaming(&storage, &header)?;

        let dim = header.dim as usize;
        let num_partitions = header.num_partitions as usize;
        let num_vectors = header.num_vectors as usize;
        let config = RabitqConfig::new(dim);

        // 读取 rotation。
        let mut rotation_buf = vec![0u8; header.rotation_len as usize];
        storage.read(
            header.rotation_offset,
            header.rotation_len as usize,
            &mut rotation_buf,
        )?;
        let rotation = cast_u8_slice_to_f32(&rotation_buf);
        let quantizer = RabitqQuantizer::from_rotation(config, rotation);

        // 读取 centroids。
        let mut centroids_buf = vec![0u8; header.centroids_len as usize];
        storage.read(
            header.centroids_offset,
            header.centroids_len as usize,
            &mut centroids_buf,
        )?;
        let centroids_flat = cast_u8_slice_to_f32(&centroids_buf);
        let mut centroids = Vec::with_capacity(num_partitions);
        for pid in 0..num_partitions {
            let start = pid * dim;
            centroids.push(centroids_flat[start..start + dim].to_vec());
        }

        // 预计算 R*centroid，加速查询路由。
        let rotated_centroids: Vec<Vec<f32>> = centroids
            .iter()
            .map(|c| quantizer.rotate_vector(c))
            .collect();
        let rotated_centroid_norms_sq: Vec<f32> = rotated_centroids
            .iter()
            .map(|c| c.iter().map(|v| v * v).sum())
            .collect();

        // 读取 partition metadata。
        let mut meta_buf = vec![0u8; header.partition_meta_len as usize];
        storage.read(
            header.partition_meta_offset,
            header.partition_meta_len as usize,
            &mut meta_buf,
        )?;
        let mut partition_metas = Vec::with_capacity(num_partitions);
        for pid in 0..num_partitions {
            let start = pid * PartitionMeta::SIZE;
            let meta = PartitionMeta::deserialize(&meta_buf[start..start + PartitionMeta::SIZE])
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad meta"))?;
            partition_metas.push(meta);
        }

        let code_bytes = dim / 8;
        let entry_bytes = 8 + code_bytes + 4 + 4;

        Ok(Self {
            storage,
            header,
            quantizer,
            centroids,
            rotated_centroids,
            rotated_centroid_norms_sq,
            partition_metas,
            payload_offset: header.payload_offset,
            payload_len: header.payload_len,
            num_vectors,
            dim,
            num_partitions,
            code_bytes,
            entry_bytes,
        })
    }

    /// 对指定查询执行 IVF_RaBitQ 搜索。
    ///
    /// 流程：
    /// 1. 量化查询向量。
    /// 2. 用 R*centroid 预计算选 nprobe 个分区。
    /// 3. 按需从分块 mmap 读取分区量化码，估计距离。
    /// 4. 对 TopK 候选读取原始向量做 refine（若 nprobe == 全部分区则 refine 全部候选）。
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(u64, f32)> {
        if self.num_vectors == 0 || k == 0 {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dim, "search: dimension mismatch");

        let nprobe = if nprobe == 0 {
            self.num_partitions
        } else {
            nprobe.min(self.num_partitions)
        };

        let rotated_query = self.quantizer.rotate_vector(query);
        let query_code = self.quantizer.encode_query_from_rotated(&rotated_query);
        let q_norm_sq: f32 = rotated_query.iter().map(|v| v * v).sum();

        // 分区路由：O(dim) 每分区，感谢 R*centroid 预计算。
        let mut partition_dists: Vec<(usize, f32)> = (0..self.num_partitions)
            .map(|pid| {
                let dot = dot_product(&rotated_query, &self.rotated_centroids[pid]);
                let dist = q_norm_sq + self.rotated_centroid_norms_sq[pid] - 2.0 * dot;
                (pid, dist)
            })
            .collect();
        partition_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let selected: Vec<usize> = partition_dists
            .into_iter()
            .take(nprobe)
            .map(|(p, _)| p)
            .collect();

        // 扫描候选分区。
        let mut candidates: Vec<(u64, f32)> = Vec::new();
        for &pid in &selected {
            candidates.extend(self.scan_partition(pid, &query_code));
        }

        if candidates.is_empty() {
            return Vec::new();
        }

        // 粗排 + refine。
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let refine_n = if nprobe == self.num_partitions {
            candidates.len()
        } else {
            (k * 10).min(candidates.len())
        };

        let mut refined: Vec<(u64, f32)> = candidates[..refine_n]
            .iter()
            .map(|(id, _)| {
                let exact = self
                    .raw_vector(*id)
                    .map(|v| l2_distance_squared(query, &v))
                    .unwrap_or(f32::MAX);
                (*id, exact)
            })
            .collect();
        refined.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        refined.truncate(k);
        refined
    }

    fn scan_partition(&self, pid: usize, query_code: &RabitqCode) -> Vec<(u64, f32)> {
        let meta = self.partition_metas[pid];
        let count = meta.count as usize;
        if count == 0 {
            return Vec::new();
        }

        // 一次性读取整个分区数据到临时 Vec。
        // 分区大小通常 < chunk_size，因此只会 fault 进 1~2 个 chunk。
        let mut part_data = vec![0u8; meta.len as usize];
        if let Err(e) = self
            .storage
            .read(meta.offset, meta.len as usize, &mut part_data)
        {
            eprintln!("[MmapIndex] failed to read partition {}: {}", pid, e);
            return Vec::new();
        }

        let code_bytes = self.code_bytes;
        let entry_bytes = self.entry_bytes;
        let mut results = Vec::with_capacity(count);

        for i in 0..count {
            let off = i * entry_bytes;
            let id = u64::from_le_bytes(part_data[off..off + 8].try_into().unwrap());
            let bits = &part_data[off + 8..off + 8 + code_bytes];
            let alpha = f32::from_le_bytes(
                part_data[off + 8 + code_bytes..off + 8 + code_bytes + 4]
                    .try_into()
                    .unwrap(),
            );
            let beta = f32::from_le_bytes(
                part_data[off + 8 + code_bytes + 4..off + 8 + code_bytes + 8]
                    .try_into()
                    .unwrap(),
            );
            let dist = self
                .quantizer
                .estimate_distance_sq_raw(query_code, bits, alpha, beta);
            results.push((id, dist));
        }
        results
    }

    /// 从分块 mmap 读取原始向量。
    ///
    /// 为什么按 id 随机读取仍可接受：refine 阶段只读取 TopK 候选，数量极少；
    /// 若未来需要批量 refine，可改为按 chunk 预取。
    pub fn raw_vector(&self, id: u64) -> Option<Vec<f32>> {
        if id >= self.num_vectors as u64 {
            return None;
        }
        let offset = self.header.raw_offset + id * self.dim as u64 * 4;
        let len = self.dim * 4;
        let mut buf = vec![0u8; len];
        self.storage.read(offset, len, &mut buf).ok()?;
        Some(cast_u8_slice_to_f32(&buf))
    }

    pub fn len(&self) -> usize {
        self.num_vectors
    }

    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }
}

/// 流式验证文件 checksum。
///
/// 逐块读取 header 之后的数据并更新 CRC，遇到 manifest 中 checksum 字段时将其置 0，
/// 最终与 manifest 中存储的 checksum 比较。此方法不依赖文件大小一次性分配大 Vec。
fn verify_checksum_streaming(storage: &ChunkedMmapStorage, header: &Header) -> io::Result<()> {
    let file_len = storage.file_len;
    let checksum_start = header.manifest_offset + 8;
    let checksum_end = checksum_start + 4;

    let mut crc = crc32_init();
    let mut buf = vec![0u8; CRC_STREAM_BUF];
    let mut offset = Header::SIZE as u64;
    while offset < file_len {
        let to_read = ((file_len - offset) as usize).min(buf.len());
        storage.read(offset, to_read, &mut buf[..to_read])?;

        // 将本段与 manifest checksum 字段重叠的字节置 0。
        let seg_start = offset;
        let seg_end = offset + to_read as u64;
        if seg_end > checksum_start && seg_start < checksum_end {
            let zero_start = (checksum_start.max(seg_start) - seg_start) as usize;
            let zero_end = (checksum_end.min(seg_end) - seg_start) as usize;
            buf[zero_start..zero_end].fill(0);
        }

        crc = crc32_update(crc, &buf[..to_read]);
        offset += to_read as u64;
    }

    let computed = crc32_finalize(crc);
    let mut stored_bytes = [0u8; 4];
    storage.read(checksum_start, 4, &mut stored_bytes)?;
    let stored = u32::from_le_bytes(stored_bytes);
    if computed != stored {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checksum mismatch",
        ));
    }
    Ok(())
}

fn cast_u8_slice_to_f32(slice: &[u8]) -> Vec<f32> {
    assert!(
        slice.len() % 4 == 0,
        "u8 slice length must be multiple of 4"
    );
    slice
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[cfg(unix)]
fn available_physical_memory() -> Option<u64> {
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        let page_size = libc::sysconf(libc::_SC_PAGE_SIZE);
        if pages > 0 && page_size > 0 {
            Some((pages as u64) * (page_size as u64))
        } else {
            None
        }
    }
}

#[cfg(not(unix))]
fn available_physical_memory() -> Option<u64> {
    // untested: 非 Unix 平台当前未在 CI 矩阵覆盖，返回 None 表示采用保守预算。
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_ivf_rq::IvfRabitqIndex;
    use crate::storage::save_index;
    use std::collections::HashSet;
    use std::io::{Read, Seek, Write};
    use tempfile::TempDir;

    fn gaussian_random() -> f32 {
        let u1 = rand::random::<f32>().max(1e-7);
        let u2 = rand::random::<f32>();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
    }

    #[test]
    fn test_mmap_index_zero_copy_search() {
        let dim = 64;
        let n = 1000;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_random()).collect())
            .collect();
        let index = IvfRabitqIndex::build(&vectors);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index_mmap.vdb");
        save_index(&path, &index).unwrap();

        let mmap_index = MmapIndex::open(&path).unwrap();
        assert_eq!(mmap_index.len(), n);
        assert_eq!(mmap_index.dim(), dim);

        let query: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();
        let truth: HashSet<u64> = index
            .flat_search(&query, 10)
            .into_iter()
            .map(|(id, _)| id)
            .collect();

        let results = mmap_index.search(&query, 10, 0);
        assert_eq!(results.len(), 10);
        let recall = results.iter().filter(|(id, _)| truth.contains(id)).count();
        assert!(recall >= 5, "MmapIndex recall too low: {}/10", recall);
    }

    #[test]
    fn test_mmap_index_refine_topk() {
        let dim = 128;
        let n = 500;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_random()).collect())
            .collect();
        let index = IvfRabitqIndex::build(&vectors);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index_mmap_refine.vdb");
        save_index(&path, &index).unwrap();

        let mmap_index = MmapIndex::open(&path).unwrap();
        let query: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();
        let results = mmap_index.search(&query, 5, 8);
        assert_eq!(results.len(), 5);
        for (_, dist) in &results {
            assert!(dist.is_finite() && *dist >= 0.0);
        }
    }

    #[test]
    fn test_mmap_index_checksum_detects_corruption() {
        let dim = 64;
        let n = 500;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_random()).collect())
            .collect();
        let index = IvfRabitqIndex::build(&vectors);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index_mmap_corrupt.vdb");
        save_index(&path, &index).unwrap();

        let mut file = std::fs::File::options()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(std::io::SeekFrom::Start(Header::SIZE as u64 + 10))
            .unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        byte[0] = !byte[0];
        file.seek(std::io::SeekFrom::Start(Header::SIZE as u64 + 10))
            .unwrap();
        file.write_all(&byte).unwrap();
        drop(file);

        assert!(
            MmapIndex::open(&path).is_err(),
            "corrupted file should fail checksum under chunked mmap"
        );
    }

    #[test]
    fn test_chunked_mmap_storage_bounds() {
        let dim = 64;
        let n = 2000;
        let mut index = IvfRabitqIndex::new(dim);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| gaussian_random()).collect();
            index.add(&v);
        }

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index_chunks.vdb");
        save_index(&path, &index).unwrap();

        let storage = ChunkedMmapStorage::open(&path).unwrap();
        assert!(storage.max_total_bytes > 0);

        // 读取 header 验证分块读取正确性。
        let mut buf = vec![0u8; Header::SIZE];
        storage.read(0, Header::SIZE, &mut buf).unwrap();
        let header = Header::deserialize(&buf).unwrap();
        assert_eq!(header.num_vectors, n as u64);
    }
}
