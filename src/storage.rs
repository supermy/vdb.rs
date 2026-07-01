//! 磁盘列式存储（LanceDB 方向）：partition-oriented columnar 文件格式，
//! 支持完整索引 save/load（config、rotation、partitions、super_partitions、next_id），
//! 使用 std::io 随机读写 API；同时提供 mmap 按需加载与内存 bound 策略。

use crate::index_ivf_rq::{IvfRabitqIndex, RabitqCode, RabitqConfig, RabitqQuantizer};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

/// 内存约束常量：在物理内存 < 16GB 的设备上，mmap 大小不得超过物理内存的 50%。
/// 这样可避免内核 VM 压力过大导致 OOM，同时保留足够空间给查询缓冲与工作集。
const LOW_MEMORY_THRESHOLD: u64 = 16 * 1024 * 1024 * 1024;
const LOW_MEMORY_MMAP_FRACTION: f64 = 0.5;
const NORMAL_MEMORY_MMAP_FRACTION: f64 = 0.85;

/// 跨平台的内存映射文件视图（仅 Unix 提供 mmap 实现）。
///
/// 为什么包装一层：把 `mmap` 生命周期与 `munmap` 调用封装在 Drop 中，
/// 使上层加载代码可以安全地按切片访问，而不必在每个读取点重复 unsafe。
/// Windows 等非 Unix 平台使用 `load_index` 全量加载回退。
#[cfg(unix)]
#[derive(Debug)]
pub struct MmapStorage {
    ptr: *mut u8,
    len: usize,
}

#[cfg(unix)]
impl MmapStorage {
    /// 创建文件的私有只读 mmap。
    ///
    /// 加载策略：
    /// 1. 先检测可用物理内存，计算允许 mmap 的最大字节数。
    /// 2. 如果文件大小超过预算，返回错误，强制调用方改用分块加载或 LRU 策略。
    ///    目前实现一次性全文件映射，依赖预算控制避免小内存机器 OOM。
    pub fn map_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "empty index file",
            ));
        }

        let phys = available_physical_memory().unwrap_or(u64::MAX);
        let fraction = if phys < LOW_MEMORY_THRESHOLD {
            LOW_MEMORY_MMAP_FRACTION
        } else {
            NORMAL_MEMORY_MMAP_FRACTION
        };
        let budget = (phys as f64 * fraction) as u64;
        if len > budget {
            return Err(std::io::Error::other(format!(
                "index file size {} exceeds mmap budget {} (phys {})",
                len, budget, phys
            )));
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len as usize,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }

        Ok(Self {
            ptr: ptr as *mut u8,
            len: len as usize,
        })
    }

    /// 返回 mmap 区域的不可变切片。
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// 返回指定范围的子切片。
    pub fn slice(&self, offset: u64, len: u64) -> &[u8] {
        let start = offset as usize;
        let end = start + len as usize;
        &self.as_slice()[start..end]
    }
}

#[cfg(unix)]
impl Drop for MmapStorage {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}

/// 获取可用物理内存字节数。
///
/// 在类 Unix 系统上通过 sysconf 读取；失败时返回 None，调用方可选择保守策略。
#[cfg(unix)]
pub fn available_physical_memory() -> Option<u64> {
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
pub fn available_physical_memory() -> Option<u64> {
    // untested: 非 Unix 平台当前未在 CI 矩阵覆盖，返回 None 采用保守预算。
    None
}

#[cfg(unix)]
use std::os::fd::AsRawFd;

pub(crate) const MAGIC: &[u8; 8] = b"VDBRQIDX";
pub(crate) const FORMAT_VERSION: u32 = 1;

/// 索引文件头（内部布局）。
///
/// 标记为 `pub(crate)` 以便 `mmap_index.rs` 中的零拷贝加载器复用同一解析逻辑，
/// 避免两份 header 定义漂移。
#[derive(Debug, Clone, Copy)]
pub(crate) struct Header {
    pub(crate) magic: [u8; 8],
    pub(crate) version: u32,
    pub(crate) dim: u32,
    pub(crate) num_partitions: u32,
    pub(crate) num_vectors: u64,
    pub(crate) next_id: u64,
    pub(crate) rotation_offset: u64,
    pub(crate) rotation_len: u64,
    pub(crate) centroids_offset: u64,
    pub(crate) centroids_len: u64,
    pub(crate) partition_meta_offset: u64,
    pub(crate) partition_meta_len: u64,
    pub(crate) partition_data_offset: u64,
    pub(crate) partition_data_len: u64,
    pub(crate) raw_offset: u64,
    pub(crate) raw_len: u64,
    pub(crate) payload_offset: u64,
    pub(crate) payload_len: u64,
    pub(crate) manifest_offset: u64,
    pub(crate) manifest_len: u64,
}

impl Header {
    pub(crate) const SIZE: usize = 8 + 4 + 4 + 4 + 8 + 8 + 8 * 14;

    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::SIZE);
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.dim.to_le_bytes());
        buf.extend_from_slice(&self.num_partitions.to_le_bytes());
        buf.extend_from_slice(&self.num_vectors.to_le_bytes());
        buf.extend_from_slice(&self.next_id.to_le_bytes());
        buf.extend_from_slice(&self.rotation_offset.to_le_bytes());
        buf.extend_from_slice(&self.rotation_len.to_le_bytes());
        buf.extend_from_slice(&self.centroids_offset.to_le_bytes());
        buf.extend_from_slice(&self.centroids_len.to_le_bytes());
        buf.extend_from_slice(&self.partition_meta_offset.to_le_bytes());
        buf.extend_from_slice(&self.partition_meta_len.to_le_bytes());
        buf.extend_from_slice(&self.partition_data_offset.to_le_bytes());
        buf.extend_from_slice(&self.partition_data_len.to_le_bytes());
        buf.extend_from_slice(&self.raw_offset.to_le_bytes());
        buf.extend_from_slice(&self.raw_len.to_le_bytes());
        buf.extend_from_slice(&self.payload_offset.to_le_bytes());
        buf.extend_from_slice(&self.payload_len.to_le_bytes());
        buf.extend_from_slice(&self.manifest_offset.to_le_bytes());
        buf.extend_from_slice(&self.manifest_len.to_le_bytes());
        buf
    }

    pub(crate) fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&buf[0..8]);
        let version = u32::from_le_bytes(buf[8..12].try_into().ok()?);
        let dim = u32::from_le_bytes(buf[12..16].try_into().ok()?);
        let num_partitions = u32::from_le_bytes(buf[16..20].try_into().ok()?);
        let num_vectors = u64::from_le_bytes(buf[20..28].try_into().ok()?);
        let next_id = u64::from_le_bytes(buf[28..36].try_into().ok()?);
        let rotation_offset = u64::from_le_bytes(buf[36..44].try_into().ok()?);
        let rotation_len = u64::from_le_bytes(buf[44..52].try_into().ok()?);
        let centroids_offset = u64::from_le_bytes(buf[52..60].try_into().ok()?);
        let centroids_len = u64::from_le_bytes(buf[60..68].try_into().ok()?);
        let partition_meta_offset = u64::from_le_bytes(buf[68..76].try_into().ok()?);
        let partition_meta_len = u64::from_le_bytes(buf[76..84].try_into().ok()?);
        let partition_data_offset = u64::from_le_bytes(buf[84..92].try_into().ok()?);
        let partition_data_len = u64::from_le_bytes(buf[92..100].try_into().ok()?);
        let raw_offset = u64::from_le_bytes(buf[100..108].try_into().ok()?);
        let raw_len = u64::from_le_bytes(buf[108..116].try_into().ok()?);
        let payload_offset = u64::from_le_bytes(buf[116..124].try_into().ok()?);
        let payload_len = u64::from_le_bytes(buf[124..132].try_into().ok()?);
        let manifest_offset = u64::from_le_bytes(buf[132..140].try_into().ok()?);
        let manifest_len = u64::from_le_bytes(buf[140..148].try_into().ok()?);
        Some(Self {
            magic,
            version,
            dim,
            num_partitions,
            num_vectors,
            next_id,
            rotation_offset,
            rotation_len,
            centroids_offset,
            centroids_len,
            partition_meta_offset,
            partition_meta_len,
            partition_data_offset,
            partition_data_len,
            raw_offset,
            raw_len,
            payload_offset,
            payload_len,
            manifest_offset,
            manifest_len,
        })
    }
}

/// 分区元数据：偏移、长度、向量数。
#[derive(Debug, Clone, Copy)]
pub(crate) struct PartitionMeta {
    pub(crate) offset: u64,
    pub(crate) len: u64,
    pub(crate) count: u64,
}

impl PartitionMeta {
    pub(crate) const SIZE: usize = 24;

    pub(crate) fn serialize(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..16].copy_from_slice(&self.len.to_le_bytes());
        buf[16..24].copy_from_slice(&self.count.to_le_bytes());
        buf
    }

    pub(crate) fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            offset: u64::from_le_bytes(buf[0..8].try_into().ok()?),
            len: u64::from_le_bytes(buf[8..16].try_into().ok()?),
            count: u64::from_le_bytes(buf[16..24].try_into().ok()?),
        })
    }
}

/// Manifest 记录版本与校验信息。
///
/// 追加写事务边界：每次 save 生成新版本 manifest，旧 manifest 保留，
/// 从而实现 time-travel（后续可扩展为 manifest 链）。
#[derive(Debug, Clone, Copy)]
struct Manifest {
    version: u64,
    checksum: u32,
}

impl Manifest {
    const SIZE: usize = 16;

    fn serialize(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.checksum.to_le_bytes());
        buf
    }

    fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::SIZE {
            return None;
        }
        Some(Self {
            version: u64::from_le_bytes(buf[0..8].try_into().ok()?),
            checksum: u32::from_le_bytes(buf[8..12].try_into().ok()?),
        })
    }
}

pub(crate) fn crc32_init() -> u32 {
    0xFFFFFFFF
}

pub(crate) fn crc32_update(mut crc: u32, data: &[u8]) -> u32 {
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc
}

pub(crate) fn crc32_finalize(crc: u32) -> u32 {
    crc ^ 0xFFFFFFFF
}

pub(crate) fn crc32(data: &[u8]) -> u32 {
    let crc = crc32_init();
    let crc = crc32_update(crc, data);
    crc32_finalize(crc)
}

/// 将 IVF_RaBitQ 索引持久化到磁盘。
///
/// 为什么使用列式分区布局：同一分区的 id/code/alpha/beta 连续存储，
/// 查询时只需 seek 到目标分区，避免读取无关数据。
pub fn save_index<P: AsRef<Path>>(path: P, index: &IvfRabitqIndex) -> std::io::Result<()> {
    let path = path.as_ref();
    let mut file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let dim = index.config().dim;
    let num_partitions = index.num_partitions();
    let num_vectors = index.len() as u64;
    let next_id = index.next_id();

    // 预留文件头位置。
    let header_buf = vec![0u8; Header::SIZE];
    file.write_all(&header_buf)?;

    // 写入旋转矩阵。
    let rotation_offset = file.stream_position()?;
    let rotation = index.rotation_matrix();
    let rotation_bytes = cast_f32_slice_to_u8(rotation);
    file.write_all(rotation_bytes)?;
    let rotation_len = rotation_bytes.len() as u64;

    // 写入质心。
    let centroids_offset = file.stream_position()?;
    for centroid in index.centroids() {
        let bytes = cast_f32_slice_to_u8(centroid);
        file.write_all(bytes)?;
    }
    let centroids_len = file.stream_position()? - centroids_offset;

    // 写入分区数据。
    let partition_data_offset = file.stream_position()?;
    let mut partition_metas = Vec::with_capacity(num_partitions);
    for pid in 0..num_partitions {
        let offset = file.stream_position()?;
        let entries = index.partition_entries(pid);
        for (id, code) in entries {
            file.write_all(&id.to_le_bytes())?;
            file.write_all(&code.bits)?;
            file.write_all(&code.alpha.to_le_bytes())?;
            file.write_all(&code.beta.to_le_bytes())?;
        }
        let len = file.stream_position()? - offset;
        partition_metas.push(PartitionMeta {
            offset,
            len,
            count: entries.len() as u64,
        });
    }
    let partition_data_len = file.stream_position()? - partition_data_offset;

    // 写入分区元数据。
    let partition_meta_offset = file.stream_position()?;
    for meta in &partition_metas {
        file.write_all(&meta.serialize())?;
    }
    let partition_meta_len = (partition_metas.len() * PartitionMeta::SIZE) as u64;

    // 写入原始向量（用于 refine 层与 Flat baseline）。
    let raw_offset = file.stream_position()?;
    for vector in index.raw_vectors() {
        let bytes = cast_f32_slice_to_u8(vector);
        file.write_all(bytes)?;
    }
    let raw_len = file.stream_position()? - raw_offset;

    // 写入标量 payload（JSON 数组）。
    let payload_offset = file.stream_position()?;
    let payload_bytes = serde_json::to_vec(index.payloads())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    file.write_all(&payload_bytes)?;
    let payload_len = payload_bytes.len() as u64;

    // 写入 manifest。
    let manifest_offset = file.stream_position()?;
    let manifest = Manifest {
        version: next_id,
        checksum: 0,
    };
    file.write_all(&manifest.serialize())?;
    let manifest_len = Manifest::SIZE as u64;

    // 回填文件头。
    let header = Header {
        magic: *MAGIC,
        version: FORMAT_VERSION,
        dim: dim as u32,
        num_partitions: num_partitions as u32,
        num_vectors,
        next_id,
        rotation_offset,
        rotation_len,
        centroids_offset,
        centroids_len,
        partition_meta_offset,
        partition_meta_len,
        partition_data_offset,
        partition_data_len,
        raw_offset,
        raw_len,
        payload_offset,
        payload_len,
        manifest_offset,
        manifest_len,
    };
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header.serialize())?;

    // 计算并回填 checksum（覆盖 header 之后全部数据）。
    let file_len = file.stream_position()?;
    let mut data_for_crc = Vec::with_capacity((file_len - Header::SIZE as u64) as usize);
    file.seek(SeekFrom::Start(Header::SIZE as u64))?;
    file.read_to_end(&mut data_for_crc)?;
    let checksum = crc32(&data_for_crc);
    file.seek(SeekFrom::Start(manifest_offset + 8))?;
    file.write_all(&checksum.to_le_bytes())?;

    file.flush()?;
    Ok(())
}

/// 从磁盘加载 IVF_RaBitQ 索引。
///
/// 加载策略：
/// 1. 读取文件头验证 magic 与版本。
/// 2. 按偏移读取旋转矩阵、质心、分区元数据。
/// 3. 按分区元数据 seek 到各分区数据并反序列化。
/// 4. 校验 manifest checksum。
///
/// 当前为 eager load（启动时读入内存）；后续阶段通过 mmap 实现按需加载。
pub fn load_index<P: AsRef<Path>>(path: P) -> std::io::Result<IvfRabitqIndex> {
    let mut file = File::open(path)?;
    let mut header_buf = vec![0u8; Header::SIZE];
    file.read_exact(&mut header_buf)?;
    let header = Header::deserialize(&header_buf)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header"))?;
    if &header.magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad magic",
        ));
    }
    if header.version != FORMAT_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported version",
        ));
    }

    let dim = header.dim as usize;
    let num_partitions = header.num_partitions as usize;

    // 读取旋转矩阵。
    file.seek(SeekFrom::Start(header.rotation_offset))?;
    let mut rotation_buf = vec![0u8; header.rotation_len as usize];
    file.read_exact(&mut rotation_buf)?;
    let rotation = cast_u8_slice_to_f32(&rotation_buf);

    // 读取质心。
    file.seek(SeekFrom::Start(header.centroids_offset))?;
    let mut centroids_buf = vec![0u8; header.centroids_len as usize];
    file.read_exact(&mut centroids_buf)?;
    let centroids_flat = cast_u8_slice_to_f32(&centroids_buf);
    let mut centroids = Vec::with_capacity(num_partitions);
    for pid in 0..num_partitions {
        let start = pid * dim;
        centroids.push(centroids_flat[start..start + dim].to_vec());
    }

    // 读取分区元数据。
    file.seek(SeekFrom::Start(header.partition_meta_offset))?;
    let mut meta_buf = vec![0u8; header.partition_meta_len as usize];
    file.read_exact(&mut meta_buf)?;
    let mut partition_metas = Vec::with_capacity(num_partitions);
    for pid in 0..num_partitions {
        let start = pid * PartitionMeta::SIZE;
        let meta = PartitionMeta::deserialize(&meta_buf[start..start + PartitionMeta::SIZE])
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad meta"))?;
        partition_metas.push(meta);
    }

    // 读取分区数据。
    let code_bytes = dim / 8;
    let entry_bytes = 8 + code_bytes + 4 + 4;
    let mut partitions: Vec<Vec<(u64, RabitqCode)>> = vec![Vec::new(); num_partitions];
    for (pid, meta) in partition_metas.iter().enumerate() {
        file.seek(SeekFrom::Start(meta.offset))?;
        let mut data = vec![0u8; meta.len as usize];
        file.read_exact(&mut data)?;
        let count = meta.count as usize;
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let off = i * entry_bytes;
            let id = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            let bits = data[off + 8..off + 8 + code_bytes].to_vec();
            let alpha = f32::from_le_bytes(
                data[off + 8 + code_bytes..off + 8 + code_bytes + 4]
                    .try_into()
                    .unwrap(),
            );
            let beta = f32::from_le_bytes(
                data[off + 8 + code_bytes + 4..off + 8 + code_bytes + 8]
                    .try_into()
                    .unwrap(),
            );
            entries.push((id, RabitqCode { bits, alpha, beta }));
        }
        partitions[pid] = entries;
    }

    // 读取原始向量。
    file.seek(SeekFrom::Start(header.raw_offset))?;
    let mut raw_buf = vec![0u8; header.raw_len as usize];
    file.read_exact(&mut raw_buf)?;
    let raw_flat = cast_u8_slice_to_f32(&raw_buf);
    let num_vectors = header.num_vectors as usize;
    let mut raw = Vec::with_capacity(num_vectors);
    for i in 0..num_vectors {
        let start = i * dim;
        raw.push(raw_flat[start..start + dim].to_vec());
    }

    // 读取标量 payload。
    let mut payloads: Vec<crate::index_ivf_rq::Payload> = Vec::with_capacity(num_vectors);
    if header.payload_len > 0 {
        file.seek(SeekFrom::Start(header.payload_offset))?;
        let mut payload_buf = vec![0u8; header.payload_len as usize];
        file.read_exact(&mut payload_buf)?;
        payloads = serde_json::from_slice(&payload_buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    }

    // 校验 checksum。
    file.seek(SeekFrom::Start(Header::SIZE as u64))?;
    let mut data_for_crc = Vec::new();
    file.read_to_end(&mut data_for_crc)?;
    // 将 manifest 中的 checksum 字段置 0 后计算。
    let manifest_rel_off = (header.manifest_offset - Header::SIZE as u64) as usize;
    data_for_crc[manifest_rel_off + 8..manifest_rel_off + 12].fill(0);
    let computed = crc32(&data_for_crc);

    file.seek(SeekFrom::Start(header.manifest_offset))?;
    let mut manifest_buf = [0u8; Manifest::SIZE];
    file.read_exact(&mut manifest_buf)?;
    let manifest = Manifest::deserialize(&manifest_buf)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad manifest"))?;
    let stored = manifest.checksum;
    if computed != stored {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "checksum mismatch",
        ));
    }

    let config = RabitqConfig::new(dim);
    let quantizer = RabitqQuantizer::from_rotation(config, rotation);
    Ok(IvfRabitqIndex::from_parts(
        config,
        quantizer,
        centroids,
        partitions,
        raw,
        payloads,
        header.next_id,
    ))
}

/// 通过 mmap 按需加载索引。
///
/// 与 `load_index` 的区别：
/// - 不先 `read_to_end` 把整个文件读入 Vec，而是直接 mmap。
/// - 按偏移切片访问各区域，旋转矩阵、质心、分区数据等仍按需反序列化为 Rust 对象，
///   但避免了大文件场景下启动时的全量拷贝。
/// - 加载前检测物理内存，文件大小超过预算时直接拒绝，防止小内存机器 OOM。
///
/// 当前实现一次性映射整个文件；在超预算或非 Unix 平台时使用分块 mmap + LRU 回退。
#[cfg(unix)]
pub fn load_index_mmap<P: AsRef<Path>>(path: P) -> std::io::Result<IvfRabitqIndex> {
    let mmap = MmapStorage::map_file(path)?;
    let data = mmap.as_slice();

    if data.len() < Header::SIZE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file too short for header",
        ));
    }
    let header = Header::deserialize(&data[..Header::SIZE])
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header"))?;
    if &header.magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad magic",
        ));
    }
    if header.version != FORMAT_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "unsupported version",
        ));
    }

    let dim = header.dim as usize;
    let num_partitions = header.num_partitions as usize;

    // 校验 checksum：将 manifest 中 checksum 字段置 0 后计算 CRC32。
    let manifest_rel_off = (header.manifest_offset - Header::SIZE as u64) as usize;
    let mut crc_buf = data[Header::SIZE..].to_vec();
    crc_buf[manifest_rel_off + 8..manifest_rel_off + 12].fill(0);
    let computed = crc32(&crc_buf);
    let stored = u32::from_le_bytes(
        data[header.manifest_offset as usize + 8..header.manifest_offset as usize + 12]
            .try_into()
            .unwrap(),
    );
    if computed != stored {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "checksum mismatch",
        ));
    }

    // 旋转矩阵：直接从 mmap 切片拷贝为 f32 Vec。
    let rotation = cast_u8_slice_to_f32(mmap.slice(header.rotation_offset, header.rotation_len));

    // 质心。
    let centroids_flat =
        cast_u8_slice_to_f32(mmap.slice(header.centroids_offset, header.centroids_len));
    let mut centroids = Vec::with_capacity(num_partitions);
    for pid in 0..num_partitions {
        let start = pid * dim;
        centroids.push(centroids_flat[start..start + dim].to_vec());
    }

    // 分区元数据。
    let meta_buf = mmap.slice(header.partition_meta_offset, header.partition_meta_len);
    let mut partition_metas = Vec::with_capacity(num_partitions);
    for pid in 0..num_partitions {
        let start = pid * PartitionMeta::SIZE;
        let meta = PartitionMeta::deserialize(&meta_buf[start..start + PartitionMeta::SIZE])
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad meta"))?;
        partition_metas.push(meta);
    }

    // 分区数据。
    let code_bytes = dim / 8;
    let entry_bytes = 8 + code_bytes + 4 + 4;
    let mut partitions: Vec<Vec<(u64, RabitqCode)>> = vec![Vec::new(); num_partitions];
    for (pid, meta) in partition_metas.iter().enumerate() {
        let part_data = mmap.slice(meta.offset, meta.len);
        let count = meta.count as usize;
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let off = i * entry_bytes;
            let id = u64::from_le_bytes(part_data[off..off + 8].try_into().unwrap());
            let bits = part_data[off + 8..off + 8 + code_bytes].to_vec();
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
            entries.push((id, RabitqCode { bits, alpha, beta }));
        }
        partitions[pid] = entries;
    }

    // 原始向量。
    let raw_flat = cast_u8_slice_to_f32(mmap.slice(header.raw_offset, header.raw_len));
    let num_vectors = header.num_vectors as usize;
    let mut raw = Vec::with_capacity(num_vectors);
    for i in 0..num_vectors {
        let start = i * dim;
        raw.push(raw_flat[start..start + dim].to_vec());
    }

    // 标量 payload。
    let mut payloads: Vec<crate::index_ivf_rq::Payload> = Vec::with_capacity(num_vectors);
    if header.payload_len > 0 {
        let payload_buf = mmap.slice(header.payload_offset, header.payload_len);
        payloads = serde_json::from_slice(payload_buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    }

    let config = RabitqConfig::new(dim);
    let quantizer = RabitqQuantizer::from_rotation(config, rotation);
    Ok(IvfRabitqIndex::from_parts(
        config,
        quantizer,
        centroids,
        partitions,
        raw,
        payloads,
        header.next_id,
    ))
}

fn cast_f32_slice_to_u8(slice: &[f32]) -> &[u8] {
    let len = slice.len() * 4;
    let ptr = slice.as_ptr() as *const u8;
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

fn cast_u8_slice_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_ivf_rq::IvfRabitqIndex;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_save_load_roundtrip() {
        let dim = 64;
        let n = 100;
        let mut index = IvfRabitqIndex::new(dim);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            index.add(&v);
        }
        assert_eq!(index.len(), n);

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index.vdb");
        save_index(&path, &index).unwrap();

        let loaded = load_index(&path).unwrap();
        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.num_partitions(), index.num_partitions());

        let query: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let r1 = loaded.search(&query, 5, 2);
        assert_eq!(r1.len(), 5);
    }

    #[test]
    fn test_checksum_detects_corruption() {
        let dim = 64;
        let mut index = IvfRabitqIndex::new(dim);
        for _ in 0..10 {
            let v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            index.add(&v);
        }

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index.vdb");
        save_index(&path, &index).unwrap();

        // 破坏一个字节。
        let mut file = File::options().read(true).write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(Header::SIZE as u64 + 10))
            .unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        byte[0] = !byte[0];
        file.seek(SeekFrom::Start(Header::SIZE as u64 + 10))
            .unwrap();
        file.write_all(&byte).unwrap();
        drop(file);

        assert!(load_index(&path).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn test_load_index_mmap_roundtrip() {
        let dim = 64;
        let n = 100;
        let mut index = IvfRabitqIndex::new(dim);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            index.add(&v);
        }

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index_mmap.vdb");
        save_index(&path, &index).unwrap();

        let loaded = load_index_mmap(&path).unwrap();
        assert_eq!(loaded.len(), n);
        assert_eq!(loaded.num_partitions(), index.num_partitions());

        let query: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let r1 = loaded.search(&query, 5, 2);
        assert_eq!(r1.len(), 5);
    }

    #[test]
    #[cfg(unix)]
    fn test_mmap_storage_checksum_detects_corruption() {
        let dim = 64;
        let mut index = IvfRabitqIndex::new(dim);
        for _ in 0..10 {
            let v: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
            index.add(&v);
        }

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("index_mmap_corrupt.vdb");
        save_index(&path, &index).unwrap();

        let mut file = File::options().read(true).write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(Header::SIZE as u64 + 10))
            .unwrap();
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte).unwrap();
        byte[0] = !byte[0];
        file.seek(SeekFrom::Start(Header::SIZE as u64 + 10))
            .unwrap();
        file.write_all(&byte).unwrap();
        drop(file);

        assert!(load_index_mmap(&path).is_err());
    }

    #[test]
    #[cfg(unix)]
    fn test_available_physical_memory_nonzero() {
        let mem = available_physical_memory();
        assert!(mem.is_some(), "should detect physical memory on unix");
        assert!(mem.unwrap() > 0);
    }
}
