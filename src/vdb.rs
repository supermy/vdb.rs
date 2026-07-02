//! Lance 列式文件子集读写、Arrow Schema 解析、零拷贝 RecordBatch、内存映射管理、
//! 追加写事务与版本快照（manifest）管理。
//!
//! 当前实现使用 `storage::save_index/load_index` 完成列式数据持久化；
//! mmap 按需加载作为后续阶段扩展点，预留了文件偏移字段。

use crate::index_ivf_rq::{IvfRabitqIndex, Payload};
use crate::search::{SearchOptions, search};
use crate::storage::{load_index, save_index};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

/// 版本快照 manifest。
///
/// 为什么用独立 JSON 文件：manifest 极小且频繁原子替换，
/// JSON 便于人工查看与调试；真正的列式数据仍用二进制格式存储。
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Manifest {
    version: u64,
    index_file: String,
}

/// 数据库统计信息。
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub version: u64,
    pub num_vectors: usize,
    pub num_partitions: usize,
}

/// 嵌入式数据库。
///
/// `index` 用 RwLock 保护，支持并发搜索；
/// `write_lock` 是实例级写锁，禁止并发十亿级写事务。
pub struct Database {
    index: RwLock<IvfRabitqIndex>,
    dir: PathBuf,
    write_lock: Mutex<()>,
    manifest: Mutex<Manifest>,
}

impl Database {
    /// 在指定目录创建新数据库。
    pub fn create<P: AsRef<Path>>(dir: P, dim: usize) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        let index = IvfRabitqIndex::new(dim);
        let manifest = Manifest {
            version: 0,
            index_file: format!("index-{}.vdb", 0),
        };
        let db = Self {
            index: RwLock::new(index),
            dir,
            write_lock: Mutex::new(()),
            manifest: Mutex::new(manifest.clone()),
        };
        db.save_index_file(&manifest)?;
        db.write_manifest(&manifest)?;
        Ok(db)
    }

    /// 打开已有数据库，加载最新 manifest 指向的版本。
    pub fn open<P: AsRef<Path>>(dir: P) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest = Self::read_manifest(&dir)?;
        let index = load_index(dir.join(&manifest.index_file))?;
        Ok(Self {
            index: RwLock::new(index),
            dir,
            write_lock: Mutex::new(()),
            manifest: Mutex::new(manifest),
        })
    }

    /// 单条插入。
    pub fn insert(&self, vector: &[f32]) -> std::io::Result<u64> {
        self.insert_with_payload(vector, Payload::new())
    }

    /// 带标量 payload 的单条插入。
    ///
    /// 事务边界：
    /// 1. 获取实例级写锁，保证同一时刻只有一个写事务。
    /// 2. 在内存索引中追加向量并分配 id。
    /// 3. 将完整索引写入新版本文件。
    /// 4. 原子替换 manifest。
    ///
    /// 旧版本文件保留，可实现 time-travel。
    pub fn insert_with_payload(&self, vector: &[f32], payload: Payload) -> std::io::Result<u64> {
        let _write_guard = self.write_lock.lock().unwrap();
        let id = {
            let mut index = self.index.write().unwrap();
            index.add_with_payload(vector, payload)
        };
        self.save_and_bump()?;
        Ok(id)
    }

    /// 批量带 payload 插入。
    ///
    /// 与单条插入相比，批量插入只在内存中追加全部向量，最后一次性写入新版本，
    /// 避免每条记录都产生一个全量索引快照，显著降低磁盘占用与写入耗时。
    /// 返回分配的起始 id（第一条向量的 id）。
    pub fn batch_insert_with_payload(
        &self,
        vectors: &[Vec<f32>],
        payloads: Vec<Payload>,
    ) -> std::io::Result<u64> {
        assert_eq!(
            vectors.len(),
            payloads.len(),
            "vectors and payloads length mismatch"
        );
        let _write_guard = self.write_lock.lock().unwrap();
        let first_id = {
            let mut index = self.index.write().unwrap();
            let mut first = None;
            for (vector, payload) in vectors.iter().zip(payloads.into_iter()) {
                let id = index.add_with_payload(vector, payload);
                if first.is_none() {
                    first = Some(id);
                }
            }
            first.unwrap_or(0)
        };
        self.save_and_bump()?;
        Ok(first_id)
    }

    /// 搜索。
    pub fn search(&self, query: &[f32], options: &SearchOptions) -> Vec<(u64, f32)> {
        let index = self.index.read().unwrap();
        search(&index, query, options, None)
    }

    /// 当前统计。
    pub fn stats(&self) -> Stats {
        let index = self.index.read().unwrap();
        let manifest = self.manifest.lock().unwrap();
        Stats {
            version: manifest.version,
            num_vectors: index.len(),
            num_partitions: index.num_partitions(),
        }
    }

    fn save_and_bump(&self) -> std::io::Result<()> {
        let mut manifest = self.manifest.lock().unwrap();
        let new_version = manifest.version + 1;
        let new_manifest = Manifest {
            version: new_version,
            index_file: format!("index-{}.vdb", new_version),
        };
        self.save_index_file(&new_manifest)?;
        self.write_manifest(&new_manifest)?;
        *manifest = new_manifest;
        Ok(())
    }

    /// 清理旧版本索引文件，只保留 manifest 指向的最新版本。
    ///
    /// 为什么需要 compact：append-only 设计下每次写入都会生成新的 index-N.vdb，
    /// 长期运行会积累大量旧版本。compact 释放这些历史快照占用的磁盘空间，
    /// 同时保证当前 manifest 原子性不被破坏。
    pub fn compact(&self) -> std::io::Result<usize> {
        let _write_guard = self.write_lock.lock().unwrap();
        let manifest = self.manifest.lock().unwrap();
        let keep = self.dir.join(&manifest.index_file);
        let mut removed = 0;
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("index-") && name.ends_with(".vdb") && path != keep {
                    fs::remove_file(&path)?;
                    removed += 1;
                }
            }
        }
        Ok(removed)
    }

    fn save_index_file(&self, manifest: &Manifest) -> std::io::Result<()> {
        let index = self.index.read().unwrap();
        let path = self.dir.join(&manifest.index_file);
        save_index(path, &index)
    }

    fn write_manifest(&self, manifest: &Manifest) -> std::io::Result<()> {
        let tmp = self.dir.join("manifest.json.tmp");
        let content = serde_json::to_vec_pretty(manifest)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(&tmp, content)?;
        fs::rename(&tmp, self.dir.join("manifest.json"))?;
        Ok(())
    }

    fn read_manifest(dir: &Path) -> std::io::Result<Manifest> {
        let path = dir.join("manifest.json");
        let content = fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(unix)]
use crate::mmap_index::MmapIndex;

/// 基于分块 mmap 的只读数据库视图。
///
/// 与 `Database` 的区别：
/// - 打开时不把分区数据/原始向量拷贝到内存，只加载元数据；
/// - 查询时通过 `ChunkedMmapStorage` 按需 fault 64MB 块，配合 LRU 缓存；
/// - 不支持写入，适合启动速度敏感、内存受限的只读场景。
#[cfg(unix)]
pub struct MmapDatabase {
    index: MmapIndex,
    manifest: Manifest,
}

#[cfg(unix)]
impl MmapDatabase {
    /// 打开已有数据库的最新版本，使用分块 mmap 零拷贝加载索引。
    pub fn open<P: AsRef<Path>>(dir: P) -> std::io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest = Database::read_manifest(&dir)?;
        let index = MmapIndex::open(dir.join(&manifest.index_file))?;
        Ok(Self { index, manifest })
    }

    /// 向量搜索。
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(u64, f32)> {
        self.index.search(query, k, nprobe)
    }

    /// 当前统计。
    pub fn stats(&self) -> Stats {
        Stats {
            version: self.manifest.version,
            num_vectors: self.index.len(),
            num_partitions: self.index.num_partitions(),
        }
    }

    /// 向量维度。
    pub fn dim(&self) -> usize {
        self.index.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_insert_search() {
        let dir = TempDir::new().unwrap();
        let db = Database::create(dir.path(), 64).unwrap();
        let v: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
        let id = db.insert(&v).unwrap();
        assert_eq!(id, 0);

        let results = db.search(&v, &SearchOptions::default());
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_persist_and_reopen() {
        let dir = TempDir::new().unwrap();
        let db = Database::create(dir.path(), 64).unwrap();
        for _ in 0..10 {
            let v: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
            db.insert(&v).unwrap();
        }
        let stats_before = db.stats();

        let db2 = Database::open(dir.path()).unwrap();
        let stats_after = db2.stats();
        assert_eq!(stats_after.num_vectors, stats_before.num_vectors);
        assert_eq!(stats_after.num_partitions, stats_before.num_partitions);
        assert_eq!(stats_after.version, stats_before.version);

        let query: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
        let results = db2.search(&query, &SearchOptions::default());
        assert_eq!(results.len(), 10);
    }

    #[test]
    #[cfg(unix)]
    fn test_mmap_database_zero_copy_open() {
        let dir = TempDir::new().unwrap();
        let db = Database::create(dir.path(), 64).unwrap();
        let mut vectors = Vec::new();
        for _ in 0..100 {
            let v: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
            vectors.push(v);
        }
        for v in &vectors {
            db.insert(v).unwrap();
        }

        let mmap_db = MmapDatabase::open(dir.path()).unwrap();
        assert_eq!(mmap_db.dim(), 64);
        assert_eq!(mmap_db.stats().num_vectors, 100);

        let query: Vec<f32> = (0..64).map(|_| rand::random::<f32>()).collect();
        let results = mmap_db.search(&query, 10, 0);
        assert_eq!(results.len(), 10);
    }
}
