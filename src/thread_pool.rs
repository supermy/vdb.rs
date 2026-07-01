//! 固定工作线程池 + 自旋锁任务队列。
//!
//! batchInsert 和 batchSearch 复用线程池，通过将任务发送到共享队列，
//! 由固定数量的工作线程并行消费。
//!
//! 为什么用固定线程池：避免查询高并发时反复创建/销毁线程的开销，
//! 同时将并发度与 CPU 核心数解耦，便于在资源受限设备上限流。

use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread;

/// 可发送到线程池的任务。
type Task = Box<dyn FnOnce() + Send + 'static>;

/// 固定大小线程池。
pub struct ThreadPool {
    /// 发送端放在 Mutex 中，以便多个生产者同时提交任务。
    sender: Mutex<mpsc::Sender<Task>>,
    /// 持有工作线程句柄，用于 `join` 等待全部任务完成。
    workers: Vec<thread::JoinHandle<()>>,
    /// 活跃任务计数 + 条件变量，支持 `wait` 等待而不消耗线程池。
    active: Arc<(Mutex<usize>, Condvar)>,
}

impl ThreadPool {
    /// 创建拥有 `size` 个工作线程的线程池。
    ///
    /// 内存策略：线程池本身只持有通道发送端与句柄，不缓存业务结果，
    /// 业务层通过闭包内的 Arc/Mutex 收集输出，避免线程池 API 复杂化。
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "ThreadPool size must be > 0");
        let (sender, receiver) = mpsc::channel::<Task>();
        let receiver = Arc::new(Mutex::new(receiver));
        let active = Arc::new((Mutex::new(0), Condvar::new()));

        let mut workers = Vec::with_capacity(size);
        for _ in 0..size {
            let rx = Arc::clone(&receiver);
            let handle = thread::spawn(move || {
                loop {
                    // 用 Mutex 保护接收端， worker 间互斥取任务。
                    // 通道关闭时 recv 返回 Err，线程退出。
                    let task = match rx.lock().unwrap().recv() {
                        Ok(t) => t,
                        Err(_) => break,
                    };
                    // execute() 已经把递减逻辑包在 task 里，worker 只负责调用。
                    task();
                }
            });
            workers.push(handle);
        }

        Self {
            sender: Mutex::new(sender),
            workers,
            active,
        }
    }

    /// 提交一个任务到线程池。
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        // 先递增计数，避免任务在 send 与执行之间出现 wait 漏判。
        {
            let (lock, _) = &*self.active;
            let mut count = lock.lock().unwrap();
            *count += 1;
        }
        let active = Arc::clone(&self.active);
        let task = Box::new(move || {
            f();
            let (lock, cvar) = &*active;
            let mut count = lock.lock().unwrap();
            *count -= 1;
            cvar.notify_all();
        });
        self.sender.lock().unwrap().send(task).unwrap();
    }

    /// 等待当前已提交的所有任务执行完毕（不关闭线程池，可复用）。
    ///
    /// untested: `wait` 由上层批量接口使用，当前测试通过 `join` 即可验证任务完成语义。
    pub fn wait(&self) {
        let (lock, cvar) = &*self.active;
        let mut count = lock.lock().unwrap();
        while *count > 0 {
            count = cvar.wait(count).unwrap();
        }
    }

    /// 等待所有已提交任务执行完毕，并关闭线程池。
    ///
    /// 执行顺序：先 drop 发送端，使工作线程 recv 失败并退出循环，
    /// 然后 join 每个线程，确保所有任务内存被安全释放。
    pub fn join(self) {
        drop(self.sender);
        for worker in self.workers {
            worker.join().unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_thread_pool_executes_tasks() {
        let pool = ThreadPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        for _ in 0..100 {
            let c = Arc::clone(&counter);
            pool.execute(move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }
        pool.join();
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }
}
