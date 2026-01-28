# rocprofv2 低 Overhead 測量機制分析

## 摘要

rocprofv2 的測量幾乎沒有 overhead，因為它使用 **GPU 硬體級別的時間戳記錄**，而不是軟體插樁。

> **注意**：rocprofv2 只能測量整個 kernel 的執行時間。如需測量 kernel **內部區段**，請使用 [MDR @TIMESTAMP](./timestamp_profiling_design.md#工具定位與價值分析2026-01-28)。

---

## 關鍵機制：AQL Profile HSA Extension

根據 [ROCProfiler Library Specification](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/reference/rocprofiler_spec.html)：

> "The library is based on **AQL profile AMD-specific HSA extension**"
> 
> "Implementation provides a **hardware-specific low-level performance analysis interface**"

這意味著 rocprofv2 直接使用 GPU 硬體內建的 profiling 功能，而不是在軟體層面添加測量代碼。

---

## 硬體時間戳結構

rocprofv2 記錄的時間戳來自 **GPU 硬體**：

```c
typedef struct {
    uint64_t dispatch;    // dispatch 時間戳（AQL packet 提交時）
    uint64_t begin;       // kernel 開始執行時間戳
    uint64_t end;         // kernel 結束執行時間戳
    uint64_t complete;    // completion signal 時間戳
} rocprofiler_dispatch_record_t;
```

來源：[ROCProfiler Spec - Intercepting API](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/reference/rocprofiler_spec.html#intercepting-api)

### 時間戳記錄流程

```
時間軸：

  AQL Packet     Kernel         Kernel        Completion
   Submit      First Instr    Last Instr      Signal
     |             |              |              |
     v             v              v              v
  [dispatch]    [begin]        [end]        [complete]
     
     ↑             ↑              ↑              ↑
     └─────────────┴──────────────┴──────────────┘
           這些都是 GPU 硬體自動記錄的時間戳
           不需要在 kernel 代碼中插入任何指令
```

---

## 為什麼 Overhead 極低？

### 1. 非軟體插樁

| 方法 | 機制 | Overhead |
|------|------|----------|
| 軟體插樁 | 在 kernel 代碼中插入 `s_memtime` 等指令 | 有（指令執行時間）|
| **rocprofv2** | GPU 硬體自動記錄時間戳 | **幾乎為零** |

### 2. 硬體計數器

> "The profiling includes **hardware performance counters** with complex performance metrics"

GPU 內建的硬體計數器在 kernel 執行時自動更新，不需要額外的軟體操作。

### 3. HSA 信號機制

rocprofv2 使用 HSA completion signal 來獲取 kernel 結束時間：

```c
// 從 callback_data 獲取時間戳
const rocprofiler_dispatch_record_t *record = callback_data->record;
uint64_t kernel_duration = record->end - record->begin;
```

這是 HSA runtime 本身就會做的事情，不是額外添加的。

---

## 實驗驗證

### 我們的實測數據（2026-01-28）

| 測量方式 | vectorAdd kernel 時間 | 說明 |
|---------|---------------------|------|
| rocprofv2（硬體測量）| **1,681 ticks** | 純 kernel 執行 |
| s_memtime（軟體插樁）| **1,748 ticks** | 包含備份指令開銷 |
| 差異 | **67 ticks (~4%)** | s_memtime 的額外指令開銷 |

### 結論

- rocprofv2 測量的是**純 kernel 執行時間**
- s_memtime 測量的時間稍長，因為包含了 `s_waitcnt` + `v_mov_b32` 備份指令
- 這證明 rocprofv2 **沒有引入額外的 overhead**

---

## 官方文檔引用

### 1. ROCProfiler Library Specification

> "The goal of the implementation is to provide a **hardware-specific low-level performance analysis interface** for profiling of GPU compute applications."

來源：https://rocm.docs.amd.com/projects/rocprofiler/en/latest/reference/rocprofiler_spec.html

### 2. Dispatch Counting Service

> "Dispatch counting mode collects counters on a **per-kernel launch basis**... allowing only a single kernel to execute in hardware at a time."

來源：https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/counter_collection_services.html

### 3. AQL Profile Extension

> "Implementation is based on **AMD-specific AQLprofile HSA extension**"

來源：https://rocm.docs.amd.com/projects/rocprofiler/en/latest/reference/rocprofiler_spec.html

---

## 總結

rocprofv2 的低 overhead 來自於：

1. **硬體級別測量** - 使用 AQL profile HSA extension
2. **自動時間戳記錄** - GPU 在 kernel dispatch/begin/end/complete 時自動記錄
3. **無軟體插樁** - 不需要在 kernel 代碼中插入額外指令
4. **HSA runtime 整合** - 利用現有的 completion signal 機制

這就是為什麼 rocprofv2 測量的 kernel 執行時間與我們使用 `s_memtime` 直接在 kernel 內部測量的結果幾乎一致（差異 < 5%）。

---

## 附錄：為什麼 Dispatch 沒有太多 Overhead？

根據 [HSA Foundation 官方簡報](https://www.slideshare.net/slideshow/hsa4122-ianbratt/28824324)（Ian Bratt, ARM）：

### 傳統方式 vs HSA User-Mode Queueing

**傳統方式（大量 overhead）：**
```
Application → Transfer buffer → Copy/Map Memory
     ↓              ↓
    OS    →    Queue Job → Schedule Job    ← 需要 OS context switch
     ↓              ↓
   GPU    →    Start Job → Finish Job
     ↓              ↓
Application ← Get Buffer ← Copy/Map Memory
```

**HSA 方式（幾乎沒有 overhead）：**
```
Application → Queue Job → [doorbell]
     ↓              ↓
   GPU    →    Start Job → Finish Job
```

### 關鍵機制

> "**Enables user space applications to directly, without OS intervention, enqueue jobs**"
> 
> — HSA Foundation, "HSA Queueing Mode"

1. **User-Mode Queueing**
   - 應用程式直接寫入 shared memory queue
   - 不需要經過 OS kernel
   - 不需要 context switch

2. **Doorbell 機制**
   - 寫完 AQL packet 後，直接觸發硬體 doorbell
   - GPU hardware scheduler 立即收到通知
   - 延遲在微秒級別

3. **Shared Virtual Memory**
   - CPU 和 GPU 共享同一個虛擬記憶體空間
   - 不需要複製資料（傳指標，不傳資料）

4. **Hardware Coherency**
   - 硬體自動維護 cache 一致性
   - 不需要軟體 flush/invalidate

### Dispatch 時間實測

在我們的實驗中，`dispatch` 到 `begin` 的時間差極小，因為：

```
[dispatch] ─────> [begin] ─────────────> [end] ─────> [complete]
    │                │                      │              │
    └── doorbell ────┘                      │              │
         (硬體直接處理)                    kernel        signal
                                          執行
```

- `dispatch → begin`：硬體級別的 doorbell 響應，通常只有幾百個 clock cycles
- `begin → end`：這才是真正的 kernel 執行時間
- `end → complete`：completion signal 通知

### 參考資料

- HSA Foundation, "HSA Queueing Mode", APU13, November 2013
  - https://www.slideshare.net/slideshow/hsa4122-ianbratt/28824324
- AMD ROCm Documentation: ROCR Runtime
  - https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/what-is-rocr-runtime.html
