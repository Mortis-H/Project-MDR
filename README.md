# MDR - AMD ISA Assembly Debug Toolkit

> 在 AMD GPU 組合語言（.s 檔案）中插入除錯功能

## 簡介

MDR（Memory Debug and Register）是專為 AMD GPU 開發者設計的 ISA 層級除錯工具，讓你可以透過簡單的註解標記來觀察暫存器值和計算表達式結果。

### 核心功能

- ✅ **Printf 注入**：在任意位置印出 VGPR 和 SGPR 暫存器值
- ✅ **快照機制**：在 `@PRINT` 標記位置捕捉暫存器值，真正觀測該時間點的狀態
- ✅ **表達式計算**：支援 `+`, `-`, `*`, `/` 四則運算
- ✅ **VGPR/SGPR 支援**：同時支援向量暫存器和純量暫存器
- ✅ **生成 HSACO**：產生可執行的 HSA Code Object
- ✅ **自動測試**：內建 `--test` 選項快速驗證

---

## 快速開始

### 環境設置

```bash
# 設定 LLVM 和 amdisa-translate 路徑
export PATH="<PROJECT_DIR>/Track_B/amdisa-toolkit/build/bin:<LLVM_BUILD_DIR>/bin:$PATH"

# 進入專案目錄
cd <PROJECT_DIR>
```

### 基本用法

```bash
# 編譯帶 printf 除錯的 HSACO
python3 mdr_printf.py input.s --output-dir output

# 一鍵編譯並測試
python3 mdr_printf.py input.s --output-dir output --test --test-size 64

# 手動執行測試
./Track_B/kernel_testcases/universal_hsaco_runner \
    output/input_debug_injected.hsaco \
    kernel_name kernel_type size
```

---

## @PRINT 指令語法

### 基本格式

```asm
; @PRINT [cond=條件] fmt="格式字串" reg=暫存器 type=類型
; @PRINT [cond=條件] fmt="格式字串" expr="表達式" type=類型
```

### 參數說明

| 參數 | 必須 | 說明 | 範例 |
|------|:----:|------|------|
| `fmt` | ✅ | printf 格式字串 | `"value=%f"` |
| `reg` | ⭕ | 暫存器（可多個，逗號分隔） | `v6` 或 `v6,v7` 或 `s4,s2` |
| `expr` | ⭕ | 表達式（可多個，分號分隔） | `"v6*v7"` 或 `"v6+v7; v6-v7"` |
| `type` | ✅ | 資料類型（需與 reg+expr 數量匹配） | `f32` 或 `f32,i32` |
| `cond` | ❌ | 條件（基於暫存器值） | `v6_eq(0.0)` 或 `s4_gt(0)` |

⭕ = `reg` 和 `expr` 至少需要一個

---

## 功能支援

### ✅ 支援的功能

| 功能 | 說明 | 範例 |
|------|------|------|
| **VGPR 印出** | 印出向量暫存器值 | `reg=v6,v7 type=f32,f32` |
| **SGPR 印出** | 印出純量暫存器值 | `reg=s4,s2 type=i32,i32` |
| **快照機制** | 在 `@PRINT` 位置捕捉暫存器值 | Before/After 觀察 |
| **條件印出** | 基於暫存器值過濾輸出 | `cond=v6_eq(0.0)` |
| **表達式計算** | 四則運算 | `expr="v6*v7"` |
| **混合模式** | 暫存器 + 表達式 | `reg=v6 expr="v6*2" type=f32,f32` |
| **多表達式** | 分號分隔 | `expr="v6+v7; v6-v7"` |
| **自動測試** | `--test` 選項 | `--test --test-size 64` |

### 支援的暫存器類型

| 暫存器 | 說明 | 快照範圍 |
|--------|------|----------|
| **VGPR** | 向量暫存器 (v0, v1, ...) | 動態計算（基於 printf overhead） |
| **SGPR** | 純量暫存器 (s0, s1, ...) | s20+ |

> 💡 VGPR 快照起始位置會根據 `printf` 需要的暫存器數量動態調整，確保不與 printf 內部使用的暫存器衝突。

### 支援的資料類型

| 類型 | 說明 | 格式符號 |
|------|------|----------|
| `f32` | 32-bit 浮點數 | `%f` |
| `f64` | 64-bit 浮點數 | `%f` |
| `i32` | 32-bit 整數 | `%d` |
| `i64` | 64-bit 整數 | `%ld` |

### 支援的表達式運算

| 運算 | f32 | i32 | 範例 |
|------|:---:|:---:|------|
| 加法 `+` | ✅ | ✅ | `v6+v7` |
| 減法 `-` | ✅ | ✅ | `v6-v7` |
| 乘法 `*` | ✅ | ✅ | `v6*v7` |
| 除法 `/` | ✅ | ✅ | `v6/v7` |
| 括號 `()` | ✅ | ✅ | `(v6+v7)*2.0` |
| 常數 | ✅ | ✅ | `4.0`, `55` |

---

## ⚠️ 使用限制

| 限制 | 說明 | 建議 |
|------|------|------|
| **@PRINT 數量** | 每個 kernel 建議 1-3 個 | 過多可能影響效能 |
| **s_barrier 衝突** | shared memory kernel 可能卡住 | 減少 printf 數量 |

### ✅ 條件式 printf 支援

支援基於**暫存器值**的條件過濾：

| 條件格式 | 說明 | 範例 |
|----------|------|------|
| `REG_eq(N)` | 暫存器 == N | `v6_eq(0.0)`, `s4_eq(64)` |
| `REG_ne(N)` | 暫存器 != N | `v0_ne(0)` |
| `REG_lt(N)` | 暫存器 < N | `v6_lt(10.0)` |
| `REG_le(N)` | 暫存器 <= N | `v0_le(7)` |
| `REG_gt(N)` | 暫存器 > N | `s4_gt(0)` |
| `REG_ge(N)` | 暫存器 >= N | `v6_ge(1.0)` |

**自動化行為**：
- 條件暫存器和類型從 `cond=` 自動提取（有小數點 → f32，沒有 → i32）
- 如果條件暫存器不在 `reg=` 中，會自動創建獨立快照

**使用範例**：
```asm
; 只印出 A > 2.0 的 thread
; v6 在 reg= 中，直接使用它的快照值作為條件
; @PRINT cond=v6_gt(2.0) fmt="A=%f, B=%f" reg=v6,v7 type=f32,f32

; v6 不在 reg= 中，自動為它創建獨立快照用於條件判斷
; @PRINT cond=v6_gt(2.0) fmt="C=%f" reg=v2 type=f32
```

### ❌ 不支援的功能

| 功能 | 原因 |
|------|------|
| AGPR 直接印出 | 需要額外指令轉換 |

---

## 範例

### 範例 1：VGPR Before/After 觀察

```asm
	global_load_dword v6, v[4:5], off      ; 載入 A[tid]
	global_load_dword v7, v[2:3], off      ; 載入 B[tid]
	s_waitcnt vmcnt(0)
; @PRINT fmt="Before ADD: A(v6)=%f, B(v7)=%f, C(v2)=%f" reg=v6,v7,v2 type=f32,f32,f32
	v_add_f32_e32 v2, v6, v7               ; C = A + B
; @PRINT fmt="After ADD:  A(v6)=%f, B(v7)=%f, C(v2)=%f" reg=v6,v7,v2 type=f32,f32,f32
```

**輸出（快照機制讓我們能觀察同一暫存器在不同時間點的值）：**
```
Before ADD: A(v6)=0.000000, B(v7)=0.000000, C(v2)=0.000611  ← C 是垃圾值
Before ADD: A(v6)=1.000000, B(v7)=2.000000, C(v2)=0.000611
After ADD:  A(v6)=0.000000, B(v7)=0.000000, C(v2)=0.000000  ← C = A + B 正確
After ADD:  A(v6)=1.000000, B(v7)=2.000000, C(v2)=3.000000
```

### 範例 1b：條件式 printf（只印 A > 2.0 的 thread）

```asm
	s_waitcnt vmcnt(0)
; v6 在 reg= 中，直接使用它的快照值作為條件
; @PRINT cond=v6_gt(2.0) fmt="Before ADD (A>2): A(v6)=%f, B(v7)=%f" reg=v6,v7 type=f32,f32
	v_add_f32_e32 v2, v6, v7               ; C = A + B
; v6 不在 reg= 中，自動為它創建獨立快照用於條件判斷
; @PRINT cond=v6_gt(2.0) fmt="After ADD (A>2): C(v2)=%f" reg=v2 type=f32
```

**輸出（只印出 A > 2.0 的 thread）：**
```
Before ADD (A>2): A(v6)=3.000000, B(v7)=6.000000
Before ADD (A>2): A(v6)=4.000000, B(v7)=8.000000
Before ADD (A>2): A(v6)=5.000000, B(v7)=10.000000
After ADD (A>2): C(v2)=9.000000
After ADD (A>2): C(v2)=12.000000
After ADD (A>2): C(v2)=15.000000
```

### 範例 2：SGPR 印出

```asm
	s_load_dword s4, s[0:1], 0x18          ; 載入 n
	s_mul_i32 s2, s2, s3                   ; 計算 base_idx
; @PRINT fmt="[SGPR] n(s4)=%d, base_idx(s2)=%d" reg=s4,s2 type=i32,i32
	v_add_u32_e32 v0, s2, v0
```

**輸出：**
```
[SGPR] n(s4)=64, base_idx(s2)=0
[SGPR] n(s4)=64, base_idx(s2)=0
...（所有 thread 輸出相同值，因為 SGPR 是 wavefront 共享的）
```

### 範例 3：混合 VGPR + SGPR

```asm
; @PRINT fmt="n=%d, A=%f, B=%f" reg=s4,v6,v7 type=i32,f32,f32
```

### 範例 4：表達式計算

```asm
	v_mov_b32_e32 v2, 45
; @PRINT fmt="v2=%d, v2*2=%d, v2^2=%d" reg=v2 expr="v2*2; v2*v2" type=i32,i32,i32
```

**輸出：**
```
v2=45, v2*2=90, v2^2=2025
```

---

## 命令列選項

```bash
python3 mdr_printf.py input.s [選項]
```

| 選項 | 說明 | 預設值 |
|------|------|--------|
| `--output-dir DIR` | 輸出目錄 | `debug_output` |
| `--chip CHIP` | GPU 架構 | `gfx950` |
| `--no-printf` | 禁用 printf（僅功能驗證） | - |
| `--kernel-name NAME` | 指定 kernel 名稱 | 自動偵測 |
| `--kernel-type TYPE` | 指定 kernel 類型 | 自動偵測 |
| `--test` | 自動執行測試 | - |
| `--test-size N` | 測試數據大小 | `64` |
| `--dry-run` | 僅解析不執行 | - |

---

## Pipeline 流程

```
輸入 .s (含 @PRINT 註解)
    │
    ▼
[1] 解析 @PRINT 指令
    │
    ▼
[2] amdisa-translate -emit=gpu → GPU MLIR
    │
    ▼
[3] 注入 printf 程式碼（快照機制）
    ├── Kernel 開頭: Kernarg Backup (保存 s[0:1] → s[18:19])
    ├── @PRINT 位置: VGPR 快照 (v_mov_b32 → v32+)
    ├── @PRINT 位置: SGPR 快照 (s_mov_b32 → s20+)
    ├── Kernel 結尾: Printf Section
    │   ├── Kernarg Restore (從 s[18:19] 恢復)
    │   ├── Value Binding (從快照讀取)
    │   └── gpu.printf
    │
    ▼
[4] mlir-opt → ROCDL → LLVM
    │
    ▼
[5] 修復 metadata
    ├── hidden_hostcall_buffer (printf 支援)
    ├── .amdhsa_next_free_vgpr (包含快照暫存器)
    └── .amdhsa_next_free_sgpr (包含快照暫存器)
    │
    ▼
[6] 重命名衝突標籤 (.LBB* → .LBBPRINTF*)
    │
    ▼
[7] llvm-mc → ld.lld → .hsaco
```

### 快照機制原理

傳統做法是在 kernel 結尾統一執行所有 `printf`，但這樣只能讀取暫存器的**最終狀態**。

快照機制解決了這個問題：

1. **在 `@PRINT` 位置插入快照指令**：
   - VGPR: `v_mov_b32 vN, v6` - 複製到動態計算的範圍
   - SGPR: `s_mov_b32 s20, s4` - 複製到 s20+ 範圍
   - 條件暫存器：如果不在 `reg=` 中，會創建獨立快照
2. **printf 使用快照暫存器**：讀取的是 `@PRINT` 當時的暫存器值，而非最終值

```
@PRINT 標記位置:
    v_mov_b32 v44, v6     ; VGPR 快照：v6 → v44（動態計算）
    s_mov_b32 s20, s4     ; SGPR 快照：s4 → s20
    v_mov_b32 v47, v6     ; 條件暫存器快照（如果 v6 不在 reg= 中）

Kernel 結尾:
    gpu.printf 使用 v44, s20  ; 印出快照值
    scf.if (v47 > 2.0)        ; 使用條件暫存器快照
```

---

## 測試案例

所有測試案例位於 `Track_B/kernel_testcases/`：

| 測試 | 類型 | Printf | SGPR | 狀態 |
|------|------|:------:|:----:|:----:|
| test_01_vector_add | float_add | ✅ | ✅ | ✅ |
| test_02_scalar_ops | int_scalar | ✅ | ✅ | ✅ |
| test_03_memory_ops | int_mem | ✅ | - | ✅ |
| test_04_conditional | int_cond | ✅ | - | ✅ |
| test_05_loop | int_loop | ✅ | - | ✅ |
| test_06_shared_memory | int_shared | ⚠️ | - | ⏱️* |
| test_07_multi_kernels | multi | ✅ | - | ✅ |

*test_06 因 s_barrier 衝突可能超時

---

## 相關工具

| 工具 | 用途 |
|------|------|
| `mdr_printf.py` | Printf 除錯工具 |
| `amdisa-translate` | ISA → GPU MLIR 轉換 |
| `mlir-opt` | MLIR 優化 |
| `llvm-mc` | 組合器 |
| `ld.lld` | 連結器 |
| `universal_hsaco_runner` | HSACO 執行器 |

---

## 目錄結構

```
Project-MDR/
├── mdr_printf.py              # Printf 除錯工具
├── examples/                   # 使用範例
│   ├── 01_vector_add/
│   │   ├── original.s         # 原始程式碼
│   │   └── with_debug.s       # 加入 @PRINT 後
│   └── 02_expression_calc/
│       └── with_expression.s  # 表達式計算範例
├── Track_B/                    # ISA 提升工具鏈
│   ├── amdisa-toolkit/        # amdisa-translate
│   └── kernel_testcases/      # 測試案例
└── README.md                   # 本文件
```

---

## 更新日誌

### v1.3 (2026-01-21)
- ✅ 簡化條件式 printf 語法：移除 `cond_reg=` 和 `cond_type=` 參數
- ✅ 條件暫存器和類型從 `cond=` 自動提取
- ✅ 如果條件暫存器不在 `reg=` 中，自動創建獨立快照

### v1.2 (2026-01-21)
- ✅ 新增暫存器值條件式 printf（`v6_eq(0.0)`, `s4_gt(0)` 等）
- ✅ VGPR 快照起始位置改為動態計算（根據 printf 值數量）
- ⚠️ `tid_*` 條件已移除（會破壞 SGPR）

### v1.1 (2026-01-21)
- ✅ 新增 SGPR 快照支援
- ✅ 修正 VGPR 快照暫存器分配（避免被 printf 覆蓋）
- ✅ 修正 `.amdhsa_next_free_vgpr/sgpr` 自動更新

### v1.0
- ✅ VGPR 快照機制
- ✅ 表達式計算
- ✅ 基本 printf 功能
