# MDR Printf - AMD ISA Assembly Debug Tool

> 在 AMD GPU 組合語言（.s 檔案）中插入 `printf` 除錯功能

## 簡介

`mdr_printf.py` 是一個專為 AMD GPU 開發者設計的除錯工具，讓你可以在 ISA（Instruction Set Architecture）階段，透過簡單的註解標記來觀察暫存器值和計算表達式結果。

### 核心功能

- ✅ **Printf 注入**：在任意位置印出 VGPR 暫存器值
- ✅ **表達式計算**：支援 `+`, `-`, `*`, `/` 四則運算
- ✅ **條件印出**：只在特定 thread 執行時印出
- ✅ **生成 HSACO**：產生可執行的 HSA Code Object

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
# 編譯帶除錯的 HSACO
python3 mdr_printf.py input.s --output-dir output

# 執行測試
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
| `reg` | ⭕ | 暫存器（可多個，逗號分隔） | `v6` 或 `v6,v7` |
| `expr` | ⭕ | 表達式（可多個，分號分隔） | `"v6*v7"` 或 `"v6+v7; v6-v7"` |
| `type` | ✅ | 資料類型（需與 reg+expr 數量匹配） | `f32` 或 `f32,i32` |
| `cond` | ❌ | 條件（建議使用） | `tid_eq(0)` |

⭕ = `reg` 和 `expr` 至少需要一個

---

## 功能支援

### ✅ 支援的功能

| 功能 | 說明 | 範例 |
|------|------|------|
| **直接印出暫存器** | 印出 VGPR 值 | `reg=v6,v7 type=f32,f32` |
| **表達式計算** | 四則運算 | `expr="v6*v7"` |
| **混合模式** | 暫存器 + 表達式 | `reg=v6 expr="v6*2" type=f32,f32` |
| **條件印出** | 只在特定 thread 印出 | `cond=tid_eq(0)` |
| **多表達式** | 分號分隔 | `expr="v6+v7; v6-v7"` |
| **暫存器保護** | 自動保護 VGPR + SGPR (s[4:N]) | 無需設定，自動啟用 |

### 支援的資料類型

| 類型 | 說明 | 格式符號 |
|------|------|----------|
| `f32` | 32-bit 浮點數 | `%f` |
| `f64` | 64-bit 浮點數 | `%f` |
| `i32` | 32-bit 整數 | `%d` |
| `i64` | 64-bit 整數 | `%ld` |

### 支援的條件

| 條件 | 說明 |
|------|------|
| `tid_eq(N)` | thread_id_x == N |
| `tid_lt(N)` | thread_id_x < N |
| `tid_le(N)` | thread_id_x <= N |
| `tid_gt(N)` | thread_id_x > N |
| `tid_ge(N)` | thread_id_x >= N |

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

### 重要限制

| 限制 | 說明 | 建議 |
|------|------|------|
| **@PRINT 數量** | 每個 kernel 建議 1-2 個 | 超過 3 個可能影響程式執行 |
| **暫存器類型** | 僅支援 VGPR（v0, v1, ...） | SGPR 印出尚不支援 |
| **s_barrier 衝突** | shared memory kernel 可能卡住 | 使用 `--no-printf` 或 `cond=tid_eq(0)` |

### ❌ 不支援的功能

| 功能 | 原因 |
|------|------|
| SGPR 直接印出 | MLIR 轉換限制 |
| AGPR 直接印出 | 需要額外指令 |
| 複雜的控制流 | scf.if 與 EXEC mask 衝突 |
| 無條件的多 thread 印出 | register pressure |

---

## 範例

### 範例 1：基本暫存器印出

```asm
	global_load_dword v6, v[4:5], off      ; 載入 A[tid]
	global_load_dword v7, v[2:3], off      ; 載入 B[tid]
	s_waitcnt vmcnt(0)
; @PRINT cond=tid_eq(0) fmt="A=%f, B=%f" reg=v6,v7 type=f32,f32
	v_add_f32_e32 v2, v6, v7
; @PRINT cond=tid_eq(0) fmt="C=%f" reg=v2 type=f32
```

**輸出：**
```
A=0.000000, B=0.000000
C=0.000000
```

### 範例 2：表達式計算

```asm
	v_mov_b32_e32 v2, 45
; @PRINT cond=tid_eq(0) fmt="v2=%d, v2*2=%d, v2^2=%d" reg=v2 expr="v2*2; v2*v2" type=i32,i32,i32
```

**輸出：**
```
v2=45, v2*2=90, v2^2=2025
```

### 範例 3：浮點運算

```asm
	s_waitcnt vmcnt(0)
; @PRINT cond=tid_eq(0) fmt="A*B=%f" expr="v6*v7" type=f32
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
[3] 注入 printf 程式碼
    ├── Register Clobbering (保護 VGPR + SGPR)
    ├── Value Binding (讀取暫存器)
    ├── Expression Eval (表達式計算)
    ├── Condition Check (條件判斷)
    └── gpu.printf
    │
    ▼
[4] mlir-opt → ROCDL → LLVM
    │
    ▼
[5] 修復 metadata (hidden_hostcall_buffer)
    │
    ▼
[6] 重命名衝突標籤 (.LBB* → .LBBPRINTF*)
    │
    ▼
[7] llvm-mc → ld.lld → .hsaco
```

---

## 測試案例

所有測試案例位於 `Track_B/kernel_testcases/`：

| 測試 | 類型 | Printf | 表達式 | 狀態 |
|------|------|:------:|:------:|:----:|
| test_01_vector_add | float_add | ✅ | ✅ | ✅ |
| test_02_scalar_ops | int_scalar | ✅ | ✅ | ✅ |
| test_03_memory_ops | int_mem | ✅ | ✅ | ✅ |
| test_04_conditional | int_cond | ✅ | ✅ | ✅ |
| test_05_loop | int_loop | ✅ | ✅ | ✅ |
| test_06_shared_memory | int_shared | ⚠️ | - | ⏱️* |
| test_07_multi_kernels | multi | ✅ | - | ✅ |

*test_06 因 s_barrier 衝突可能超時，但 printf 輸出正常

---

## 相關工具

| 工具 | 用途 |
|------|------|
| `mdr_printf.py` | 主工具 |
| `amdisa-translate` | ISA → GPU MLIR 轉換 |
| `mlir-opt` | MLIR 優化 |
| `llvm-mc` | 組合器 |
| `ld.lld` | 連結器 |
| `universal_hsaco_runner` | HSACO 執行器 |

---

## 目錄結構

```
Project-MDR/
├── mdr_printf.py              # 主工具
├── examples/                   # 使用範例
│   ├── 01_vector_add/
│   │   ├── original.s         # 原始程式碼
│   │   └── with_debug.s       # 加入 @PRINT 後
│   └── 02_expression_calc/
│       └── with_expression.s  # 表達式計算範例
├── Track_A/                    # GPU Dialect 探索
├── Track_B/                    # ISA 提升工具鏈
│   ├── amdisa-toolkit/        # amdisa-translate
│   └── kernel_testcases/      # 測試案例
└── README.md                   # 本文件
```

---
