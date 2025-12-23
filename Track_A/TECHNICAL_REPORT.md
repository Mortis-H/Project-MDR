# Track_A: MDR (MLIR-to-AMDGPU Device Pipeline) 技術報告

## 📌 專案概述

### 目標
開發一個能夠在 AMD GPU kernel 的組合語言（ISA）中插入高階除錯邏輯（DSL）的工具鏈，實現 **低階組合語言** 與 **高階抽象** 的混合編程。

### 願景 vs. 現況
- **最終願景**：ISA → Raise to MLIR → 插入 DSL → Lower to ISA
- **目前實作（PoC）**：硬編碼 MLIR（包含 inline_asm + DSL）→ Lower to ISA

目前 `mdr.py` 是一個**概念驗證（Proof of Concept）**，展示了如何在 MLIR 層級混合使用低階組合語言和高階 GPU Dialect，但尚未實現自動化的 ISA raising 功能。

---

## 🔑 核心技術：Register Clobbing

這是整個專案最關鍵的技術，解決了「在組合語言中插入高階代碼而不破壞原始邏輯」的核心問題。

### 問題描述

當你在原始組合語言中間插入高階代碼時，LLVM 編譯器生成的新指令可能會**覆蓋原始邏輯正在使用的暫存器**，導致計算錯誤。

### 解決方案：四步驟 Register Clobbing

#### **步驟 1：暫存器保護**

```mlir
%reserved = llvm.inline_asm has_side_effects asm_dialect = att "", 
            "={v[0:31]}": () -> vector<32xi32>
```

**作用**：
- 告訴 LLVM：`v[0:31]` 這 32 個 VGPR 已經被「輸出」佔用
- `=` 表示輸出約束（output constraint）
- 編譯器在後續生成代碼時，**不會使用這些暫存器**

#### **步驟 2：暫存器綁定（數據橋接）**

```mlir
// 從原始邏輯使用的暫存器中讀取值
%val_A = llvm.inline_asm has_side_effects asm_dialect = att 
         "v_mov_b32 $0, v6", "=v": () -> f32
%val_B = llvm.inline_asm has_side_effects asm_dialect = att 
         "v_mov_b32 $0, v7", "=v": () -> f32
```

**作用**：
- 將硬體暫存器（v6, v7）的內容綁定到 MLIR 的 SSA 值（%val_A, %val_B）
- 讓 DSL 代碼可以訪問原始計算的數據

#### **步驟 3：DSL 代碼執行**

```mlir
%tid = gpu.thread_id x
%flag = arith.constant 3 : index
%is_positive = arith.cmpi eq, %tid, %flag : index
gpu.printf "TID = %d, FLAG = %d, is_positive = %d\n", %tid, %flag, %is_positive

cf.cond_br %is_positive, ^bbTRUE, ^bbMERGE
^bbTRUE:
    gpu.printf "A[%d] = %4.3f\n", %tid, %val_A : index, f32
    cf.br ^bbMERGE
^bbMERGE:
```

**重點**：
- 因為 `v[0:31]` 已被 clobber，編譯器會為 DSL 代碼分配 **v32 之後的暫存器**
- **完全不干擾原始邏輯**

#### **步驟 4：暫存器釋放**

```mlir
llvm.inline_asm has_side_effects asm_dialect = att "", 
                "{v[0:31]}" %reserved : (vector<32xi32>)-> ()
```

**作用**：
- 將 `%reserved` 作為輸入約束傳入
- 確保 DSL 代碼不會被優化掉
- 正確維護暫存器的生命週期

---

## 📊 實際驗證結果

### 測試案例：向量加法 C = A + B

**原始代碼**：
```c
// main.cpp 設定的數據
A = [50, 51, 52, 53]
B = [150, 152, 154, 156]
```

**預期結果**：
```
C = [200, 203, 206, 209]
```

### Register 分配驗證

通過比對 `vec_add_kernel-hip-amdgcn-amd-amdhsa-gfx950.s`（原始）和 `vec_add_kernel_after.s`（插入 DSL 後）：

#### 原始 Kernel
- **使用 VGPR**：v0-v7（共 8 個）
- **關鍵暫存器**：
  - v6: 儲存 A 值
  - v7: 儲存 B 值
  - v2: 儲存 C 值（計算結果）

#### 插入 DSL 後
- **原始邏輯區域**：v0-v31（被保護）
- **DSL 代碼區域**：v32+ (v32, v34-55...)
- **數據橋接**：
  ```asm
  v_mov_b32 v50, v6    # 將 A 值複製到 v50
  v_mov_b32 v51, v7    # 將 B 值複製到 v51
  ```

✅ **驗證成功**：原始邏輯的暫存器配置完全未受影響

---

## 🎯 DSL 功能展示

### 功能 1：條件分支與 Printf（TID == 3）

**MLIR 代碼**：
```mlir
%tid = gpu.thread_id x
%flag = arith.constant 3 : index
%is_positive = arith.cmpi eq, %tid, %flag : index
gpu.printf "TID = %d, FLAG = %d, condition = eq, is_positive = %d\n", 
           %tid, %flag, %is_positive

cf.cond_br %is_positive, ^bbPOSITIVE_1, ^bbMERGE_1
^bbPOSITIVE_1:
    gpu.printf "A[%d] printed inside kernel = %4.3f\n", %tid, %val_A
    gpu.printf "B[%d] printed inside kernel = %4.3f\n", %tid, %val_B
    cf.br ^bbMERGE_1
^bbMERGE_1:
```

**執行結果**：
```
TID = 0, FLAG = 3, condition = eq, is_positive = 0
TID = 1, FLAG = 3, condition = eq, is_positive = 0
TID = 2, FLAG = 3, condition = eq, is_positive = 0
TID = 3, FLAG = 3, condition = eq, is_positive = 1  ← 只有 TID=3 滿足條件
A[3] printed inside kernel = 53.000              ← 只有 TID=3 執行此分支
B[3] printed inside kernel = 156.000
```

✅ **驗證**：條件分支正確執行，只有 thread 3 進入 true 分支

---

### 功能 2：數學運算（B² - 4AC）

**MLIR 代碼**：
```mlir
%val_B2 = arith.mulf %val_B, %val_B : f32
%val_AC = arith.mulf %val_A, %val_C : f32
%c4 = arith.constant 4.0 : f32
%val_4AC = arith.mulf %c4, %val_AC : f32
%val_B2_4AC = arith.subf %val_B2, %val_4AC : f32
gpu.printf "B[%d]^2-4A[%d]C[%d] = %4.3f\n", 
           %tid_2, %tid_2, %tid_2, %val_B2_4AC
```

**執行結果**：
```
B[0]^2-4A[0]C[0] = -17500.000  ← 150² - 4×50×200 = -17500 ✓
B[1]^2-4A[1]C[1] = -18308.000  ← 152² - 4×51×203 = -18308 ✓
B[2]^2-4A[2]C[2] = -19132.000  ← 154² - 4×52×206 = -19132 ✓
B[3]^2-4A[3]C[3] = -19972.000  ← 156² - 4×53×209 = -19972 ✓
```

✅ **驗證**：複雜數學運算完全正確

---

### 功能 3：原始邏輯未受影響

**Host 輸出**：
```
Result: OK
C printed at host = 200.000  ← 50 + 150 = 200 ✓
C printed at host = 203.000  ← 51 + 152 = 203 ✓
C printed at host = 206.000  ← 52 + 154 = 206 ✓
C printed at host = 209.000  ← 53 + 156 = 209 ✓
```

✅ **驗證**：原始的 `C = A + B` 計算完全正確，沒有被 DSL 插入干擾

---

## 🛠️ 支援的 MLIR Dialects

### 1. GPU Dialect
- `gpu.thread_id`：取得 thread ID
- `gpu.printf`：在 kernel 內部打印除錯訊息
- `gpu.func`、`gpu.module`：定義 GPU 函數和模組
- `gpu.return`：從 GPU 函數返回

### 2. CF (Control Flow) Dialect
- `cf.cond_br`：條件分支
- `cf.br`：無條件跳轉
- 支援基本塊（basic block）：`^bbNAME`

### 3. Arith Dialect
- `arith.constant`：常數定義
- `arith.cmpi`：整數比較（eq, slt, sgt 等）
- `arith.mulf`：浮點乘法
- `arith.subf`：浮點減法
- `arith.addf`：浮點加法

### 4. LLVM Dialect
- `llvm.inline_asm`：內嵌組合語言
- `llvm.ptr`：指標類型
- 支援 AT&T 和 Intel 語法

---

## 🔄 編譯流程

### Device-Only Pipeline (ISA Generation)

```bash
mlir-opt kernel.mlir \
  --pass-pipeline='builtin.module(
    gpu-kernel-outlining,
    rocdl-attach-target{chip=gfx950},
    gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{index-bitwidth=32 runtime=HIP}),
    gpu-to-llvm,
    gpu-module-to-binary{format=isa}
  )'
```

**關鍵 Passes**：
1. `gpu-kernel-outlining`：提取 GPU kernel 為獨立模組
2. `rocdl-attach-target{chip=gfx950}`：指定目標 GPU 架構
3. `convert-scf-to-cf`：將結構化控制流轉為基本塊
4. `convert-gpu-to-rocdl`：GPU dialect → ROCDL dialect
5. `gpu-to-llvm`：ROCDL → LLVM dialect
6. `gpu-module-to-binary{format=isa}`：生成 AMDGPU ISA

### ISA to HSACO

```bash
# 組裝成目標檔
llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx950 -filetype=obj \
        kernel_isa.s -o kernel.o

# 連結成 HSA Code Object
ld.lld -shared kernel.o -o kernel.hsaco
```

---

## 📁 檔案結構

```
Track_A/
├── mdr.py                    # 主要腳本（PoC）
├── src.env                   # 環境設定（LLVM 路徑）
├── README.md                 # 使用說明
├── kernel_template.mlir      # 提取的 MLIR 模板（純粹的 DSL 範例）
└── e2e_test/                 # 端對端測試
    ├── main.cpp              # HIP 宿主程式
    ├── vec_add_kernel.hip    # 原始 HIP kernel
    ├── vec_add_kernel-*.s    # 編譯產生的 ISA（原始版本）
    ├── vec_add_kernel_after.s # 插入 DSL 後的 ISA
    ├── vec_add_kernel.hsaco  # 最終的 HSA Code Object
    ├── vec_add_module        # 編譯後的可執行檔
    ├── Makefile              # 建置腳本
    └── README.md             # 測試流程說明
```

---

## 💡 關鍵洞察

### 1. Register Clobbing 是核心
沒有這個機制，插入的 DSL 代碼會破壞原始邏輯。這是整個方法可行的基石。

### 2. 混合編程的可能性
證明了可以在同一個 kernel 中混合使用：
- **低階**：直接的組合語言指令（`llvm.inline_asm`）
- **高階**：抽象的 GPU/CF/Arith dialect

### 3. MLIR 的靈活性
MLIR 的多層次 IR 設計，讓這種混合編程成為可能。同一個函數中可以包含不同抽象層級的代碼。

### 4. Inline Assembly 的約束系統
LLVM 的 inline assembly 約束（constraints）系統非常強大：
- 輸出約束：`"={v[0:31]}"`
- 輸入約束：`"{v[0:31]}"`
- 輸入輸出約束：`"=v"`（分配新暫存器）

---

## ⚠️ 當前限制

### 1. 缺乏自動化 Raising
- 目前需要**手動**將 ISA 轉換為 `llvm.inline_asm`
- 需要**手動**分析哪些暫存器被使用
- 需要**手動**決定在哪裡插入 DSL

### 2. 硬編碼的 MLIR
- `mdr.py` 的 `generate_mlir_module()` 函數直接返回硬編碼的字串
- 不是真正的「讀取 ISA → 分析 → 插入」流程

### 3. 暫存器分析依賴人工
- 需要人工研究原始 ISA，找出使用的暫存器範圍
- Clobbing 範圍（v[0:31]）是根據觀察決定的

### 4. 插入點選擇依賴專業知識
- 需要理解 GPU ISA 的執行流程
- 需要知道在哪個點插入不會違反數據依賴

---

## 🚀 未來方向

### Phase 1：自動 ISA 分析
開發工具自動分析 AMDGPU ISA：
- 識別指令類型
- 追蹤暫存器使用
- 建立數據流圖

### Phase 2：自動 Raising
將 ISA 自動轉換為 MLIR：
- 生成 `llvm.inline_asm` 包裝
- 推斷 kernel 簽名（參數類型）
- 識別安全的插入點

### Phase 3：DSL 編譯器
開發高階 DSL 讓使用者指定想要插入的邏輯：
```python
@insert_at("after_load")
def debug_values(A, B):
    if thread_id() == 3:
        print(f"A = {A}, B = {B}")
```

### Phase 4：優化與安全性
- 最小化 register pressure
- 驗證插入不會破壞原始語義
- 支援更複雜的控制流

---

## 📚 參考資料

### LLVM/MLIR 文件
- [MLIR GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
- [MLIR Control Flow Dialect](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/)
- [LLVM Inline Assembly](https://llvm.org/docs/LangRef.html#inline-assembler-expressions)

### AMDGPU 架構
- [AMDGPU ISA Documentation](https://www.amd.com/en/support/gpu-isa-documentation)
- [ROCm Documentation](https://rocm.docs.amd.com/)

### 編譯工具
- [ROCm LLVM Project](https://github.com/rocm/llvm-project)
- Branch: `amd-staging`

---

## 🎓 學習建議

如果你是新接手這個專案的人，建議按以下順序學習：

1. **閱讀本文檔** 👈 你在這裡
2. **運行 e2e_test**：
   ```bash
   cd Track_A/e2e_test
   source ../src.env
   make
   ./vec_add_module
   ```
3. **研究 kernel_template.mlir**：理解 MLIR 結構
4. **對比 vec_add_kernel-*.s 和 vec_add_kernel_after.s**：觀察暫存器分配
5. **閱讀 mdr.py 原始碼**：理解編譯流程
6. **實驗修改 DSL**：嘗試加入自己的邏輯

---

## 📝 版本歷史

- **v1.0** (2025-12-23)：初始版本，記錄 PoC 階段的核心技術和驗證結果

---

## 👥 貢獻者

- 核心技術開發：[填入姓名]
- 技術文檔：AI Assistant（Claude）

---

**最後更新**：2025-12-23

