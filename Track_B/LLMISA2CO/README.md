# LLMISA2CO - LLM ISA to Code Object Pipeline

這個專案提供了一套完整的工具鏈，用於處理 LLM 生成的 AMD GPU ISA（Instruction Set Architecture），並驗證其正確性。

## 📁 目錄結構

```
Track_B/
├── LLMISA2CO/                   # 本專案目錄
│   ├── script/                  # 所有腳本工具
│   │   ├── wrap_isa.py         # ISA 包裝工具（Python）
│   │   ├── pipeline.py         # ISA 到 HSACO 的 Pipeline 工具
│   │   ├── batch_wrap_isa.sh   # 批量 ISA 包裝腳本
│   │   ├── test_wrapped_isa_pipeline.sh  # 完整測試 pipeline
│   │   ├── relink_and_compare.sh         # 重新連結與比較工具
│   │   └── clean_results.sh    # 清理測試結果工具
│   ├── kernels/                 # Kernel 測試案例目錄
│   │   └── [kernel_hash]/      # 各個 kernel 的資料夾
│   └── README.md                # 本文檔
│
└── amdisa-toolkit/              # AMDISA 工具集（依賴）
    └── build/bin/               # 構建後的工具
        └── amdisa-translate    # ISA 轉換工具

wrapped_isa_test_results_*/      # 測試結果（在 Track_B/ 層級）
```

## 🎯 工作流程概覽

```
原始 ISA (.s)  →  LLM 生成指令 (_func.s)  →  包裝 (_wrapped.s)  
    ↓
Pipeline 處理 (.hsaco)  →  重新連結  →  執行測試  →  結果比較
```

## 🚀 快速開始

### 方式 A：批量測試（推薦用於多個 kernel）

```bash
# 1. 環境設置（首次使用）
cd Project-MDR/Track_B/LLMISA2CO
# 請先完成「環境設置」章節的步驟

# 2. 包裝所有 ISA
./script/batch_wrap_isa.sh -p kernels/

# 3. 運行測試（測試前 5 個 kernel）
# 注意：根據您的 GPU 型號選擇正確的晶片參數
# MI250/MI250X 使用: -c gfx90a
# MI300A/MI300X 使用: -c gfx942
# 預設為 gfx950
./script/test_wrapped_isa_pipeline.sh -p kernels/ -l 5 -c gfx90a

# 4. 查看結果
cat ../wrapped_isa_test_results_*/test_summary.txt

# 5. 清理測試輸出
./script/clean_results.sh --type all -p kernels/
```

### 方式 B：單個 Kernel 快速生成 HSACO（推薦用於調試）

```bash
# 1. 環境設置（首次使用）
cd Project-MDR/Track_B/LLMISA2CO

# 2. 包裝單個 ISA（使用實際 kernel 範例）
python3 script/wrap_isa.py \
  -o kernels/0a32840dac54a492de687e418c588476d324dcdc/0a32840dac54a492de687e418c588476d324dcdc-hip-amdgcn-amd-amdhsa-gfx950.s \
  -n kernels/0a32840dac54a492de687e418c588476d324dcdc/0a32840dac54a492de687e418c588476d324dcdc_func.s \
  -O kernels/0a32840dac54a492de687e418c588476d324dcdc/0a32840dac54a492de687e418c588476d324dcdc_wrapped.s

# 3. 生成 HSACO
python3 script/pipeline.py \
  --chip gfx950 \
  --workdir kernels/0a32840dac54a492de687e418c588476d324dcdc/output \
  --original-isa kernels/0a32840dac54a492de687e418c588476d324dcdc/0a32840dac54a492de687e418c588476d324dcdc-hip-amdgcn-amd-amdhsa-gfx950.s \
  kernels/0a32840dac54a492de687e418c588476d324dcdc/0a32840dac54a492de687e418c588476d324dcdc_wrapped.s

# 4. HSACO 已生成在：
# kernels/0a32840dac54a492de687e418c588476d324dcdc/output/0a32840dac54a492de687e418c588476d324dcdc_wrapped_rebuilt.hsaco

# 5. （可選）測試生成的 HSACO
./script/relink_and_compare.sh \
  -p kernels/0a32840dac54a492de687e418c588476d324dcdc/0a32840dac54a492de687e418c588476d324dcdc \
  -H kernels/0a32840dac54a492de687e418c588476d324dcdc/0a32840dac54a492de687e418c588476d324dcdc-host-x86_64-unknown-linux-gnu.o \
  -G kernels/0a32840dac54a492de687e418c588476d324dcdc/output/0a32840dac54a492de687e418c588476d324dcdc_wrapped_rebuilt.hsaco
```

**針對您的 kernel 調整路徑**：
- 將 `0a32840dac54a492de687e418c588476d324dcdc` 替換為您的 kernel hash
- 將 `gfx950` 替換為您的 GPU 型號（MI250 使用 `gfx90a`，MI300 使用 `gfx942`）

## 環境設置

### 步驟 1：準備 LLVM/MLIR

您需要已構建或安裝的 LLVM/MLIR。如果還沒有：

**選項 A：使用系統安裝的 LLVM**
```bash
# 如果系統已安裝 LLVM，跳到步驟 2
# 在某些系統上可以使用套件管理器安裝：
sudo apt install llvm mlir-tools  # Ubuntu/Debian
```

**選項 B：從源碼構建 LLVM**
```bash
# 克隆 LLVM 專案
git clone --depth 1 https://github.com/llvm/llvm-project.git

cd llvm-project

# 配置構建（建議安裝到自訂位置）
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-install

# 構建並安裝（這會需要一些時間）
ninja -C build install

# 將 LLVM 工具加入 PATH
export PATH=$HOME/llvm-install/bin:$PATH
```

### 步驟 2：構建 AMDISA Toolkit

```bash
# 進入 amdisa-toolkit 目錄（與 LLMISA2CO 同級）
cd ../amdisa-toolkit

# 方法 1：使用 cmake.sh 腳本（自動檢測 LLVM）
./cmake.sh

# 方法 2：手動配置（指定 LLVM 路徑）
cmake -B build -G Ninja \
  -DMLIR_DIR=$HOME/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=$HOME/llvm-install/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release

# 構建
ninja -C build

# 將 amdisa-translate 加入 PATH
export PATH=$(pwd)/build/bin:$PATH
```

### 步驟 3：安裝 HIP 工具鏈

```bash
# ROCm/HIP 安裝
# 詳細安裝說明請參考：https://rocm.docs.amd.com/

# 將 ROCm 工具加入 PATH
export PATH=/opt/rocm/bin:$PATH
```

### 步驟 4：驗證安裝

```bash
# 檢查所有必需工具是否可用
which python3
which amdisa-translate
which mlir-opt
which llvm-mc
which lld
which hipcc
which clang-offload-bundler

```
---


## 🔧 工具說明

### 1. wrap_isa.py - ISA 包裝工具

**功能**：將 LLM 生成的純指令 ISA 包裝成完整的 AMD GPU ISA 格式。

**使用方式**：
```bash
python3 script/wrap_isa.py -o <原始ISA> -n <新指令ISA> -O <輸出檔案>
```

**參數**：
- `-o, --old-isa`：原始的完整 ISA 文件（作為模板）
- `-n, --new-isa`：LLM 生成的純指令 ISA
- `-O, --output`：輸出的包裝後 ISA 文件

**範例**：
```bash
python3 script/wrap_isa.py \
  -o kernels/abc123/abc123-hip-amdgcn-amd-amdhsa-gfx950.s \
  -n kernels/abc123/abc123_func.s \
  -O kernels/abc123/abc123_wrapped.s
```

---

### 2. pipeline.py - ISA 到 HSACO 轉換工具

**功能**：將 wrapped ISA 文件轉換為可執行的 HSACO（Heterogeneous System Architecture Code Object）文件。

**使用方式**：
```bash
python3 script/pipeline.py [選項] <wrapped_isa.s>
```

**參數**：
- `<wrapped_isa.s>`：輸入的 wrapped ISA 文件（必需）
- `--chip CHIP`：指定 GPU 晶片型號（預設：gfx950）
- `--workdir DIR`：工作目錄，存放中間文件（預設：pipeline_output）
- `--emit-isa`：生成 ISA 和 HSACO（預設啟用）
- `--no-emit-isa`：停用 ISA/HSACO 生成
- `--emit-llvm-ir`：同時生成 LLVM IR（用於調試）
- `--output-prefix PREFIX`：輸出文件前綴
- `--original-isa FILE`：原始完整 ISA（用於提取 metadata）

**範例**：

```bash
# 基本使用（生成 HSACO）- 使用實際 kernel
python3 script/pipeline.py \
  --chip gfx950 \
  --workdir kernels/0a32840dac54a492de687e418c588476d324dcdc/pipeline_output \
  --original-isa kernels/0a32840dac54a492de687e418c588476d324dcdc/0a32840dac54a492de687e418c588476d324dcdc-hip-amdgcn-amd-amdhsa-gfx950.s \
  kernels/0a32840dac54a492de687e418c588476d324dcdc/0a32840dac54a492de687e418c588476d324dcdc_wrapped.s

# 針對 MI250 使用（假設有 gfx90a 的 ISA）
python3 script/pipeline.py \
  --chip gfx90a \
  --workdir output \
  --original-isa kernels/{kernel_hash}/{kernel_hash}-hip-amdgcn-amd-amdhsa-gfx90a.s \
  kernels/{kernel_hash}/{kernel_hash}_wrapped.s
```

**輸出**：
- `{workdir}/{prefix}.amdisamlir` - AMDISA MLIR 表示
- `{workdir}/{prefix}.gpumlir` - GPU MLIR 表示
- `{workdir}/{prefix}_rebuilt.s` - 重建的 ISA
- `{workdir}/{prefix}_rebuilt.o` - 目標檔案
- `{workdir}/{prefix}_rebuilt.hsaco` - 最終的 HSACO 文件 ⭐
- `{workdir}/{prefix}_llvm.ll` - LLVM IR（如果使用 --emit-llvm-ir）

**重要提示**：
- 必須提供 `--original-isa` 參數以正確提取 metadata
- `--chip` 必須與原始 ISA 文件名中的晶片型號匹配
- HSACO 文件是最終可用於 GPU 執行的代碼對象

---

### 3. batch_wrap_isa.sh - 批量包裝工具

**功能**：遞迴搜尋指定目錄，自動為所有 kernel 生成包裝後的 ISA。

**使用方式**：
```bash
./script/batch_wrap_isa.sh -p <目錄路徑> [選項]
```

**參數**：
- `-p PATH`：指定要處理的目錄（必需）
- `-k, --kernel HASH`：只處理指定的 kernel
- `-h, --help`：顯示幫助信息

**範例**：
```bash
# 處理所有 kernel
./script/batch_wrap_isa.sh -p kernels/

# 只處理特定 kernel
./script/batch_wrap_isa.sh -p kernels/ -k abc123def456...
```

**輸出**：
- 成功：在各 kernel 目錄生成 `*_wrapped.s` 文件
- 失敗：將失敗的 kernel 複製到 `fail_wrap_TIMESTAMP/` 目錄

---

### 4. test_wrapped_isa_pipeline.sh - 完整測試 Pipeline

**功能**：自動化測試 wrapped ISA 的完整流程，包括：
1. 使用 `pipeline.py` 生成 HSACO
2. 使用 `relink_and_compare.sh` 重新連結
3. 執行並比較結果

**前置需求**：
- 需要先完成環境設置（參見「環境設置」章節）
- 內部會自動調用 `script/pipeline.py`
- 確保 `amdisa-translate` 和相關 LLVM 工具在 PATH 中

**使用方式**：
```bash
./script/test_wrapped_isa_pipeline.sh -p <目錄> [選項]
```

**參數**：
- `-p PATH`：指定 kernels 目錄（必需）
- `-c, --chip CHIP`：指定 GPU 晶片型號（預設：gfx950）
  - **重要**：必須與您的硬體和 ISA 文件匹配
  - 常見型號：
    - `gfx90a` - MI200 系列（MI210, MI250, MI250X）
    - `gfx942` - MI300 系列（MI300A, MI300X）
    - `gfx950` - 下一代 GPU（預設值）
    - `gfx908` - MI100
- `-t, --timeout SECONDS`：執行超時時間（預設：60 秒）
- `-k, --kernel HASH`：只測試指定的 kernel
- `-l, --limit N`：只測試前 N 個 kernel
- `-s, --skip-pipeline`：跳過 pipeline.py，使用已存在的 .hsaco
- `-h, --help`：顯示幫助信息

**範例**：
```bash
# 測試所有 kernel（使用預設晶片 gfx950）
./script/test_wrapped_isa_pipeline.sh -p kernels/

# 針對 MI250/MI250X 測試（使用 gfx90a）
./script/test_wrapped_isa_pipeline.sh -p kernels/ -c gfx90a

# 針對 MI300A/MI300X 測試（使用 gfx942）
./script/test_wrapped_isa_pipeline.sh -p kernels/ -c gfx942

# 只測試前 5 個 kernel
./script/test_wrapped_isa_pipeline.sh -p kernels/ -l 5

# 測試特定 kernel
./script/test_wrapped_isa_pipeline.sh -p kernels/ -k abc123def456...

# 設定超時時間為 120 秒
./script/test_wrapped_isa_pipeline.sh -p kernels/ -t 120

# 跳過 pipeline，使用已存在的 .hsaco
./script/test_wrapped_isa_pipeline.sh -p kernels/ -s
```

**輸出**：
- 測試結果目錄：`../wrapped_isa_test_results_TIMESTAMP/`
- 測試總結：`test_summary.txt`
- 各 kernel 日誌：`{kernel_hash}_pipeline.log`、`{kernel_hash}_relink.log`

**測試結果狀態**：
- ✅ `PASS`：完全一致
- ⚠️ `PARTIAL`：退出碼相同但輸出有差異
- ❌ `FAIL`：測試失敗
- ⏱ `TIMEOUT`：執行超時
- ⊘ `SKIP`：跳過（缺少必要文件）

---

### 5. relink_and_compare.sh - 重新連結與比較工具

**功能**：將重建的 HSACO 與原始執行檔連結，並比較執行結果。

**使用方式**：
```bash
./script/relink_and_compare.sh -p <執行檔> -H <host.o> -G <hsaco> [-a "參數"]
```

**參數**：
- `-p PATH`：原始執行檔路徑（必需）
- `-H PATH`：Host 目標檔案（host-x86_64-unknown-linux-gnu.o）（必需）
- `-G PATH`：重建的 HSACO 文件（必需）
- `-a ARGS`：測試參數（可選，用引號包起來）
- `-h, --help`：顯示幫助信息

**環境變數**：
- `EXECUTION_TIMEOUT`：執行超時時間（秒），預設 60
- `OUTPUT_BASE_DIR`：輸出目錄基礎路徑

**範例**：
```bash
# 基本使用
./script/relink_and_compare.sh \
  -p kernels/abc123/compiled_output_*/executable \
  -H kernels/abc123/offload_output_*/host-x86_64-unknown-linux-gnu.o \
  -G kernels/abc123/wrapped_output/abc123_wrapped_rebuilt.hsaco

# 帶測試參數
./script/relink_and_compare.sh \
  -p kernels/abc123/executable \
  -H kernels/abc123/host.o \
  -G kernels/abc123/rebuilt.hsaco \
  -a "64 128"

# 自訂超時時間
EXECUTION_TIMEOUT=120 ./script/relink_and_compare.sh \
  -p kernels/abc123/executable \
  -H kernels/abc123/host.o \
  -G kernels/abc123/rebuilt.hsaco
```

**輸出**：
- 輸出目錄：`{kernel_dir}/relink_test_results_TIMESTAMP/`
- 包含：原始輸出、重連結輸出、差異比較等

**退出碼**：
- `0`：完全一致
- `1`：部分通過（輸出有差異）
- `2`：測試失敗
- `3`：執行超時

---

### 6. clean_results.sh - 清理工具

**功能**：清理測試過程中產生的各種輸出目錄。

**使用方式**：
```bash
./script/clean_results.sh --type <類型> [-p <路徑>] [選項]
```

**參數**：
- `--type TYPE`：清除類型（必需）
  - `pipeline`：清除 pipeline_test* 輸出
  - `compile`：清除 compiled_output* 輸出
  - `relink`：清除 relink_test_results* 輸出
  - `wrapped`：清除 wrapped* 輸出
  - `failed`：清除 failed_kernels* 資料夾
  - `all`：清除以上所有輸出
- `-p PATH`：指定目標目錄（預設：系統預設目錄）
- `--yes`：自動確認（預設）
- `--no`：詢問確認
- `--dry-run`：只列出不刪除
- `--help`：顯示幫助信息

**範例**：
```bash
# 清除 pipeline 輸出
./script/clean_results.sh --type pipeline

# 清除特定目錄的 relink 輸出
./script/clean_results.sh --type relink -p kernels/

# 清除所有輸出（預覽）
./script/clean_results.sh --type all --dry-run

# 清除所有輸出（需確認）
./script/clean_results.sh --type all --no

# 清除 wrapped 文件
./script/clean_results.sh --type wrapped -p kernels/
```

---

## 🛠 依賴工具

- **Python 3**：用於 `wrap_isa.py`
- **Bash**：用於所有 shell 腳本
- **LLVM/MLIR 工具鏈**：
  - `mlir-opt`：MLIR 優化工具
  - `llvm-mc`：LLVM 機器碼組裝器
  - `lld`：LLVM 鏈接器
  - `llvm-dis`：LLVM bitcode 反組譯器（可選）
- **AMDISA Toolkit**：
  - `amdisa-translate`：AMD ISA 到 MLIR 的轉換工具
- **HIP 工具鏈**：
  - `hipcc`：HIP 編譯器
  - `clang-offload-bundler`：Offload bundler 工具
