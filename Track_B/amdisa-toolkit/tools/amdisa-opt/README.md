# amdisa-opt - AMDISA Optimizer Driver

## 概述

`amdisa-opt` 是一個自定義的 MLIR 優化工具，類似於標準的 `mlir-opt`，但包含了 AMDISA dialect 和相關的 passes。

## 功能

- ✅ 支援所有標準 MLIR dialects
- ✅ 支援 AMDISA dialect
- ✅ 支援 AMDISA 的 lowering passes
- ✅ 可以通過命令行指定 pass pipeline
- ✅ 支援所有 mlir-opt 的標準功能

## 使用方法

### 基本語法

```bash
amdisa-opt [options] <input-file>
```

### 常用選項

```bash
# 顯示幫助
amdisa-opt --help

# 列出所有可用的 passes
amdisa-opt --help-list-hidden | grep -i amdisa

# 運行特定的 pass
amdisa-opt --pass-pipeline="builtin.module(amdisa-lower-to-gpu-inline-asm)" input.mlir

# 驗證 MLIR
amdisa-opt --verify-diagnostics input.mlir

# 輸出到文件
amdisa-opt --pass-pipeline="..." input.mlir -o output.mlir
```

## 使用範例

### 範例 1：查看 AMDISA dialect 的 MLIR

```bash
# 假設您有一個包含 AMDISA operations 的 MLIR 文件
amdisa-opt input.mlir
```

### 範例 2：運行 AMDISA lowering pass

```bash
# 將 AMDISA operations 降低為 GPU inline assembly
amdisa-opt --pass-pipeline="builtin.module(amdisa-lower-to-gpu-inline-asm)" input.mlir
```

### 範例 3：組合多個 passes

```bash
# 先運行 AMDISA lowering，再運行其他 GPU passes
amdisa-opt \
  --pass-pipeline="builtin.module(amdisa-lower-to-gpu-inline-asm,gpu-kernel-outlining)" \
  input.mlir
```

### 範例 4：完整的 pipeline 範例

```bash
# 1. 使用 amdisa-translate 將 .s 轉換為 MLIR
amdisa-translate -x s -emit=mlir kernel.s -o kernel.mlir

# 2. 使用 amdisa-opt 進行優化和轉換
amdisa-opt --pass-pipeline="builtin.module(amdisa-lower-to-gpu-inline-asm)" \
  kernel.mlir -o kernel_gpu.mlir

# 3. 繼續使用其他 MLIR 工具處理
mlir-translate --mlir-to-llvmir kernel_gpu.mlir -o kernel.ll
```

## 與 amdisa-translate 的區別

| 特性 | amdisa-translate | amdisa-opt |
|------|-----------------|------------|
| **主要用途** | 將 AMD ISA (.s) 轉換為 MLIR | 對 MLIR 進行優化和轉換 |
| **輸入格式** | .s (assembly) 或 .mlir | .mlir |
| **輸出格式** | .mlir | .mlir |
| **Pass 支援** | 固定的 pipeline | 靈活的 pass pipeline |
| **命令行控制** | 簡單（-emit=mlir/gpu） | 完整（任意 pass 組合）|
| **使用場景** | 初始轉換 | 調試、優化、實驗 |

## Pass 說明

### amdisa-lower-to-gpu-inline-asm

將 AMDISA dialect 的 operations 降低為 GPU dialect 的 inline assembly。

**輸入：**
```mlir
module {
  amdisa.label {name = "kernel"}
  amdisa.inst {mnemonic = "v_add_f32", operands = ["v0", "v1", "v2"]}
}
```

**輸出：**
```mlir
module {
  gpu.module @amdisa_module {
    gpu.func @amdisa_kernel(...) {
      %0 = llvm.inline_asm "v_add_f32 v0, v1, v2", ...
      gpu.return
    }
  }
}
```

**使用：**
```bash
amdisa-opt --amdisa-lower-to-gpu-inline-asm input.mlir
# 或
amdisa-opt --pass-pipeline="builtin.module(amdisa-lower-to-gpu-inline-asm)" input.mlir
```

## 調試技巧

### 1. 顯示詳細的 pass 信息

```bash
amdisa-opt --mlir-print-ir-before-all --mlir-print-ir-after-all \
  --pass-pipeline="..." input.mlir
```

### 2. 只運行到某個 pass

```bash
# 使用 pass pipeline 語法精確控制
amdisa-opt --pass-pipeline="builtin.module(pass1,pass2)" input.mlir
```

### 3. 顯示 pass 統計信息

```bash
amdisa-opt --mlir-pass-statistics \
  --pass-pipeline="..." input.mlir
```

### 4. 驗證每個 pass 之後的 IR

```bash
amdisa-opt --verify-each \
  --pass-pipeline="..." input.mlir
```

## 與標準 MLIR 工具鏈整合

```bash
# 完整的工作流程範例

# 1. AMD ISA → AMDISA MLIR
amdisa-translate -x s -emit=mlir kernel.s -o step1_amdisa.mlir

# 2. AMDISA MLIR → GPU MLIR (使用 amdisa-opt)
amdisa-opt --amdisa-lower-to-gpu-inline-asm step1_amdisa.mlir -o step2_gpu.mlir

# 3. GPU MLIR → LLVM IR (使用標準 MLIR 工具)
mlir-opt --gpu-kernel-outlining --convert-gpu-to-llvm step2_gpu.mlir -o step3_llvm_dialect.mlir

# 4. LLVM Dialect → LLVM IR
mlir-translate --mlir-to-llvmir step3_llvm_dialect.mlir -o step4.ll

# 5. LLVM IR → Object file
llc -march=amdgcn -mcpu=gfx90a step4.ll -o kernel.o
```

## 常見錯誤

### 錯誤 1: 找不到 AMDISA dialect

```
error: Dialect `amdisa` not found
```

**原因：** 使用了標準的 `mlir-opt` 而非 `amdisa-opt`。

**解決：** 使用 `amdisa-opt` 而非 `mlir-opt`。

### 錯誤 2: 找不到 pass

```
error: 'amdisa-lower-to-gpu-inline-asm' does not refer to a registered pass
```

**原因：** Pass 名稱錯誤或 pass 未正確註冊。

**解決：** 
- 檢查 pass 名稱拼寫
- 使用 `amdisa-opt --help-list-hidden` 查看所有可用的 passes

### 錯誤 3: 依賴的 dialect 未加載

```
error: Dialect `gpu` not found
```

**原因：** AMDISA lowering pass 需要 GPU dialect，但未加載。

**解決：** `amdisa-opt` 已自動加載所有標準 dialects，這個錯誤不應該出現。如果出現，請檢查 CMake 配置。

## 進階用法

### 自定義 Pass Pipeline

創建一個 pass pipeline 文件：

```mlir
// pipeline.txt
builtin.module(
  amdisa-lower-to-gpu-inline-asm,
  gpu-kernel-outlining,
  convert-gpu-to-nvvm
)
```

使用：
```bash
amdisa-opt --pass-pipeline-file=pipeline.txt input.mlir
```

### 與 Python Bindings 配合使用

```python
import mlir.ir as ir
import subprocess

# 使用 amdisa-opt 處理 MLIR
def run_amdisa_opt(mlir_code, passes):
    with open('temp.mlir', 'w') as f:
        f.write(mlir_code)
    
    result = subprocess.run(
        ['amdisa-opt', f'--pass-pipeline={passes}', 'temp.mlir'],
        capture_output=True, text=True
    )
    
    return result.stdout
```

## 性能考量

- `amdisa-opt` 會加載所有標準 MLIR dialects，啟動時間可能較長（通常 < 1 秒）
- 對於批量處理，建議使用 pipeline 模式而非多次調用
- 大型 MLIR 文件的處理時間與 pass 複雜度成正比

## 相關工具

- **amdisa-translate**: AMD ISA assembly 到 MLIR 的轉換工具
- **mlir-opt**: 標準 MLIR 優化工具（不包含 AMDISA）
- **mlir-translate**: MLIR 到其他格式的轉換工具
- **pipeline.py**: 端到端的 AMD ISA 到 HSACO 的工具鏈

## 貢獻與擴展

要添加新的 AMDISA passes：

1. 在 `include/Dialect/AMDISA/Passes.td` 中定義 pass
2. 在 `lib/Dialect/AMDISA/Transforms/` 中實現 pass
3. 重新構建 `amdisa-opt`
4. Pass 會自動註冊並可通過命令行使用

## 授權

本工具是 AMDISA Toolkit 的一部分，遵循與主專案相同的授權協議。

