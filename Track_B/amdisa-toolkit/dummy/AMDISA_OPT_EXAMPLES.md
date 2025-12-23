# amdisa-opt 使用範例

## 📋 基本概念

`amdisa-opt` 的使用方式與標準的 `mlir-opt` **完全相同**，只是額外支援了：
- ✅ AMDISA dialect
- ✅ AMDISA passes（如 `amdisa-lower-to-gpu-inline-asm`）
- ✅ 所有標準 MLIR dialects 和 passes

---

## 🚀 基本使用

### 1. 顯示幫助

```bash
# 基本幫助
./build/bin/amdisa-opt --help

# 列出所有可用的 passes
./build/bin/amdisa-opt --help-list-hidden | less

# 搜尋 AMDISA 相關的 passes
./build/bin/amdisa-opt --help-list-hidden | grep -i amdisa
```

### 2. 查看 MLIR 文件（不做任何轉換）

```bash
# 讀取並輸出 MLIR 文件（驗證語法）
./build/bin/amdisa-opt input.mlir

# 輸出到文件
./build/bin/amdisa-opt input.mlir -o output.mlir
```

---

## 🎯 完整工作流程範例

### 步驟 1：從 AMD ISA 生成 AMDISA MLIR

首先使用 `amdisa-translate` 將 `.s` 文件轉換為 MLIR：

```bash
# 假設您有一個 AMD ISA assembly 文件
cat > kernel.s << 'EOF'
.text
.globl vector_add
vector_add:
    s_load_dwordx4 s[0:3], s[0:1], 0x0
    s_waitcnt lgkmcnt(0)
    v_add_f32 v0, v1, v2
    s_endpgm
EOF

# 轉換為 AMDISA MLIR
./build/bin/amdisa-translate -x s -emit=mlir kernel.s -o kernel_amdisa.mlir

# 查看生成的 AMDISA MLIR
cat kernel_amdisa.mlir
```

**輸出類似：**
```mlir
module {
  amdisa.label {name = "vector_add"}
  amdisa.inst {mnemonic = "s_load_dwordx4", operands = ["s[0:3]", "s[0:1]", "0x0"], raw_text = "s_load_dwordx4 s[0:3], s[0:1], 0x0"}
  amdisa.inst {mnemonic = "s_waitcnt", operands = ["lgkmcnt(0)"], raw_text = "s_waitcnt lgkmcnt(0)"}
  amdisa.inst {mnemonic = "v_add_f32", operands = ["v0", "v1", "v2"], raw_text = "v_add_f32 v0, v1, v2"}
  amdisa.inst {mnemonic = "s_endpgm", operands = [], raw_text = "s_endpgm"}
}
```

### 步驟 2：使用 amdisa-opt 驗證和檢查

```bash
# 驗證 MLIR 語法
./build/bin/amdisa-opt kernel_amdisa.mlir

# 使用 --verify-diagnostics 進行嚴格檢查
./build/bin/amdisa-opt --verify-diagnostics kernel_amdisa.mlir
```

### 步驟 3：運行 AMDISA lowering pass

```bash
# 將 AMDISA operations 降低為 GPU inline assembly
./build/bin/amdisa-opt \
  --amdisa-lower-to-gpu-inline-asm \
  kernel_amdisa.mlir \
  -o kernel_gpu.mlir

# 查看結果
cat kernel_gpu.mlir
```

**輸出類似：**
```mlir
module {
  gpu.module @amdisa_kernels {
    gpu.func @vector_add(%arg0: ..., %arg1: ...) kernel {
      %0 = llvm.inline_asm "
        s_load_dwordx4 s[0:3], s[0:1], 0x0
        s_waitcnt lgkmcnt(0)
        v_add_f32 v0, v1, v2
        s_endpgm
      ", ... : () -> ()
      gpu.return
    }
  }
}
```

### 步驟 4：使用 Pass Pipeline 語法

```bash
# 更精確的 pass 控制
./build/bin/amdisa-opt \
  --pass-pipeline="builtin.module(amdisa-lower-to-gpu-inline-asm)" \
  kernel_amdisa.mlir
```

---

## 🔧 進階使用範例

### 範例 1：組合多個 Passes

```bash
# AMDISA → GPU → 其他 GPU passes
./build/bin/amdisa-opt \
  --pass-pipeline="builtin.module(
    amdisa-lower-to-gpu-inline-asm,
    gpu-kernel-outlining
  )" \
  kernel_amdisa.mlir \
  -o kernel_outlined.mlir
```

### 範例 2：顯示每個 Pass 前後的 IR

```bash
# 用於調試：顯示 pass 執行前後的 IR
./build/bin/amdisa-opt \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all \
  --amdisa-lower-to-gpu-inline-asm \
  kernel_amdisa.mlir \
  2>&1 | less
```

### 範例 3：顯示 Pass 統計信息

```bash
# 顯示 pass 執行的統計數據
./build/bin/amdisa-opt \
  --mlir-pass-statistics \
  --amdisa-lower-to-gpu-inline-asm \
  kernel_amdisa.mlir
```

### 範例 4：只顯示修改後的 IR

```bash
# 只在 IR 被修改時才打印
./build/bin/amdisa-opt \
  --mlir-print-ir-after-change \
  --amdisa-lower-to-gpu-inline-asm \
  kernel_amdisa.mlir
```

### 範例 5：驗證每個 Pass 之後的 IR

```bash
# 確保每個 pass 後 IR 都是有效的
./build/bin/amdisa-opt \
  --verify-each \
  --amdisa-lower-to-gpu-inline-asm \
  kernel_amdisa.mlir
```

---

## 📝 使用 Pipeline 文件

### 創建 Pipeline 配置文件

```bash
cat > amdisa_pipeline.txt << 'EOF'
builtin.module(
  amdisa-lower-to-gpu-inline-asm,
  gpu-kernel-outlining,
  inline,
  cse
)
EOF
```

### 使用 Pipeline 文件

```bash
./build/bin/amdisa-opt \
  --pass-pipeline-file=amdisa_pipeline.txt \
  kernel_amdisa.mlir
```

---

## 🎓 與標準 MLIR Passes 結合

### 範例 1：AMDISA + 標準優化

```bash
# AMDISA lowering + 標準優化 passes
./build/bin/amdisa-opt \
  --pass-pipeline="builtin.module(
    amdisa-lower-to-gpu-inline-asm,
    canonicalize,
    cse,
    symbol-dce
  )" \
  kernel_amdisa.mlir
```

### 範例 2：與其他 Dialects 互操作

```bash
# 如果您的 MLIR 中還有其他 dialects
./build/bin/amdisa-opt \
  --pass-pipeline="builtin.module(
    amdisa-lower-to-gpu-inline-asm,
    gpu-kernel-outlining,
    convert-gpu-to-llvm
  )" \
  mixed.mlir
```

---

## 🔍 調試技巧

### 1. 檢查 Pass 是否註冊

```bash
# 列出所有 passes 並搜尋 AMDISA
./build/bin/amdisa-opt --help-list-hidden | grep -A5 amdisa

# 預期看到：
# --amdisa-lower-to-gpu-inline-asm
#     Lower AMDISA ops into gpu.module + gpu.func + llvm.inline_asm
```

### 2. 調試 Pass 失敗

```bash
# 顯示詳細的診斷信息
./build/bin/amdisa-opt \
  --mlir-print-debuginfo \
  --mlir-print-op-generic \
  --amdisa-lower-to-gpu-inline-asm \
  kernel_amdisa.mlir
```

### 3. 只運行到特定步驟

```bash
# 使用 pipeline 控制執行順序
./build/bin/amdisa-opt \
  --pass-pipeline="builtin.module(
    amdisa-lower-to-gpu-inline-asm
  )" \
  kernel_amdisa.mlir

# 然後檢查輸出，決定是否繼續
```

---

## 📊 實際工作流程範例

### 完整的 AMD ISA → LLVM IR 流程

```bash
#!/bin/bash
# 完整的轉換流程

INPUT="kernel.s"
OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR

# 步驟 1: AMD ISA → AMDISA MLIR
echo "Step 1: AMD ISA → AMDISA MLIR"
./build/bin/amdisa-translate -x s -emit=mlir $INPUT \
  -o $OUTPUT_DIR/step1_amdisa.mlir

# 步驟 2: 驗證 AMDISA MLIR
echo "Step 2: Verify AMDISA MLIR"
./build/bin/amdisa-opt --verify-diagnostics $OUTPUT_DIR/step1_amdisa.mlir

# 步驟 3: AMDISA → GPU Inline ASM
echo "Step 3: AMDISA → GPU Inline ASM"
./build/bin/amdisa-opt \
  --amdisa-lower-to-gpu-inline-asm \
  $OUTPUT_DIR/step1_amdisa.mlir \
  -o $OUTPUT_DIR/step2_gpu.mlir

# 步驟 4: GPU Kernel Outlining
echo "Step 4: GPU Kernel Outlining"
mlir-opt --gpu-kernel-outlining \
  $OUTPUT_DIR/step2_gpu.mlir \
  -o $OUTPUT_DIR/step3_outlined.mlir

# 步驟 5: 轉換為 LLVM Dialect
echo "Step 5: Convert to LLVM Dialect"
mlir-opt --convert-gpu-to-llvm \
  $OUTPUT_DIR/step3_outlined.mlir \
  -o $OUTPUT_DIR/step4_llvm.mlir

# 步驟 6: 生成 LLVM IR
echo "Step 6: Generate LLVM IR"
mlir-translate --mlir-to-llvmir \
  $OUTPUT_DIR/step4_llvm.mlir \
  -o $OUTPUT_DIR/step5.ll

echo "Done! Output in $OUTPUT_DIR/"
```

---

## 🎯 常用命令模板

### 模板 1：快速驗證

```bash
./build/bin/amdisa-opt input.mlir
```

### 模板 2：單一 Pass

```bash
./build/bin/amdisa-opt --<pass-name> input.mlir -o output.mlir
```

### 模板 3：多個 Passes（簡單）

```bash
./build/bin/amdisa-opt \
  --pass1 \
  --pass2 \
  --pass3 \
  input.mlir -o output.mlir
```

### 模板 4：多個 Passes（Pipeline 語法）

```bash
./build/bin/amdisa-opt \
  --pass-pipeline="builtin.module(pass1,pass2,pass3)" \
  input.mlir -o output.mlir
```

### 模板 5：調試模式

```bash
./build/bin/amdisa-opt \
  --mlir-print-ir-before-all \
  --mlir-print-ir-after-all \
  --mlir-pass-statistics \
  --verify-each \
  --<pass-name> \
  input.mlir 2>&1 | tee debug.log
```

---

## 📋 與 mlir-opt 的對比

### mlir-opt（標準）

```bash
# mlir-opt 不認識 AMDISA dialect
mlir-opt kernel_amdisa.mlir
# ❌ 錯誤：Dialect `amdisa` not found
```

### amdisa-opt（您的工具）

```bash
# amdisa-opt 認識 AMDISA dialect
./build/bin/amdisa-opt kernel_amdisa.mlir
# ✅ 成功！
```

### 功能對比

| 功能 | mlir-opt | amdisa-opt |
|------|----------|-----------|
| 標準 MLIR dialects | ✅ | ✅ |
| 標準 MLIR passes | ✅ | ✅ |
| AMDISA dialect | ❌ | ✅ |
| AMDISA passes | ❌ | ✅ |
| 命令行接口 | 完全相同 | 完全相同 |

---

## 🔗 與其他工具配合

### 與 amdisa-translate 配合

```bash
# 1. amdisa-translate: .s → .mlir
./build/bin/amdisa-translate -x s -emit=mlir input.s -o amdisa.mlir

# 2. amdisa-opt: 優化和轉換
./build/bin/amdisa-opt --amdisa-lower-to-gpu-inline-asm amdisa.mlir -o gpu.mlir

# 3. mlir-opt: 標準 MLIR 優化
mlir-opt --canonicalize --cse gpu.mlir -o optimized.mlir
```

### 與 pipeline.py 配合

```bash
# pipeline.py 使用 amdisa-translate
# 您可以在 pipeline.py 中添加對 amdisa-opt 的調用

# 例如修改 pipeline.py：
# 在轉換步驟之間插入 amdisa-opt
```

---

## 💡 最佳實踐

### 1. 逐步測試

```bash
# 不要一次運行所有 passes
# 而是逐步測試每個 pass

# 步驟 1
./build/bin/amdisa-opt --pass1 input.mlir -o step1.mlir

# 檢查 step1.mlir

# 步驟 2
./build/bin/amdisa-opt --pass2 step1.mlir -o step2.mlir

# 檢查 step2.mlir
```

### 2. 使用驗證

```bash
# 在每個步驟後驗證 IR
./build/bin/amdisa-opt --verify-each --pass-pipeline="..." input.mlir
```

### 3. 保存中間結果

```bash
# 保存每個階段的輸出，便於調試
./build/bin/amdisa-opt --pass1 input.mlir -o stage1.mlir
./build/bin/amdisa-opt --pass2 stage1.mlir -o stage2.mlir
./build/bin/amdisa-opt --pass3 stage2.mlir -o stage3.mlir
```

---

## 🎉 總結

**`amdisa-opt` 與 `mlir-opt` 完全相同的使用方式：**

1. ✅ 相同的命令行參數
2. ✅ 相同的 pass pipeline 語法
3. ✅ 相同的輸入輸出格式
4. ✅ 額外支援 AMDISA dialect 和 passes

**唯一的區別：**
- `mlir-opt`: 只認識標準 MLIR dialects
- `amdisa-opt`: 認識標準 MLIR dialects **+ AMDISA dialect**

**推薦用法：**
- 處理 AMDISA 相關的 MLIR → 使用 `amdisa-opt`
- 處理標準 MLIR → 可以使用 `mlir-opt` 或 `amdisa-opt`（都可以）

