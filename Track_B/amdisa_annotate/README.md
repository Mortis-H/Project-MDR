# AMD GCN ISA Annotator

AMD GCN (Graphics Core Next) 組合語言標註工具。

## 功能

解析 AMD GCN ISA 組合語言文件（`.s`），並標註每一行的類型：

- `[Label]` - 標籤定義
- `[Instruction]` - 指令（含 opcode 和 operands）
- `[Directive]` - 組譯器指示
- `[Comment]` - 註解
- `[Metadata]` - YAML metadata 區塊
- `[Kernel Name]` - Kernel 函數名稱（`.globl`/`.global`）

## 快速開始

### 編譯

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

編譯完成後會生成 `build/amdisa_annotate` 執行檔。

### 使用方式

#### 1. 基本標註

```bash
./build/amdisa_annotate input.s > output.txt
```

輸出範例：
```
[Kernel Name]        .globl  _Z23attend_bwd_combined_kerILi128EEv...
[Label]              _Z23attend_bwd_combined_kerILi128EEv...:
[Instruction]        s_load_dwordx2 s[50:51], s[0:1], 0x0
[Instruction]        s_load_dwordx4 s[20:23], s[0:1], 0x10
[Comment]            ; %bb.0:
[Directive]          .text
```

#### 2. 顯示操作數類型

```bash
./build/amdisa_annotate --show-types input.s > output_with_types.txt
```

輸出範例：
```
[Instruction] s_load_dwordx2 ["s[50:51]":reg, "s[0:1]":reg, "0x0":imm]
[Instruction] v_add_f32 ["v0":reg, "v1":reg, "v2":reg]
[Instruction] s_branch ["BB0_3":label]
```

操作數類型：
- `:reg` - 暫存器（SGPR/VGPR/ACC）
- `:imm` - 立即值（數字、十六進位）
- `:label` - 標籤引用
- `:expr` - 表達式
- `:unknown` - 無法識別的類型

#### 3. 隱藏註解行

```bash
./build/amdisa_annotate --hide-comments input.s > output.txt
```

過濾掉所有 `[Comment]` 行，只保留有效的指令、標籤和 directive。

範例（從 14,013 行減少到 5,026 行）：
- 原始輸出包含 8,987 行註解
- 使用 `--hide-comments` 後全部過濾

#### 4. 顯示統計摘要

```bash
./build/amdisa_annotate --summary input.s 2>&1 >/dev/null
```

輸出範例：
```
=== Parsing Summary ===
Total lines:     5053
Instructions:    4885
Labels:          2
Directives:      153
Comments:        0
Metadata:        13
Kernel Names:    2
```

## 項目結構

```
02_amdisa-paser-II/
├── main.cpp               # 核心程式
├── ParsedProgram.h        # 解析結果數據結構
├── CMakeLists.txt         # CMake 構建配置
├── README.md              # 本文件
└── build/                 # 構建目錄
    └── amdisa_annotate    # 編譯後的執行檔
```

## 進階選項

### 指定目標架構

```bash
./build/amdisa_annotate --triple=amdgcn-amd-amdhsa --mcpu=gfx950 input.s
```

預設值：
- `--triple`: `amdgcn-amd-amdhsa`
- `--mcpu`: `gfx950`

### 從標準輸入讀取

```bash
cat input.s | ./build/amdisa_annotate
```

## 技術細節

### 使用的 LLVM 組件

- **MCParser**: 組合語言解析器
- **MCStreamer**: 處理解析事件
- **MCContext**: 管理符號和區段
- **MCInstrInfo**: 指令信息
- **MCRegisterInfo**: 暫存器信息
- **MCSubtargetInfo**: 子架構信息

### 處理流程

1. **第一次掃描**：移除不支援的 AMD 特定 directives（`.amdhsa_*`, `.amdgpu_*`）
2. **LLVM MC 解析**：使用 LLVM 的 MC Parser 解析組合語言
3. **第二次掃描**：分析每一行並分類（Label/Instruction/Directive/Comment/Metadata/Kernel Name）
4. **輸出格式化**：根據選項輸出標註結果

### YAML Metadata 處理

特別處理 `.amdgpu_metadata` ... `.end_amdgpu_metadata` 區塊：

```
[Metadata]      .amdgpu_metadata
                amdhsa.kernels:
                  - .args: []
                    .group_segment_fixed_size: 65536
                    ...
[Metadata]      .end_amdgpu_metadata
```

## 系統需求

- **LLVM**: 需要 LLVM 開發庫（MC, Support 等）
- **CMake**: 3.13 或更高版本
- **C++**: C++17 或更高標準
- **編譯器**: GCC 或 Clang

## 故障排除

### 找不到 LLVM

```bash
# 指定 LLVM 路徑
cmake -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm ..
```

### 執行檔太大

確認使用 Release 模式編譯：

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

執行檔會自動執行 `strip` 移除 debug symbols。

### 解析錯誤

如果遇到解析錯誤，可能的原因：
1. 使用了不支援的指令或 directive
2. 目標架構不匹配（檢查 `--triple` 和 `--mcpu`）
3. 組合語言語法錯誤

## 範例

### 完整工作流程

```bash
# 1. 編譯工具
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
cd ..

# 2. 基本標註
./build/amdisa_annotate kernel.s > kernel_annotated.txt

# 3. 帶類型標註
./build/amdisa_annotate --show-types kernel.s > kernel_with_types.txt

# 4. 查看統計
./build/amdisa_annotate --summary kernel.s 2>&1 >/dev/null

# 5. 只看指令行
./build/amdisa_annotate kernel.s | grep '^\[Instruction\]'

# 6. 統計指令數
./build/amdisa_annotate kernel.s | grep -c '^\[Instruction\]'
```

