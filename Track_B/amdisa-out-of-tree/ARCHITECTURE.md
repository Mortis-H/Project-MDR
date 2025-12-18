# AMDISA Out-of-Tree 架構說明

## 📐 架構對比

### In-Tree 架構 (原始)

```
llvm-project/
├── llvm/
├── clang/
├── mlir/
│   ├── include/mlir/
│   │   └── Dialect/
│   │       ├── AMDGPU/
│   │       ├── AMDISA/         ← 您的 Dialect
│   │       │   ├── IR/
│   │       │   │   ├── AMDISAOps.h
│   │       │   │   └── AMDISAOps.td
│   │       │   └── Passes.h
│   │       ├── GPU/
│   │       └── ...
│   │
│   ├── lib/
│   │   ├── Dialect/
│   │   │   ├── AMDISA/         ← 您的實作
│   │   │   │   ├── IR/
│   │   │   │   └── Transforms/
│   │   │   └── ...
│   │   └── RegisterAllDialects.cpp  ← 註冊所有 Dialects
│   │
│   └── tools/
│       ├── mlir-opt/
│       ├── mlir-translate/
│       └── amdisa-translate/   ← 您的工具
│
└── build/                      ← 建置所有元件
    ├── bin/
    │   ├── mlir-opt
    │   ├── clang
    │   └── amdisa-translate
    └── lib/

問題：
❌ 修改需要重新建置整個 LLVM (數小時)
❌ 無法獨立發佈
❌ 版本控制混雜
❌ 難以分享給其他人
```

### Out-of-Tree 架構 (新)

```
/home/morhuang/Project-MDR/Track_B/
│
├── llvm-project/               ← LLVM/MLIR 基礎 (一次性建置)
│   ├── llvm/
│   ├── mlir/
│   └── build/
│       └── install/            ← 安裝的 LLVM/MLIR
│           ├── include/        (提供 API)
│           ├── lib/            (提供函式庫)
│           └── bin/
│
└── amdisa-out-of-tree/         ← 您的獨立專案 ✨
    ├── include/AMDISA/         ← 您的公開介面
    │   ├── IR/
    │   │   ├── AMDISAOps.h
    │   │   └── AMDISAOps.td
    │   └── Passes.h
    │
    ├── lib/AMDISA/             ← 您的實作
    │   ├── IR/
    │   │   └── AMDISAOps.cpp
    │   └── Transforms/
    │       └── LowerToGPUInlineAsm.cpp
    │
    ├── tools/
    │   └── amdisa-translate/   ← 您的工具
    │       ├── amdisa-translate.cpp
    │       └── ...
    │
    └── build/                  ← 只建置您的專案 (分鐘級)
        ├── bin/
        │   └── amdisa-translate
        └── lib/
            ├── libMLIRAMDISA.a
            └── libMLIRAMDISATransforms.a

優點：
✅ 快速迭代 (只建置您的程式碼)
✅ 獨立版本控制
✅ 容易分享和發佈
✅ 清晰的依賴關係
```

## 🔄 建置流程對比

### In-Tree 建置流程

```
┌─────────────────────────────────────────────┐
│ 1. 修改 AMDISA 程式碼                       │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ 2. 建置整個 LLVM/MLIR                      │
│    - Clang (不需要但被建置)                │
│    - 所有 MLIR Dialects                     │
│    - 所有工具                              │
│    時間: 1-4 小時 ⏰⏰⏰                     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ 3. 測試                                     │
└─────────────────────────────────────────────┘

總時間: 數小時 (每次修改都要重複)
```

### Out-of-Tree 建置流程

```
┌─────────────────────────────────────────────┐
│ 0. 建置 LLVM/MLIR (只做一次)               │
│    時間: 1-4 小時 ⏰⏰⏰                     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ 1. 修改 AMDISA 程式碼                       │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ 2. 只建置 AMDISA 專案                       │
│    - 您的 Dialect                          │
│    - 您的 Transforms                        │
│    - 您的工具                              │
│    時間: 10-30 秒 ⚡                        │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│ 3. 測試                                     │
└─────────────────────────────────────────────┘

總時間: 秒級 (後續修改)
```

## 🔌 依賴關係圖

### In-Tree 依賴

```
amdisa-translate
       │
       ├─→ AMDISA Dialect (您的)
       │
       └─→ 所有 MLIR 內部元件
              ├─→ GPU Dialect
              ├─→ LLVM Dialect
              ├─→ Arith Dialect
              ├─→ ... (50+ Dialects)
              └─→ MLIR Core

問題: 緊密耦合，無法單獨使用
```

### Out-of-Tree 依賴

```
amdisa-translate
       │
       ├─→ libMLIRAMDISA.a (您的 Dialect)
       │      └─→ AMDISA IR
       │
       ├─→ libMLIRAMDISATransforms.a (您的 Passes)
       │      └─→ LowerToGPUInlineAsm
       │
       └─→ 已安裝的 MLIR (清晰的 API 邊界)
              ├─→ MLIRIR (核心)
              ├─→ MLIRGPUDialect (GPU 支援)
              ├─→ MLIRLLVMDialect (LLVM 後端)
              ├─→ MLIRParser (解析器)
              └─→ MLIRPass (Pass 基礎設施)

優點: 明確的依賴，可獨立發佈
```

## 📦 模組化設計

### AMDISA 模組結構

```
┌─────────────────────────────────────────────────┐
│             amdisa-translate (工具)            │
│  解析 .s → AMDISA IR → GPU Inline ASM         │
└───────┬────────────────────────┬────────────────┘
        │                        │
        ▼                        ▼
┌──────────────────┐    ┌──────────────────────┐
│ MLIRAMDISA       │    │ MLIRAMDISATransforms │
│ (Dialect 定義)   │    │ (轉換 Passes)        │
│                  │    │                      │
│ - InstOp         │    │ - LowerToGPU...Pass  │
│ - LabelOp        │    │                      │
└──────────────────┘    └──────────────────────┘
        │                        │
        └────────┬───────────────┘
                 ▼
        ┌─────────────────┐
        │   MLIR Core     │
        │   (已安裝)      │
        └─────────────────┘
```

## 🎯 使用場景對比

### In-Tree 適合

- ✅ 為 LLVM 專案貢獻上游程式碼
- ✅ 需要修改 MLIR 核心
- ✅ 開發新的基礎設施

### Out-of-Tree 適合 (您的情況)

- ✅ 獨立的 Dialect 開發
- ✅ 自訂工具和應用
- ✅ 快速原型和迭代
- ✅ 私有或內部專案
- ✅ 需要獨立發佈和版本控制

## 📊 效能影響

| 操作 | In-Tree | Out-of-Tree |
|------|---------|-------------|
| **初始建置** | 2-4 小時 | 2-4 小時 (LLVM) + 1 分鐘 (AMDISA) |
| **增量建置** | 30 秒 - 2 小時 | 5-30 秒 |
| **完全重建** | 2-4 小時 | 1 分鐘 |
| **執行效能** | 相同 | 相同 |
| **記憶體使用** | 相同 | 相同 |

## 🔧 維護優勢

### 版本控制

**In-Tree:**
```bash
cd llvm-project
git status  # 混雜 LLVM 和您的修改
```

**Out-of-Tree:**
```bash
cd amdisa-out-of-tree
git init
git add .
git commit -m "Initial AMDISA project"
git remote add origin your-repo-url
git push  # 只包含您的程式碼
```

### 更新 LLVM

**In-Tree:**
```bash
cd llvm-project
git pull  # 可能衝突
# 解決與您修改的衝突
ninja  # 重新建置一切
```

**Out-of-Tree:**
```bash
cd llvm-project
git pull
ninja install  # 更新已安裝的 LLVM

cd amdisa-out-of-tree
# 如果 API 相容，不需要任何修改
ninja  # 快速重建
```

## 🚀 分發策略

### Out-of-Tree 可以

1. **發佈為獨立套件**
   ```bash
   tar czf amdisa-v1.0.tar.gz amdisa-out-of-tree/
   ```

2. **提供建置說明**
   - 使用者只需安裝 LLVM/MLIR
   - 然後建置您的專案

3. **容器化**
   ```dockerfile
   FROM llvm-mlir:latest
   COPY amdisa-out-of-tree /workspace
   RUN cd /workspace && mkdir build && cd build && \
       cmake .. && ninja
   ```

4. **CI/CD 整合**
   - 快速測試 (只建置您的程式碼)
   - 多版本測試 (針對不同 LLVM 版本)

## 📈 擴展性

### 新增功能

**In-Tree:**
- 修改 → 建置整個 LLVM → 測試

**Out-of-Tree:**
- 修改 → 建置 AMDISA (秒級) → 測試

### 團隊協作

**Out-of-Tree 優勢:**
- 清晰的專案邊界
- 獨立的 Git 倉庫
- 簡化的 CI/CD
- 更容易的程式碼審查

---

## 總結

Out-of-tree 架構為您提供：

1. **⚡ 開發效率**: 秒級建置時間
2. **📦 模組化**: 清晰的依賴關係
3. **🔄 獨立性**: 自主版本控制
4. **🚀 分發**: 容易分享和部署
5. **👥 協作**: 簡化的團隊工作流程

**您的 AMDISA 專案現在已經完全準備好以 out-of-tree 方式運作！** 🎉

