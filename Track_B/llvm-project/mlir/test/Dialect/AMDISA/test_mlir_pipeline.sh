#!/bin/bash

# 端到端測試：驗證 MLIR 轉換管道的正確性
# 流程：原始 .s → AMDISA → GPU → 重建 .s → 組裝 → GPU 執行驗證

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置路徑
MLIR_OPT="/home/morhuang/llvm-project/build/bin/mlir-opt"
MLIR_TRANSLATE="/home/morhuang/llvm-project/build/bin/mlir-translate"
LLVM_MC="/home/morhuang/llvm-project/build/bin/llvm-mc"
LD_LLD="/home/morhuang/llvm-project/build/bin/ld.lld"
RUNNER="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner"
GPU_ARCH="gfx950"

# 檢查工具
check_tool() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}✗ 錯誤: 找不到工具 $1${NC}"
        exit 1
    fi
}

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}MLIR 轉換管道端到端測試${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 檢查所有必要工具
echo -e "${BLUE}檢查工具...${NC}"
check_tool "$MLIR_OPT"
check_tool "$MLIR_TRANSLATE"
check_tool "$LLVM_MC"
check_tool "$LD_LLD"
check_tool "$RUNNER"
echo -e "${GREEN}✓ 所有工具就緒${NC}"
echo ""

# 使用方式
if [ $# -lt 4 ]; then
    echo "Usage: $0 <test_dir> <kernel_name> <kernel_type> <test_size>"
    echo ""
    echo "範例: $0 test_results/test_01_vector_add _Z9vectorAddPKfS0_Pfi float_add 1024"
    exit 1
fi

TEST_DIR="$1"
KERNEL_NAME="$2"
KERNEL_TYPE="$3"
TEST_SIZE="$4"

# 檢查測試目錄
if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}✗ 錯誤: 找不到測試目錄 $TEST_DIR${NC}"
    exit 1
fi

# 輸入檔案
ORIGINAL_S="$TEST_DIR/original.s"
if [ ! -f "$ORIGINAL_S" ]; then
    echo -e "${RED}✗ 錯誤: 找不到原始組裝檔案 $ORIGINAL_S${NC}"
    exit 1
fi

# 中間和輸出檔案
STAGE1_AMDISA="$TEST_DIR/stage1_amdisa.mlir"
STAGE2_GPU="$TEST_DIR/stage2_gpu.mlir"
STAGE3_REBUILT_S="$TEST_DIR/stage3_rebuilt.s"
REBUILT_O="$TEST_DIR/rebuilt.o"
REBUILT_HSACO="$TEST_DIR/rebuilt.hsaco"
EXEC_LOG="$TEST_DIR/execution.log"

echo -e "${BLUE}配置資訊：${NC}"
echo "  測試目錄: $TEST_DIR"
echo "  Kernel 名稱: $KERNEL_NAME"
echo "  Kernel 類型: $KERNEL_TYPE"
echo "  測試大小: $TEST_SIZE"
echo "  GPU 架構: $GPU_ARCH"
echo ""

# Stage 1: 原始 .s → AMDISA dialect
echo -e "${YELLOW}[Stage 1/5] 原始 .s → AMDISA dialect${NC}"
echo "  輸入: $ORIGINAL_S"
echo "  輸出: $STAGE1_AMDISA"
if $MLIR_TRANSLATE --import-amdisa "$ORIGINAL_S" -o "$STAGE1_AMDISA" 2>"$TEST_DIR/stage1.log"; then
    lines=$(wc -l < "$STAGE1_AMDISA")
    echo -e "  ${GREEN}✓${NC} 轉換成功 ($lines 行)"
else
    echo -e "  ${RED}✗${NC} 轉換失敗"
    cat "$TEST_DIR/stage1.log"
    exit 1
fi
echo ""

# Stage 2: AMDISA → GPU dialect
echo -e "${YELLOW}[Stage 2/5] AMDISA → GPU dialect${NC}"
echo "  輸入: $STAGE1_AMDISA"
echo "  輸出: $STAGE2_GPU"
if $MLIR_OPT --convert-amdisa-to-gpu "$STAGE1_AMDISA" -o "$STAGE2_GPU" 2>"$TEST_DIR/stage2.log"; then
    lines=$(wc -l < "$STAGE2_GPU")
    echo -e "  ${GREEN}✓${NC} 轉換成功 ($lines 行)"
else
    echo -e "  ${RED}✗${NC} 轉換失敗"
    cat "$TEST_DIR/stage2.log"
    exit 1
fi
echo ""

# Stage 3: GPU dialect → 重建 .s
echo -e "${YELLOW}[Stage 3/5] GPU dialect → 重建 .s${NC}"
echo "  輸入: $STAGE2_GPU"
echo "  輸出: $STAGE3_REBUILT_S"
if $MLIR_TRANSLATE --mlir-to-amdisa "$STAGE2_GPU" -o "$STAGE3_REBUILT_S" 2>"$TEST_DIR/stage3.log"; then
    lines=$(wc -l < "$STAGE3_REBUILT_S")
    echo -e "  ${GREEN}✓${NC} 轉換成功 ($lines 行)"
else
    echo -e "  ${RED}✗${NC} 轉換失敗"
    cat "$TEST_DIR/stage3.log"
    exit 1
fi
echo ""

# Stage 4: 組裝 .s → .o → .hsaco
echo -e "${YELLOW}[Stage 4/5] 組裝 .s → .o → .hsaco${NC}"
echo "  [4.1] 組裝 .s → .o"
if $LLVM_MC -triple amdgcn-amd-amdhsa -mcpu=$GPU_ARCH -filetype=obj "$STAGE3_REBUILT_S" -o "$REBUILT_O" 2>"$TEST_DIR/assemble.log"; then
    obj_size=$(stat -c%s "$REBUILT_O")
    echo -e "    ${GREEN}✓${NC} 組裝成功: $REBUILT_O ($obj_size bytes)"
else
    echo -e "    ${RED}✗${NC} 組裝失敗"
    cat "$TEST_DIR/assemble.log"
    exit 1
fi

echo "  [4.2] 連結 .o → .hsaco"
if $LD_LLD -shared "$REBUILT_O" -o "$REBUILT_HSACO" 2>"$TEST_DIR/link.log"; then
    hsaco_size=$(stat -c%s "$REBUILT_HSACO")
    echo -e "    ${GREEN}✓${NC} 連結成功: $REBUILT_HSACO ($hsaco_size bytes)"
else
    echo -e "    ${RED}✗${NC} 連結失敗"
    cat "$TEST_DIR/link.log"
    exit 1
fi
echo ""

# Stage 5: GPU 執行驗證
echo -e "${YELLOW}[Stage 5/5] GPU 執行驗證${NC}"
echo "  執行: $RUNNER"
echo ""

if $RUNNER "$REBUILT_HSACO" "$KERNEL_NAME" "$KERNEL_TYPE" "$TEST_SIZE" > "$EXEC_LOG" 2>&1; then
    if grep -q "✅ PASS" "$EXEC_LOG"; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✅ 完整測試通過！${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        cat "$EXEC_LOG"
        echo ""
        echo -e "${GREEN}驗證總結：${NC}"
        echo -e "  ${GREEN}✓${NC} Stage 1: 原始 .s → AMDISA dialect"
        echo -e "  ${GREEN}✓${NC} Stage 2: AMDISA → GPU dialect"
        echo -e "  ${GREEN}✓${NC} Stage 3: GPU → 重建 .s"
        echo -e "  ${GREEN}✓${NC} Stage 4: 重建 .s → .o → .hsaco"
        echo -e "  ${GREEN}✓${NC} Stage 5: GPU 執行並驗證正確"
        echo ""
        echo -e "${GREEN}結論: MLIR 轉換完全正確，重建的 kernel 可以在 GPU 上正確執行！${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠ GPU 執行成功但驗證資訊不完整${NC}"
        cat "$EXEC_LOG"
        exit 0
    fi
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ GPU 執行失敗${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    cat "$EXEC_LOG"
    exit 1
fi

