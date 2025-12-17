#!/bin/bash

# 測試流程：從 .s 組裝檔案 → .o 物件檔 → .hsaco → GPU 執行驗證
# 這個腳本驗證 MLIR 轉換後的 .s 檔案是否能正確執行

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置路徑
LLVM_MC="/home/morhuang/llvm-project/build/bin/llvm-mc"
LD_LLD="/home/morhuang/llvm-project/build/bin/ld.lld"
RUNNER="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner"
GPU_ARCH="gfx950"

# 檢查工具是否存在
check_tool() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}✗ 錯誤: 找不到工具 $1${NC}"
        exit 1
    fi
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}從組裝檔案到 GPU 執行驗證${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

check_tool "$LLVM_MC"
check_tool "$LD_LLD"
check_tool "$RUNNER"

# 使用方式
if [ $# -lt 4 ]; then
    echo "Usage: $0 <input.s> <kernel_name> <kernel_type> <test_size>"
    echo ""
    echo "Kernel types:"
    echo "  float_add   - float vector addition"
    echo "  int_scalar  - int scalar operations"
    echo "  int_mem     - int memory operations"
    echo "  int_cond    - int conditional"
    echo "  int_loop    - int loop"
    echo "  int_shared  - int shared memory"
    echo ""
    echo "Example: $0 stage3_rebuilt.s _Z9vectorAddPKfS0_Pfi float_add 1024"
    exit 1
fi

INPUT_S="$1"
KERNEL_NAME="$2"
KERNEL_TYPE="$3"
TEST_SIZE="$4"

# 檢查輸入檔案
if [ ! -f "$INPUT_S" ]; then
    echo -e "${RED}✗ 錯誤: 找不到輸入檔案 $INPUT_S${NC}"
    exit 1
fi

# 輸出檔案
BASENAME=$(basename "$INPUT_S" .s)
DIR=$(dirname "$INPUT_S")
OUTPUT_O="$DIR/${BASENAME}_test.o"
OUTPUT_HSACO="$DIR/${BASENAME}_test.hsaco"

echo -e "${BLUE}配置資訊：${NC}"
echo "  輸入檔案: $INPUT_S"
echo "  Kernel 名稱: $KERNEL_NAME"
echo "  Kernel 類型: $KERNEL_TYPE"
echo "  測試大小: $TEST_SIZE"
echo "  GPU 架構: $GPU_ARCH"
echo ""

# Step 1: 組裝 .s → .o
echo -e "${YELLOW}[1/3] 組裝 .s → .o${NC}"
echo "  命令: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=$GPU_ARCH -filetype=obj"
if $LLVM_MC -triple amdgcn-amd-amdhsa -mcpu=$GPU_ARCH -filetype=obj "$INPUT_S" -o "$OUTPUT_O" 2>&1; then
    obj_size=$(stat -c%s "$OUTPUT_O")
    echo -e "  ${GREEN}✓${NC} 組裝成功: $OUTPUT_O ($obj_size bytes)"
else
    echo -e "  ${RED}✗${NC} 組裝失敗"
    exit 1
fi
echo ""

# Step 2: 連結 .o → .hsaco
echo -e "${YELLOW}[2/3] 連結 .o → .hsaco${NC}"
echo "  命令: ld.lld -shared"
if $LD_LLD -shared "$OUTPUT_O" -o "$OUTPUT_HSACO" 2>&1; then
    hsaco_size=$(stat -c%s "$OUTPUT_HSACO")
    echo -e "  ${GREEN}✓${NC} 連結成功: $OUTPUT_HSACO ($hsaco_size bytes)"
else
    echo -e "  ${RED}✗${NC} 連結失敗"
    exit 1
fi
echo ""

# Step 3: GPU 執行驗證
echo -e "${YELLOW}[3/3] GPU 執行驗證${NC}"
echo "  命令: universal_hsaco_runner $OUTPUT_HSACO $KERNEL_NAME $KERNEL_TYPE $TEST_SIZE"
echo ""

EXEC_LOG="$DIR/${BASENAME}_exec.log"
if $RUNNER "$OUTPUT_HSACO" "$KERNEL_NAME" "$KERNEL_TYPE" "$TEST_SIZE" > "$EXEC_LOG" 2>&1; then
    if grep -q "✅ PASS" "$EXEC_LOG"; then
        echo -e "${GREEN}✓ GPU 執行成功並通過驗證${NC}"
        echo ""
        echo -e "${BLUE}執行結果：${NC}"
        cat "$EXEC_LOG"
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✅ 測試通過！${NC}"
        echo -e "${GREEN}從 .s 檔案到 GPU 執行完全正確${NC}"
        echo -e "${GREEN}========================================${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠ GPU 執行成功但驗證資訊不完整${NC}"
        cat "$EXEC_LOG"
        exit 0
    fi
else
    echo -e "${RED}✗ GPU 執行失敗${NC}"
    echo ""
    echo -e "${RED}錯誤日誌：${NC}"
    cat "$EXEC_LOG"
    exit 1
fi

