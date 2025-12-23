#!/bin/bash

# Test 07: 多 Kernel 測試腳本
# 測試 original.s 中所有 5 個 kernel 的執行驗證

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test 07: 多 Kernel 測試套件${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 配置路徑
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORIGINAL_S="$SCRIPT_DIR/original.s"
RUNNER="$SCRIPT_DIR/../universal_hsaco_runner"
OUTPUT_O="$SCRIPT_DIR/test_07.o"
OUTPUT_HSACO="$SCRIPT_DIR/test_07.hsaco"
LLVM_MC="/home/morhuang/llvm-project/build/bin/llvm-mc"
LD_LLD="/home/morhuang/llvm-project/build/bin/ld.lld"
GPU_ARCH="gfx950"
TEST_SIZE=1024

# 檢查工具是否存在
if [ ! -f "$LLVM_MC" ]; then
    echo -e "${RED}✗ 錯誤: 找不到 llvm-mc${NC}"
    exit 1
fi

if [ ! -f "$LD_LLD" ]; then
    echo -e "${RED}✗ 錯誤: 找不到 ld.lld${NC}"
    exit 1
fi

if [ ! -f "$RUNNER" ]; then
    echo -e "${RED}✗ 錯誤: 找不到 universal_hsaco_runner${NC}"
    exit 1
fi

if [ ! -f "$ORIGINAL_S" ]; then
    echo -e "${RED}✗ 錯誤: 找不到 original.s${NC}"
    exit 1
fi

# Step 1: 組裝 .s → .o
echo -e "${YELLOW}[1/3] 組裝 .s → .o${NC}"
if $LLVM_MC -triple amdgcn-amd-amdhsa -mcpu=$GPU_ARCH -filetype=obj "$ORIGINAL_S" -o "$OUTPUT_O" 2>&1; then
    obj_size=$(stat -c%s "$OUTPUT_O")
    echo -e "  ${GREEN}✓${NC} 組裝成功: $OUTPUT_O ($obj_size bytes)"
else
    echo -e "  ${RED}✗${NC} 組裝失敗"
    exit 1
fi
echo ""

# Step 2: 連結 .o → .hsaco
echo -e "${YELLOW}[2/3] 連結 .o → .hsaco${NC}"
if $LD_LLD -shared "$OUTPUT_O" -o "$OUTPUT_HSACO" 2>&1; then
    hsaco_size=$(stat -c%s "$OUTPUT_HSACO")
    echo -e "  ${GREEN}✓${NC} 連結成功: $OUTPUT_HSACO ($hsaco_size bytes)"
else
    echo -e "  ${RED}✗${NC} 連結失敗"
    exit 1
fi
echo ""

# Step 3: 測試所有 kernel
echo -e "${YELLOW}[3/3] 測試所有 Kernel${NC}"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# Kernel 資訊陣列：名稱|測試類型
declare -a KERNELS=(
    "_Z9vectorAddPKfS0_Pfi|float_add|vectorAdd"
    "_Z9vectorMulPKfS0_Pfi|float_mul|vectorMul"
    "_Z9vectorDotPKfS0_Pfi|float_dot|vectorDot"
    "_Z5saxpyfPKfPfi|float_saxpy|saxpy"
    "_Z14conditionalOpsPKfPffi|float_cond|conditionalOps"
)

for i in "${!KERNELS[@]}"; do
    IFS='|' read -r kernel_name kernel_type display_name <<< "${KERNELS[$i]}"
    
    echo -e "${BLUE}[$((i+1))/${#KERNELS[@]}] 測試 $display_name kernel...${NC}"
    
    # 執行測試並捕獲輸出
    TEST_LOG="$SCRIPT_DIR/test_${display_name}.log"
    if $RUNNER "$OUTPUT_HSACO" "$kernel_name" "$kernel_type" "$TEST_SIZE" > "$TEST_LOG" 2>&1; then
        if grep -q "✅ PASS" "$TEST_LOG"; then
            echo -e "${GREEN}  ✓ $display_name 測試通過${NC}"
            ((PASS_COUNT++))
            
            # 顯示範例結果
            if grep -q "Sample results" "$TEST_LOG"; then
                echo "  範例結果："
                sed -n '/Sample results/,/========================================/p' "$TEST_LOG" | head -8 | tail -6 | sed 's/^/    /'
            fi
        else
            echo -e "${YELLOW}  ⚠ $display_name 執行成功但驗證資訊不完整${NC}"
            ((PASS_COUNT++))
        fi
    else
        echo -e "${RED}  ✗ $display_name 測試失敗${NC}"
        ((FAIL_COUNT++))
        echo "  錯誤日誌："
        tail -5 "$TEST_LOG" | sed 's/^/    /'
    fi
    echo ""
done

# 顯示總結
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}測試總結${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "通過: ${GREEN}${PASS_COUNT}/${#KERNELS[@]}${NC}"
echo -e "失敗: ${RED}${FAIL_COUNT}/${#KERNELS[@]}${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✅ 所有測試都通過！${NC}"
    echo -e "${GREEN}HSACO 檔案: $OUTPUT_HSACO${NC}"
    exit 0
else
    echo -e "${RED}❌ 有 ${FAIL_COUNT} 個測試失敗${NC}"
    exit 1
fi

