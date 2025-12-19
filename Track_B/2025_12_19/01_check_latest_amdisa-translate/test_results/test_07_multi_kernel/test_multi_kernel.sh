#!/usr/bin/env bash
# ============================================================================
# Multi-Kernel HSACO Test Script
# ============================================================================
# 測試包含多個 kernel 的 HSACO 文件
# 
# 使用方法：
#   ./test_multi_kernel.sh
#

set -e  # 遇到錯誤立即退出

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 路徑配置
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_PY="/home/morhuang/Project-MDR/Track_B/llvm-project/my_test/pipeline.py"
RUNNER="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner"

# Kernel 配置 (從 test_config.json 讀取)
declare -a KERNELS=(
    "_Z9vectorAddPKfS0_Pfi:float_add:1024:vectorAdd"
    "_Z9scalarOpsPii:int_scalar:1024:scalarOps"
    "_Z9memoryOpsPKiPii:int_mem:1024:memoryOps"
)

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}               Multi-Kernel HSACO 測試${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "測試目錄: $TEST_DIR"
echo "輸入文件: original.s"
echo "Kernel 數量: ${#KERNELS[@]}"
echo ""

# ============================================================================
# 步驟 1: 使用 pipeline.py 處理
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}步驟 1: 使用 pipeline.py 處理 original.s${NC}"
echo -e "${BLUE}============================================================================${NC}"

cd "$TEST_DIR"

if [ ! -f "original.s" ]; then
    echo -e "${RED}❌ 錯誤: 找不到 original.s${NC}"
    exit 1
fi

echo "執行 pipeline.py..."
python3 "$PIPELINE_PY" \
    --chip gfx950 \
    --workdir pipeline_output \
    original.s

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ pipeline.py 執行成功${NC}"
else
    echo -e "${RED}❌ pipeline.py 執行失敗${NC}"
    exit 1
fi

echo ""

# ============================================================================
# 步驟 2: 檢查輸出文件
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}步驟 2: 檢查輸出文件${NC}"
echo -e "${BLUE}============================================================================${NC}"

FILES=(
    "pipeline_output/original_rebuilt.amdisamlir"
    "pipeline_output/original_rebuilt.gpumlir"
    "pipeline_output/original_rebuilt.s"
    "pipeline_output/original_rebuilt.hsaco"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo -e "${GREEN}✅${NC} $file ($size)"
    else
        echo -e "${RED}❌${NC} $file (不存在)"
    fi
done

echo ""

# ============================================================================
# 步驟 3: 檢查 MLIR 中的 kernel 數量
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}步驟 3: 檢查 MLIR 中的 kernel 數量${NC}"
echo -e "${BLUE}============================================================================${NC}"

if [ -f "pipeline_output/original_rebuilt.gpumlir" ]; then
    kernel_count=$(grep -c "gpu.func @" pipeline_output/original_rebuilt.gpumlir || true)
    echo "檢測到的 GPU kernel 數量: $kernel_count"
    
    if [ "$kernel_count" -eq "${#KERNELS[@]}" ]; then
        echo -e "${GREEN}✅ Kernel 數量正確 (預期: ${#KERNELS[@]}, 實際: $kernel_count)${NC}"
    else
        echo -e "${YELLOW}⚠️  Kernel 數量不符 (預期: ${#KERNELS[@]}, 實際: $kernel_count)${NC}"
    fi
    
    echo ""
    echo "Kernel 列表:"
    grep "gpu.func @" pipeline_output/original_rebuilt.gpumlir | while read -r line; do
        kernel_name=$(echo "$line" | sed -n 's/.*gpu\.func @\([^(]*\).*/\1/p')
        echo "  • $kernel_name"
    done
else
    echo -e "${RED}❌ 找不到 GPU MLIR 文件${NC}"
fi

echo ""

# ============================================================================
# 步驟 4: 生成原始 HSACO 用於對比
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}步驟 4: 生成原始 HSACO${NC}"
echo -e "${BLUE}============================================================================${NC}"

echo "組裝 original.s..."
llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx950 -filetype=obj original.s -o original.o

echo "連結生成 original.hsaco..."
ld.lld -shared original.o -o original.hsaco

if [ -f "original.hsaco" ]; then
    size=$(du -h original.hsaco | cut -f1)
    echo -e "${GREEN}✅ original.hsaco 生成成功 ($size)${NC}"
else
    echo -e "${RED}❌ original.hsaco 生成失敗${NC}"
    exit 1
fi

echo ""

# ============================================================================
# 步驟 5: 測試每個 kernel
# ============================================================================
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}步驟 5: 測試每個 kernel (原始 vs 重建)${NC}"
echo -e "${BLUE}============================================================================${NC}"

declare -i TOTAL=0
declare -i PASSED=0
declare -i FAILED=0

for kernel_info in "${KERNELS[@]}"; do
    IFS=':' read -r kernel_name kernel_type test_size display_name <<< "$kernel_info"
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}測試 kernel: $display_name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "  函數名: $kernel_name"
    echo "  類型: $kernel_type"
    echo "  測試大小: $test_size"
    echo ""
    
    TOTAL=$((TOTAL + 1))
    
    # 測試原始 HSACO
    echo -e "${YELLOW}🔵 測試原始 HSACO...${NC}"
    ORIGINAL_OUTPUT=$("$RUNNER" original.hsaco "$kernel_name" "$kernel_type" "$test_size" 2>&1)
    ORIGINAL_EXIT=$?
    
    if [ $ORIGINAL_EXIT -eq 0 ] && echo "$ORIGINAL_OUTPUT" | grep -q "PASS"; then
        echo -e "${GREEN}✅ 原始 HSACO 執行成功${NC}"
        ORIGINAL_RESULT=$(echo "$ORIGINAL_OUTPUT" | grep "^\[0\]" | head -1)
        echo "  樣本: $ORIGINAL_RESULT"
    else
        echo -e "${RED}❌ 原始 HSACO 執行失敗${NC}"
        echo "$ORIGINAL_OUTPUT" | tail -5
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # 測試重建 HSACO
    echo ""
    echo -e "${YELLOW}🟢 測試重建 HSACO...${NC}"
    REBUILT_OUTPUT=$("$RUNNER" pipeline_output/original_rebuilt.hsaco "$kernel_name" "$kernel_type" "$test_size" 2>&1)
    REBUILT_EXIT=$?
    
    if [ $REBUILT_EXIT -eq 0 ] && echo "$REBUILT_OUTPUT" | grep -q "PASS"; then
        echo -e "${GREEN}✅ 重建 HSACO 執行成功${NC}"
        REBUILT_RESULT=$(echo "$REBUILT_OUTPUT" | grep "^\[0\]" | head -1)
        echo "  樣本: $REBUILT_RESULT"
    else
        echo -e "${RED}❌ 重建 HSACO 執行失敗${NC}"
        echo "$REBUILT_OUTPUT" | tail -5
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # 比較結果
    echo ""
    echo -e "${YELLOW}🔍 比較結果...${NC}"
    
    if [ "$ORIGINAL_RESULT" = "$REBUILT_RESULT" ]; then
        echo -e "${GREEN}✅ 結果一致！${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}❌ 結果不一致！${NC}"
        echo "  原始: $ORIGINAL_RESULT"
        echo "  重建: $REBUILT_RESULT"
        FAILED=$((FAILED + 1))
    fi
done

# ============================================================================
# 總結
# ============================================================================
echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}                           測試總結${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "總測試數: $TOTAL"
echo -e "${GREEN}通過: $PASSED${NC}"
echo -e "${RED}失敗: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 所有測試通過！${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}❌ 部分測試失敗${NC}"
    exit 1
fi

