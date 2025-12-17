#!/bin/bash

# 執行所有 6 個 kernel 的完整 MLIR 轉換管道測試

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 配置
TEST_RESULTS_DIR="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/test_results"
PIPELINE_SCRIPT="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/test_mlir_pipeline.sh"
TEST_SIZE=1024

# Kernel 配置 (測試目錄, mangled 名稱, 類型)
declare -a TESTS=(
    "test_01_vector_add:_Z9vectorAddPKfS0_Pfi:float_add"
    "test_02_scalar_ops:_Z9scalarOpsPii:int_scalar"
    "test_03_memory_ops:_Z9memoryOpsPKiPii:int_mem"
    "test_04_conditional:_Z17conditionalKernelPKiPii:int_cond"
    "test_05_loop:_Z10loopKernelPii:int_loop"
    "test_06_shared_memory:_Z15sharedMemKernelPKiPii:int_shared"
)

echo -e "${CYAN}========================================"
echo -e "MLIR 轉換管道完整測試"
echo -e "測試所有 6 個 Kernel"
echo -e "========================================${NC}"
echo ""

TOTAL=0
PASSED=0
FAILED=0
declare -a FAILED_TESTS

for test_info in "${TESTS[@]}"; do
    IFS=':' read -r test_name kernel_name kernel_type <<< "$test_info"
    test_dir="$TEST_RESULTS_DIR/$test_name"
    
    TOTAL=$((TOTAL + 1))
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[$TOTAL/6] $test_name${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ ! -d "$test_dir" ]; then
        echo -e "${RED}✗ 測試目錄不存在: $test_dir${NC}"
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$test_name (目錄不存在)")
        echo ""
        continue
    fi
    
    if [ ! -f "$test_dir/original.s" ]; then
        echo -e "${RED}✗ 找不到 original.s${NC}"
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$test_name (缺少 original.s)")
        echo ""
        continue
    fi
    
    # 執行管道測試
    if $PIPELINE_SCRIPT "$test_dir" "$kernel_name" "$kernel_type" "$TEST_SIZE"; then
        echo -e "${GREEN}✓ $test_name 測試通過${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ $test_name 測試失敗${NC}"
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$test_name")
    fi
    
    echo ""
done

# 總結報告
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}測試總結${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "總測試數: ${BLUE}$TOTAL${NC}"
echo -e "通過: ${GREEN}$PASSED${NC}"
echo -e "失敗: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}🎉 所有測試通過！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${GREEN}驗證結果：${NC}"
    echo -e "  • ${GREEN}✓${NC} 6/6 kernel: 原始 .s → AMDISA dialect"
    echo -e "  • ${GREEN}✓${NC} 6/6 kernel: AMDISA → GPU dialect"
    echo -e "  • ${GREEN}✓${NC} 6/6 kernel: GPU → 重建 .s"
    echo -e "  • ${GREEN}✓${NC} 6/6 kernel: 重建 .s → .o → .hsaco"
    echo -e "  • ${GREEN}✓${NC} 6/6 kernel: 在 AMD Instinct MI350X 上成功執行"
    echo ""
    echo -e "${GREEN}✅ 結論: MLIR AMDISA dialect 轉換完全正確！${NC}"
    echo -e "${GREEN}   從組裝檔案轉換後的 kernel 可以在 GPU 上正確執行${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}⚠️  有 $FAILED 個測試失敗${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${RED}失敗的測試：${NC}"
    for failed_test in "${FAILED_TESTS[@]}"; do
        echo -e "  • ${RED}✗${NC} $failed_test"
    done
    echo ""
    exit 1
fi

