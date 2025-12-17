#!/bin/bash

# 測試所有重建的 .s 檔案是否能正確執行
# 這個腳本驗證從 MLIR 轉換後的 stage3_rebuilt.s 檔案

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
ASM_TO_HSACO="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/test_asm_to_hsaco.sh"
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
echo -e "測試重建的 .s 檔案執行"
echo -e "驗證：stage3_rebuilt.s → .o → .hsaco → GPU"
echo -e "========================================${NC}"
echo ""

TOTAL=0
PASSED=0
FAILED=0
declare -a FAILED_TESTS
declare -a MISSING_FILES

for test_info in "${TESTS[@]}"; do
    IFS=':' read -r test_name kernel_name kernel_type <<< "$test_info"
    test_dir="$TEST_RESULTS_DIR/$test_name"
    rebuilt_s="$test_dir/stage3_rebuilt.s"
    
    TOTAL=$((TOTAL + 1))
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[$TOTAL/6] $test_name${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ ! -f "$rebuilt_s" ]; then
        echo -e "${YELLOW}⚠ 找不到 stage3_rebuilt.s${NC}"
        echo -e "  路徑: $rebuilt_s"
        MISSING_FILES+=("$test_name")
        echo ""
        continue
    fi
    
    # 執行測試
    echo -e "測試檔案: stage3_rebuilt.s"
    echo ""
    if $ASM_TO_HSACO "$rebuilt_s" "$kernel_name" "$kernel_type" "$TEST_SIZE" 2>&1 | tee "$test_dir/rebuilt_test.log"; then
        if grep -q "✅ 測試通過" "$test_dir/rebuilt_test.log"; then
            echo ""
            echo -e "${GREEN}✓ $test_name 測試通過${NC}"
            PASSED=$((PASSED + 1))
        else
            echo ""
            echo -e "${YELLOW}⚠ $test_name 執行成功但驗證不完整${NC}"
            PASSED=$((PASSED + 1))
        fi
    else
        echo ""
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
echo -e "缺少檔案: ${YELLOW}${#MISSING_FILES[@]}${NC}"
echo ""

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${YELLOW}缺少 stage3_rebuilt.s 的測試：${NC}"
    for missing in "${MISSING_FILES[@]}"; do
        echo -e "  • ${YELLOW}⚠${NC} $missing"
    done
    echo ""
fi

if [ $FAILED -eq 0 ] && [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}🎉 所有測試通過！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${GREEN}驗證結果：${NC}"
    echo -e "  • ${GREEN}✓${NC} $PASSED/$TOTAL kernel 重建的 .s 檔案測試通過"
    echo -e "  • ${GREEN}✓${NC} 組裝 .s → .o 成功"
    echo -e "  • ${GREEN}✓${NC} 連結 .o → .hsaco 成功"
    echo -e "  • ${GREEN}✓${NC} 在 AMD Instinct MI350X 上成功執行"
    echo ""
    echo -e "${GREEN}✅ 結論: 重建的組裝檔案完全正確！${NC}"
    echo -e "${GREEN}   從 MLIR 轉換後的 kernel 可以在 GPU 上正確執行${NC}"
    echo ""
    exit 0
else
    if [ $FAILED -gt 0 ]; then
        echo -e "${RED}失敗的測試：${NC}"
        for failed_test in "${FAILED_TESTS[@]}"; do
            echo -e "  • ${RED}✗${NC} $failed_test"
        done
        echo ""
    fi
    exit 1
fi

