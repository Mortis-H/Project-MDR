#!/bin/bash
# Universal Runner 模式測試 - 測試所有 hip_kernels
# 使用 universal_hsaco_runner，不需要 host source

set -e  # 遇到錯誤立即停止

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

echo -e "${MAGENTA}============================================================${NC}"
echo -e "${MAGENTA}Pipeline 正確性測試 - Universal Runner 模式${NC}"
echo -e "${MAGENTA}測試所有 hip_kernels 目錄中的 Kernel${NC}"
echo -e "${MAGENTA}============================================================${NC}"
echo ""

# 通用配置
TEST_SIZE=1024
ARCH="gfx950"
RUNNER="../../kernel_testcases/universal_hsaco_runner"
KERNEL_DIR="hip_kernels"

# Kernel 配置 (檔案名稱:mangled 名稱:類型)
declare -a TESTS=(
    "test_01_vector_add.hip:_Z9vectorAddPKfS0_Pfi:float_add"
    "test_02_scalar_ops.hip:_Z9scalarOpsPii:int_scalar"
    "test_03_memory_ops.hip:_Z9memoryOpsPKiPii:int_mem"
    "test_04_conditional.hip:_Z17conditionalKernelPKiPii:int_cond"
    "test_05_loop.hip:_Z10loopKernelPii:int_loop"
    "test_06_shared_memory.hip:_Z15sharedMemKernelPKiPii:int_shared"
)

# 檢查 universal_hsaco_runner 是否存在
if [ ! -f "$RUNNER" ]; then
    echo -e "${YELLOW}⚠️  universal_hsaco_runner 不存在，嘗試編譯...${NC}"
    echo ""
    
    RUNNER_DIR="../mlir/test/Dialect/AMDISA"
    if [ -d "$RUNNER_DIR" ]; then
        cd "$RUNNER_DIR"
        echo -e "${BLUE}執行 make 編譯 universal_hsaco_runner...${NC}"
        make
        cd - > /dev/null
        echo ""
        
        if [ -f "$RUNNER" ]; then
            echo -e "${GREEN}✓ universal_hsaco_runner 編譯成功${NC}"
        else
            echo -e "${RED}✗ 編譯失敗，請手動編譯${NC}"
            exit 1
        fi
    else
        echo -e "${RED}錯誤: 找不到 universal_hsaco_runner 目錄: $RUNNER_DIR${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ 找到 universal_hsaco_runner: $RUNNER${NC}"

# 檢查 kernel 目錄是否存在
if [ ! -d "$KERNEL_DIR" ]; then
    echo -e "${RED}錯誤: Kernel 目錄不存在: $KERNEL_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 找到 Kernel 目錄: $KERNEL_DIR${NC}"
echo ""

# 顯示測試配置
echo -e "${CYAN}測試配置:${NC}"
echo -e "  模式:        Universal Runner（不需要 host source）"
echo -e "  Test Size:   $TEST_SIZE"
echo -e "  架構:        $ARCH"
echo -e "  Kernel 目錄: $KERNEL_DIR"
echo -e "  總測試數:    ${#TESTS[@]}"
echo ""

# 計數器
TOTAL=0
PASSED=0
FAILED=0
declare -a FAILED_TESTS

# 循環測試每個 kernel
for test_info in "${TESTS[@]}"; do
    IFS=':' read -r kernel_file kernel_name kernel_type <<< "$test_info"
    kernel_path="$KERNEL_DIR/$kernel_file"
    test_name="${kernel_file%.hip}"
    workdir="output/${test_name}"
    
    TOTAL=$((TOTAL + 1))
    
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}[$TOTAL/${#TESTS[@]}] 測試: $test_name${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo -e "  Kernel:      $kernel_path"
    echo -e "  Kernel Name: $kernel_name"
    echo -e "  Kernel Type: $kernel_type"
    echo -e "  工作目錄:    $workdir"
    echo ""
    
    # 檢查 kernel 文件是否存在
    if [ ! -f "$kernel_path" ]; then
        echo -e "${RED}✗ Kernel 文件不存在: $kernel_path${NC}"
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$test_name (文件不存在)")
        echo ""
        continue
    fi
    
    # 執行測試
    if python3 test_pipeline_correctness.py \
        --use-universal-runner \
        --kernel "$kernel_path" \
        --runner "$RUNNER" \
        --kernel-name "$kernel_name" \
        --kernel-type "$kernel_type" \
        --test-size "$TEST_SIZE" \
        --arch "$ARCH" \
        --workdir "$workdir"; then
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
echo -e "${MAGENTA}============================================================${NC}"
echo -e "${MAGENTA}測試總結${NC}"
echo -e "${MAGENTA}============================================================${NC}"
echo ""
echo -e "總測試數: ${BLUE}$TOTAL${NC}"
echo -e "通過: ${GREEN}$PASSED${NC}"
echo -e "失敗: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}🎉 所有測試通過！${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo ""
    echo -e "${GREEN}驗證結果：${NC}"
    echo -e "  • ${GREEN}✓${NC} 6/6 kernel: 原始 .s → AMDISA dialect"
    echo -e "  • ${GREEN}✓${NC} 6/6 kernel: AMDISA → GPU dialect"
    echo -e "  • ${GREEN}✓${NC} 6/6 kernel: GPU → 重建 .s"
    echo -e "  • ${GREEN}✓${NC} 6/6 kernel: 重建 .s → .o → .hsaco"
    echo -e "  • ${GREEN}✓${NC} 6/6 kernel: Pipeline 轉換結果正確"
    echo ""
    echo -e "${GREEN}✅ 結論: Pipeline 轉換完全正確！${NC}"
    echo -e "${GREEN}   所有 kernel 轉換後執行結果與原始版本一致${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}⚠️  有 $FAILED 個測試失敗${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    echo -e "${RED}失敗的測試：${NC}"
    for failed_test in "${FAILED_TESTS[@]}"; do
        echo -e "  • ${RED}✗${NC} $failed_test"
    done
    echo ""
    exit 1
fi

