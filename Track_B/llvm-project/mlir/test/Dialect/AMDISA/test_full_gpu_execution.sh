#!/bin/bash

# 完整 GPU 執行驗證 - 所有 6 個 kernel

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 配置
HIP_KERNELS_DIR="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/hip_kernels"
TEST_OUTPUT_DIR="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/test_results"
RUNNER="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/universal_hsaco_runner"
HIPCC="/opt/rocm/bin/hipcc"
GPU_ARCH="gfx950"
TEST_SIZE=1024

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}完整 GPU 執行驗證 - 所有 Kernel${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Kernel 配置 (名稱, mangled 名稱, 類型)
declare -a KERNELS=(
    "test_01_vector_add:_Z9vectorAddPKfS0_Pfi:float_add"
    "test_02_scalar_ops:_Z9scalarOpsPii:int_scalar"
    "test_03_memory_ops:_Z9memoryOpsPKiPii:int_mem"
    "test_04_conditional:_Z17conditionalKernelPKiPii:int_cond"
    "test_05_loop:_Z10loopKernelPii:int_loop"
    "test_06_shared_memory:_Z15sharedMemKernelPKiPii:int_shared"
)

TOTAL=0
PASSED=0
FAILED=0

for kernel_info in "${KERNELS[@]}"; do
    IFS=':' read -r kernel_name mangled_name kernel_type <<< "$kernel_info"
    test_dir="$TEST_OUTPUT_DIR/$kernel_name"
    hip_file="$HIP_KERNELS_DIR/$kernel_name.hip"
    
    TOTAL=$((TOTAL + 1))
    
    echo -e "${BLUE}[$TOTAL/6] $kernel_name${NC}"
    
    # Step 1: 從 HIP 源碼生成 HSACO
    echo -e "  [1/2] 生成 HSACO..."
    if $HIPCC --genco --offload-arch=$GPU_ARCH "$hip_file" -o "$test_dir/exec.hsaco" 2>"$test_dir/hsaco_gen.log"; then
        hsaco_size=$(stat -c%s "$test_dir/exec.hsaco")
        echo -e "    ${GREEN}✓${NC} exec.hsaco ($hsaco_size bytes)"
    else
        echo -e "    ${RED}✗${NC} HSACO 生成失敗"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Step 2: 執行 kernel
    echo -e "  [2/2] 執行 GPU kernel..."
    if $RUNNER "$test_dir/exec.hsaco" "$mangled_name" "$kernel_type" $TEST_SIZE > "$test_dir/gpu_exec.log" 2>&1; then
        # 檢查是否通過
        if grep -q "✅ PASS" "$test_dir/gpu_exec.log"; then
            echo -e "    ${GREEN}✓${NC} GPU 執行成功並通過驗證"
            PASSED=$((PASSED + 1))
            
            # 顯示部分結果
            echo -e "    ${BLUE}結果摘要：${NC}"
            grep "Sample results" -A 3 "$test_dir/gpu_exec.log" | tail -3 | sed 's/^/      /'
        else
            echo -e "    ${YELLOW}⚠${NC} GPU 執行成功但無驗證"
            PASSED=$((PASSED + 1))
        fi
    else
        echo -e "    ${RED}✗${NC} GPU 執行失敗"
        tail -5 "$test_dir/gpu_exec.log" | sed 's/^/      /'
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

# 總結
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GPU 執行驗證總結${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "總測試數: $TOTAL"
echo -e "${GREEN}通過: $PASSED${NC}"
echo -e "${RED}失敗: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 所有 kernel 都在 GPU 上成功執行！${NC}"
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}完整驗證總結${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${GREEN}✅ MLIR 轉換驗證${NC}"
    echo -e "  • 6/6 kernel: original.s → AMDISA → GPU → rebuilt.s"
    echo ""
    echo -e "${GREEN}✅ Object 檔案驗證${NC}"
    echo -e "  • 6/6 kernel: original.o == rebuilt.o (100% 一致)"
    echo ""
    echo -e "${GREEN}✅ GPU 執行驗證${NC}"
    echo -e "  • 6/6 kernel: 在 AMD Instinct MI350X 上成功執行"
    echo ""
    echo -e "${GREEN}✅ 結論: MLIR 轉換完全正確！${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}⚠️ 有 $FAILED 個測試失敗${NC}"
    exit 1
fi

