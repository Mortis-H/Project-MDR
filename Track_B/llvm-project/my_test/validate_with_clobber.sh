#!/bin/bash
# 完整驗證：ISA → GPU MLIR（添加 clobber）→ ISA（LLVM 計算 metadata）→ 執行測試

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 測試配置
declare -A TESTS=(
    ["test_01_vector_add"]="float_add:1024"
    ["test_02_scalar_ops"]="int_scalar:256"
    ["test_03_memory_ops"]="int_mem:512"
    ["test_04_conditional"]="int_cond:512"
    ["test_05_loop"]="int_loop:256"
    ["test_06_shared_memory"]="int_shared:512"
)

echo "========================================="
echo "Clobber-Based Validation"
echo "目標：證明 Register Clobber 能讓 LLVM 正確計算 metadata"
echo "========================================="
echo ""

PASS_COUNT=0
FAIL_COUNT=0

for test_name in test_01_vector_add test_02_scalar_ops test_03_memory_ops test_04_conditional test_05_loop test_06_shared_memory; do
    IFS=':' read -r kernel_type test_size <<< "${TESTS[$test_name]}"
    
    echo "========================================="
    echo "Testing: $test_name"
    echo "========================================="
    
    original_s="../../kernel_testcases/$test_name/original.s"
    
    if [ ! -f "$original_s" ]; then
        echo "❌ SKIP: $original_s not found"
        echo ""
        continue
    fi
    
    # Step 1: ISA → GPU MLIR
    echo "[1/5] ISA → GPU MLIR..."
    python3 pipeline.py "$original_s" \
        --workdir "clobber_${test_name}/stage1" \
        --output-prefix "${test_name}" \
        --no-emit-isa > /dev/null 2>&1
    
    gpumlir_file="clobber_${test_name}/stage1/${test_name}.gpumlir"
    
    # Step 2: 添加 Register Clobber
    echo "[2/5] Adding register clobber..."
    python3 add_register_clobber_v2.py "$gpumlir_file" \
        -o "clobber_${test_name}/${test_name}_with_clobber.gpumlir" 2>&1 | grep "Register usage"
    
    # Step 3: GPU MLIR (with clobber) → ISA (LLVM 計算 metadata)
    echo "[3/5] GPU MLIR → ISA (trusting LLVM for resource counts)..."
    python3 pipeline.py "clobber_${test_name}/${test_name}_with_clobber.gpumlir" \
        --workdir "clobber_${test_name}/stage2" \
        --output-prefix "${test_name}_final" \
        --trust-llvm-resources 2>&1 | grep -E "(trusting LLVM|Fixed ISA)" || true
    
    final_s="clobber_${test_name}/stage2/${test_name}_final.s"
    final_hsaco="clobber_${test_name}/stage2/${test_name}_final.hsaco"
    
    # Step 4: 比較 Metadata
    echo "[4/5] Comparing metadata..."
    
    orig_vgpr=$(grep "\.vgpr_count:" "$original_s" | awk '{print $2}')
    orig_sgpr=$(grep "\.sgpr_count:" "$original_s" | awk '{print $2}')
    orig_kernarg=$(grep "kernarg_segment_size:" "$original_s" | head -1 | awk '{print $2}')
    
    final_vgpr=$(grep "\.vgpr_count:" "$final_s" | awk '{print $2}')
    final_sgpr=$(grep "\.sgpr_count:" "$final_s" | awk '{print $2}')
    final_kernarg=$(grep "kernarg_segment_size:" "$final_s" | head -1 | awk '{print $2}')
    
    metadata_match=true
    
    printf "   Original: VGPR=%s, SGPR=%s, kernarg=%s\n" "$orig_vgpr" "$orig_sgpr" "$orig_kernarg"
    printf "   Final:    VGPR=%s, SGPR=%s, kernarg=%s\n" "$final_vgpr" "$final_sgpr" "$final_kernarg"
    
    if [ "$orig_vgpr" != "$final_vgpr" ]; then
        echo "   ⚠️  VGPR mismatch!"
        metadata_match=false
    fi
    
    if [ "$orig_sgpr" != "$final_sgpr" ]; then
        echo "   ⚠️  SGPR mismatch!"
        metadata_match=false
    fi
    
    if [ "$orig_kernarg" != "$final_kernarg" ]; then
        echo "   ⚠️  Kernarg mismatch!"
        metadata_match=false
    fi
    
    if [ "$metadata_match" = true ]; then
        echo "   ✅ Metadata matches!"
    fi
    
    # Step 5: 執行驗證
    echo "[5/5] Executing kernel..."
    
    kernel_name=$(readelf -W -s "$final_hsaco" | grep " FUNC " | grep -v "\.kd" | awk '{print $8}' | head -1)
    echo "   Kernel: $kernel_name"
    
    cd ../../kernel_testcases
    
    ./universal_hsaco_runner \
        "../llvm-project/my_test/$final_hsaco" \
        "$kernel_name" \
        "$kernel_type" \
        "$test_size" > /tmp/clobber_${test_name}.log 2>&1
    
    exec_result="UNKNOWN"
    if grep -q "✅ PASS" /tmp/clobber_${test_name}.log; then
        exec_result="PASS"
    elif grep -q "❌ FAIL" /tmp/clobber_${test_name}.log; then
        exec_result="FAIL"
    fi
    
    cd "$SCRIPT_DIR"
    
    echo "   Execution: $exec_result"
    echo ""
    
    # 判斷最終結果
    if [ "$metadata_match" = true ] && [ "$exec_result" = "PASS" ]; then
        echo "✅ PASS: $test_name"
        echo "   證明：Register Clobber 讓 LLVM 正確計算了 metadata！"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "❌ FAIL: $test_name"
        if [ "$metadata_match" = false ]; then
            echo "   原因：Metadata 不匹配"
        fi
        if [ "$exec_result" != "PASS" ]; then
            echo "   原因：執行失敗"
        fi
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo ""
done

echo "========================================="
echo "Summary"
echo "========================================="
echo "✅ Passed: $PASS_COUNT / 6"
echo "❌ Failed: $FAIL_COUNT / 6"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 All tests passed!"
    echo ""
    echo "✅ 核心假設得到驗證："
    echo "   Register Clobber + 信任 LLVM = 正確的 Metadata"
    echo "   不再需要 fix_isa_metadata()！"
    exit 0
else
    echo "⚠️  Some tests failed"
    exit 1
fi

