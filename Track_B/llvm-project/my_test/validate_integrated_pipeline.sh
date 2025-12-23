#!/bin/bash
# 驗證新整合的 pipeline.py（自動 Register Clobber）
# 測試 test_01 到 test_06，比較前後結果

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

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     新整合 Pipeline 驗證（自動 Register Clobber）            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

for test_name in test_01_vector_add test_02_scalar_ops test_03_memory_ops test_04_conditional test_05_loop test_06_shared_memory; do
    IFS=':' read -r kernel_type test_size <<< "${TESTS[$test_name]}"
    
    echo "═══════════════════════════════════════════════════════════════"
    echo "Testing: $test_name"
    echo "═══════════════════════════════════════════════════════════════"
    
    original_s="../../kernel_testcases/$test_name/original.s"
    
    if [ ! -f "$original_s" ]; then
        echo "❌ SKIP: $original_s not found"
        echo ""
        continue
    fi
    
    # Step 1: 使用新的 pipeline.py（一步完成，自動添加 clobber）
    echo "[1/4] Running integrated pipeline (ISA → MLIR with clobber → ISA)..."
    mkdir -p "integrated_${test_name}"
    python3 pipeline.py "$original_s" \
        --workdir "integrated_${test_name}" \
        --output-prefix "${test_name}_new" > "integrated_${test_name}/pipeline.log" 2>&1
    
    rebuilt_s="integrated_${test_name}/${test_name}_new.s"
    rebuilt_hsaco="integrated_${test_name}/${test_name}_new.hsaco"
    
    # Step 2: 比較 Metadata
    echo "[2/4] Comparing metadata..."
    
    # 提取 metadata
    orig_vgpr=$(grep "\.vgpr_count:" "$original_s" | awk '{print $2}')
    orig_sgpr=$(grep "\.sgpr_count:" "$original_s" | awk '{print $2}')
    orig_kernarg=$(grep "kernarg_segment_size:" "$original_s" | head -1 | awk '{print $2}')
    
    new_vgpr=$(grep "\.vgpr_count:" "$rebuilt_s" | awk '{print $2}')
    new_sgpr=$(grep "\.sgpr_count:" "$rebuilt_s" | awk '{print $2}')
    new_kernarg=$(grep "kernarg_segment_size:" "$rebuilt_s" | head -1 | awk '{print $2}')
    
    metadata_match=true
    
    printf "   Original: VGPR=%s, SGPR=%s, kernarg=%s\n" "$orig_vgpr" "$orig_sgpr" "$orig_kernarg"
    printf "   Rebuilt:  VGPR=%s, SGPR=%s, kernarg=%s\n" "$new_vgpr" "$new_sgpr" "$new_kernarg"
    
    if [ "$orig_vgpr" != "$new_vgpr" ]; then
        echo "   ⚠️  VGPR count mismatch!"
        metadata_match=false
    fi
    
    if [ "$orig_sgpr" != "$new_sgpr" ]; then
        echo "   ⚠️  SGPR count mismatch!"
        metadata_match=false
    fi
    
    if [ "$orig_kernarg" != "$new_kernarg" ]; then
        echo "   ⚠️  Kernarg size mismatch!"
        metadata_match=false
    fi
    
    if [ "$metadata_match" = true ]; then
        echo "   ✅ Metadata matches!"
    fi
    
    # Step 3: 檢測 kernel 名稱
    echo "[3/4] Detecting kernel name..."
    kernel_name=$(readelf -W -s "$rebuilt_hsaco" | grep " FUNC " | grep -v "\.kd" | awk '{print $8}' | head -1)
    echo "   Kernel: $kernel_name"
    
    # Step 4: 執行驗證
    echo "[4/4] Executing kernel..."
    
    cd ../../kernel_testcases
    
    ./universal_hsaco_runner \
        "../llvm-project/my_test/$rebuilt_hsaco" \
        "$kernel_name" \
        "$kernel_type" \
        "$test_size" > /tmp/integrated_${test_name}.log 2>&1
    
    exec_result="UNKNOWN"
    if grep -q "✅ PASS" /tmp/integrated_${test_name}.log; then
        exec_result="PASS"
    elif grep -q "❌ FAIL" /tmp/integrated_${test_name}.log; then
        exec_result="FAIL"
    fi
    
    cd "$SCRIPT_DIR"
    
    echo "   Execution: $exec_result"
    echo ""
    
    # 判斷最終結果
    if [ "$metadata_match" = true ] && [ "$exec_result" = "PASS" ]; then
        echo "✅ PASS: $test_name"
        echo "   └─ 自動 Register Clobber + LLVM 計算成功！"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "❌ FAIL: $test_name"
        if [ "$metadata_match" = false ]; then
            echo "   └─ Metadata 不匹配"
        fi
        if [ "$exec_result" != "PASS" ]; then
            echo "   └─ 執行失敗"
        fi
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
echo "Summary"
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Passed: $PASS_COUNT / 6"
echo "❌ Failed: $FAIL_COUNT / 6"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "🎉 All tests passed with integrated pipeline!"
    echo ""
    echo "✅ 核心驗證成功："
    echo "   • Pipeline 自動添加 Register Clobber"
    echo "   • LLVM 正確計算 metadata"
    echo "   • 一個命令完成所有操作"
    echo "   • 無需手動干預"
    exit 0
else
    echo "⚠️  Some tests failed"
    exit 1
fi

