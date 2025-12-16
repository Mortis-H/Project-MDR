#!/bin/bash

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
HIP_KERNELS_DIR="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/hip_kernels"
TEST_OUTPUT_DIR="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/test_results"
AMDISA_TRANSLATE="/home/morhuang/Project-MDR/Track_B/llvm-project/build/bin/amdisa-translate"
HIPCC="/opt/rocm/bin/hipcc"
CLANG="/opt/rocm/llvm/bin/clang"
EXTRACT_SCRIPT="/home/morhuang/Project-MDR/Track_B/llvm-project/mlir/test/Dialect/AMDISA/extract_device_asm.sh"

# 創建測試結果目錄
mkdir -p "$TEST_OUTPUT_DIR"

# 清理舊結果
rm -rf "$TEST_OUTPUT_DIR"/*

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AMDISA Dialect - 全面測試${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 統計變數
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 找到所有 .hip 文件
HIP_FILES=($(ls "$HIP_KERNELS_DIR"/*.hip 2>/dev/null || true))

if [ ${#HIP_FILES[@]} -eq 0 ]; then
    echo -e "${RED}錯誤：找不到任何 .hip 文件${NC}"
    exit 1
fi

echo -e "${GREEN}找到 ${#HIP_FILES[@]} 個 kernel 文件${NC}"
echo ""

# 為每個 kernel 執行測試
for hip_file in "${HIP_FILES[@]}"; do
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # 提取 kernel 名稱
    kernel_name=$(basename "$hip_file" .hip)
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}測試 $TOTAL_TESTS: $kernel_name${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # 創建測試目錄
    test_dir="$TEST_OUTPUT_DIR/$kernel_name"
    mkdir -p "$test_dir"
    
    # 記錄開始時間
    start_time=$(date +%s)
    
    # 初始化測試狀態
    TEST_PASSED=true
    ERROR_MSG=""
    
    # Step 1: 編譯 HIP -> Assembly (with offload bundle)
    echo -e "${YELLOW}[1/7]${NC} 編譯 HIP 到 Assembly..."
    if $HIPCC -S --offload-arch=gfx950 "$hip_file" -o "$test_dir/bundled.s" 2>"$test_dir/hipcc.log"; then
        echo -e "  ${GREEN}✓${NC} 成功生成 bundled.s ($(wc -l < "$test_dir/bundled.s") 行)"
    else
        echo -e "  ${RED}✗${NC} 編譯失敗"
        TEST_PASSED=false
        ERROR_MSG="HIP 編譯失敗"
    fi
    
    if [ "$TEST_PASSED" = true ]; then
        # Step 2: 提取 device assembly
        echo -e "${YELLOW}[2/7]${NC} 提取 device assembly..."
        if $EXTRACT_SCRIPT "$test_dir/bundled.s" "$test_dir/original.s" 2>"$test_dir/extract.log"; then
            echo -e "  ${GREEN}✓${NC} 成功提取 original.s ($(wc -l < "$test_dir/original.s") 行)"
        else
            echo -e "  ${RED}✗${NC} 提取失敗"
            TEST_PASSED=false
            ERROR_MSG="提取 device assembly 失敗"
        fi
    fi
    
    if [ "$TEST_PASSED" = true ]; then
        # Step 3: Assembly -> AMDISA MLIR
        echo -e "${YELLOW}[3/7]${NC} 解析 Assembly 到 AMDISA MLIR..."
        if $AMDISA_TRANSLATE -x s -emit mlir "$test_dir/original.s" > "$test_dir/stage1_amdisa.mlir" 2>"$test_dir/stage1.log"; then
            echo -e "  ${GREEN}✓${NC} 成功生成 amdisa.mlir ($(wc -l < "$test_dir/stage1_amdisa.mlir") 行)"
        else
            echo -e "  ${RED}✗${NC} 解析失敗"
            TEST_PASSED=false
            ERROR_MSG="Assembly -> AMDISA MLIR 失敗"
        fi
    fi
    
    if [ "$TEST_PASSED" = true ]; then
        # Step 4: AMDISA MLIR -> GPU MLIR
        echo -e "${YELLOW}[4/7]${NC} 降級到 GPU Inline ASM..."
        if $AMDISA_TRANSLATE -x mlir -emit gpuinlineasm "$test_dir/stage1_amdisa.mlir" > "$test_dir/stage2_gpu.mlir" 2>"$test_dir/stage2.log"; then
            echo -e "  ${GREEN}✓${NC} 成功生成 gpu.mlir ($(wc -l < "$test_dir/stage2_gpu.mlir") 行)"
        else
            echo -e "  ${RED}✗${NC} 降級失敗"
            TEST_PASSED=false
            ERROR_MSG="AMDISA MLIR -> GPU MLIR 失敗"
        fi
    fi
    
    if [ "$TEST_PASSED" = true ]; then
        # Step 5: GPU MLIR -> Rebuilt Assembly
        echo -e "${YELLOW}[5/7]${NC} 重建 Assembly..."
        if $AMDISA_TRANSLATE -x mlir -emit s "$test_dir/stage2_gpu.mlir" > "$test_dir/stage3_rebuilt.s" 2>"$test_dir/stage3.log"; then
            echo -e "  ${GREEN}✓${NC} 成功生成 rebuilt.s ($(wc -l < "$test_dir/stage3_rebuilt.s") 行)"
        else
            echo -e "  ${RED}✗${NC} 重建失敗"
            TEST_PASSED=false
            ERROR_MSG="GPU MLIR -> Assembly 失敗"
        fi
    fi
    
    if [ "$TEST_PASSED" = true ]; then
        # Step 6: 編譯驗證 - Original
        echo -e "${YELLOW}[6/7]${NC} 編譯驗證 - Original..."
        if $CLANG -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 "$test_dir/original.s" -o "$test_dir/original.o" 2>"$test_dir/original_compile.log"; then
            orig_size=$(stat -c%s "$test_dir/original.o")
            echo -e "  ${GREEN}✓${NC} original.o ($orig_size bytes)"
        else
            echo -e "  ${RED}✗${NC} 編譯失敗"
            TEST_PASSED=false
            ERROR_MSG="original.s 編譯失敗"
        fi
    fi
    
    if [ "$TEST_PASSED" = true ]; then
        # Step 7: 編譯驗證 - Rebuilt
        echo -e "${YELLOW}[7/7]${NC} 編譯驗證 - Rebuilt..."
        if $CLANG -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 "$test_dir/stage3_rebuilt.s" -o "$test_dir/rebuilt.o" 2>"$test_dir/rebuilt_compile.log"; then
            rebuilt_size=$(stat -c%s "$test_dir/rebuilt.o")
            echo -e "  ${GREEN}✓${NC} rebuilt.o ($rebuilt_size bytes)"
            
            # 比較大小
            if [ "$orig_size" -eq "$rebuilt_size" ]; then
                echo -e "  ${GREEN}✓${NC} 檔案大小一致！"
            else
                echo -e "  ${YELLOW}⚠${NC} 檔案大小不同 (原始: $orig_size, 重建: $rebuilt_size, 差異: $((orig_size - rebuilt_size)))"
            fi
        else
            echo -e "  ${RED}✗${NC} 編譯失敗"
            TEST_PASSED=false
            ERROR_MSG="rebuilt.s 編譯失敗"
        fi
    fi
    
    # 計算耗時
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    # 生成測試報告
    cat > "$test_dir/TEST_REPORT.md" << EOF
# $kernel_name 測試報告

## 測試日期
$(date '+%Y-%m-%d %H:%M:%S')

## 測試結果
EOF
    
    if [ "$TEST_PASSED" = true ]; then
        echo -e "\n${GREEN}✅ 測試通過${NC} (耗時: ${elapsed}s)\n"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        
        cat >> "$test_dir/TEST_REPORT.md" << EOF
**狀態**: ✅ 通過

## Pipeline 流程

\`\`\`
bundled.s ($(wc -l < "$test_dir/bundled.s") 行) - hipcc 輸出
    ↓ extract device assembly
original.s ($(wc -l < "$test_dir/original.s") 行)
    ↓ amdisa-translate -x s -emit mlir
stage1_amdisa.mlir ($(wc -l < "$test_dir/stage1_amdisa.mlir") 行)
    ↓ amdisa-translate -x mlir -emit gpuinlineasm
stage2_gpu.mlir ($(wc -l < "$test_dir/stage2_gpu.mlir") 行)
    ↓ amdisa-translate -x mlir -emit s
stage3_rebuilt.s ($(wc -l < "$test_dir/stage3_rebuilt.s") 行)
\`\`\`

## 編譯驗證

| 檔案 | 狀態 | 大小 |
|------|------|------|
| original.o | ✅ | $orig_size bytes |
| rebuilt.o | ✅ | $rebuilt_size bytes |

**結果**: 檔案大小$([ "$orig_size" -eq "$rebuilt_size" ] && echo "完全一致" || echo "不同 (差異: $((orig_size - rebuilt_size)) bytes)")

## 耗時
${elapsed} 秒
EOF
    else
        echo -e "\n${RED}❌ 測試失敗${NC}: $ERROR_MSG (耗時: ${elapsed}s)\n"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        
        cat >> "$test_dir/TEST_REPORT.md" << EOF
**狀態**: ❌ 失敗

## 錯誤信息
$ERROR_MSG

## 耗時
${elapsed} 秒

## 日誌文件
請查看以下文件以獲取詳細錯誤信息：
- hipcc.log
- extract.log
- stage1.log
- stage2.log
- stage3.log
- original_compile.log
- rebuilt_compile.log
EOF
    fi
done

# 生成總體報告
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}測試總結${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "總測試數: $TOTAL_TESTS"
echo -e "${GREEN}通過: $PASSED_TESTS${NC}"
echo -e "${RED}失敗: $FAILED_TESTS${NC}"
echo ""

# 生成總體報告文件
cat > "$TEST_OUTPUT_DIR/SUMMARY.md" << EOF
# AMDISA Dialect 全面測試總結

## 測試日期
$(date '+%Y-%m-%d %H:%M:%S')

## 測試統計

| 項目 | 數量 |
|------|------|
| 總測試數 | $TOTAL_TESTS |
| 通過 | $PASSED_TESTS |
| 失敗 | $FAILED_TESTS |
| 成功率 | $(awk "BEGIN {printf \"%.1f\", ($PASSED_TESTS/$TOTAL_TESTS)*100}")% |

## 測試詳情

EOF

# 為每個 kernel 添加詳情
for hip_file in "${HIP_FILES[@]}"; do
    kernel_name=$(basename "$hip_file" .hip")
    test_dir="$TEST_OUTPUT_DIR/$kernel_name"
    
    if [ -f "$test_dir/original.o" ] && [ -f "$test_dir/rebuilt.o" ]; then
        orig_size=$(stat -c%s "$test_dir/original.o")
        rebuilt_size=$(stat -c%s "$test_dir/rebuilt.o")
        if [ "$orig_size" -eq "$rebuilt_size" ]; then
            status="✅"
            result="通過 (大小一致)"
        else
            status="⚠️"
            result="通過 (大小不同: $((orig_size - rebuilt_size)) bytes)"
        fi
    else
        orig_size="N/A"
        rebuilt_size="N/A"
        status="❌"
        result="失敗"
    fi
    
    cat >> "$TEST_OUTPUT_DIR/SUMMARY.md" << EOF
### $kernel_name
- **狀態**: $status $result
- **原始 .o**: $orig_size bytes
- **重建 .o**: $rebuilt_size bytes
- **詳細報告**: [$kernel_name/TEST_REPORT.md]($kernel_name/TEST_REPORT.md)

EOF
done

cat >> "$TEST_OUTPUT_DIR/SUMMARY.md" << EOF

## 測試環境

- **LLVM/MLIR**: Track_B/llvm-project
- **GPU 架構**: AMD GCN (gfx950)
- **工具**: amdisa-translate, hipcc, clang

## Pipeline 說明

1. **HIP 編譯**: 使用 hipcc 將 .hip 文件編譯成 .s (包含 offload bundle)
2. **提取 device assembly**: 從 bundle 中提取 AMD GPU assembly
3. **解析到 MLIR**: 使用 amdisa-translate 解析成 AMDISA Dialect
4. **降級**: 將 AMDISA 降級到 GPU Inline ASM
5. **重建**: 從 GPU MLIR 重建出完整的 .s 文件
6. **編譯驗證**: 使用 clang 編譯 original 和 rebuilt，比較結果

## 結論

$(if [ $FAILED_TESTS -eq 0 ]; then
    echo "🎉 **所有測試通過！** AMDISA Dialect 對所有測試 kernel 均工作正常。"
else
    echo "⚠️ **部分測試失敗**，請查看各個 kernel 的詳細報告。"
fi)
EOF

echo -e "${GREEN}總體報告已生成: $TEST_OUTPUT_DIR/SUMMARY.md${NC}"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}🎉 所有測試通過！${NC}"
    exit 0
else
    echo -e "${RED}⚠️ 有 $FAILED_TESTS 個測試失敗${NC}"
    exit 1
fi

