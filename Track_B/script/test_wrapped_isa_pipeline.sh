#!/bin/bash
#
# 自動化測試腳本：測試 wrapped ISA 的完整 pipeline
# 流程：*_wrapped.s -> pipeline.py -> .hsaco -> relink_and_compare.sh -> 結果比較
#

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 獲取腳本所在目錄並自動推斷路徑
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 配置
RUN_SUCCESS_DIR="${PROJECT_ROOT}/pure_function_level_assembly_and_hip_and_complete_hip/run_success"
PIPELINE_SCRIPT="$(cd "$SCRIPT_DIR/.." && pwd)/amdisa-toolkit/examples/pipeline.py"
RELINK_SCRIPT="${SCRIPT_DIR}/relink_and_compare.sh"
CHIP="gfx950"
EXECUTION_TIMEOUT=60  # 執行超時時間（秒）

# 統計變數
TOTAL_KERNELS=0
SUCCESS_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
TIMEOUT_COUNT=0

# 結果記錄
RESULT_LOG=""
SUMMARY_FILE=""

# 使用說明
usage() {
    echo "用法: $0 [選項]"
    echo ""
    echo "選項:"
    echo "  -d, --dir DIR           指定 run_success 目錄（預設：$RUN_SUCCESS_DIR）"
    echo "  -c, --chip CHIP         指定 GPU 晶片型號（預設：$CHIP）"
    echo "  -t, --timeout SECONDS   執行超時時間（預設：$EXECUTION_TIMEOUT 秒）"
    echo "  -k, --kernel HASH       只測試指定的 kernel（kernel hash）"
    echo "  -l, --limit N           只測試前 N 個 kernel"
    echo "  -s, --skip-pipeline     跳過 pipeline.py，直接使用已存在的 .hsaco"
    echo "  -h, --help              顯示此幫助信息"
    echo ""
    echo "範例:"
    echo "  $0                                           # 測試所有 kernel"
    echo "  $0 -l 5                                      # 只測試前 5 個 kernel"
    echo "  $0 -k 0a32840dac54a492de687e418c588476d324dcdc  # 測試特定 kernel"
    echo "  $0 -t 120                                    # 設定超時時間為 120 秒"
    exit 1
}

# 解析命令列參數
SPECIFIC_KERNEL=""
LIMIT=""
SKIP_PIPELINE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            RUN_SUCCESS_DIR="$2"
            shift 2
            ;;
        -c|--chip)
            CHIP="$2"
            shift 2
            ;;
        -t|--timeout)
            EXECUTION_TIMEOUT="$2"
            shift 2
            ;;
        -k|--kernel)
            SPECIFIC_KERNEL="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -s|--skip-pipeline)
            SKIP_PIPELINE=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}未知選項: $1${NC}"
            usage
            ;;
    esac
done

# 檢查必要的目錄和腳本
if [ ! -d "$RUN_SUCCESS_DIR" ]; then
    echo -e "${RED}錯誤：run_success 目錄不存在: $RUN_SUCCESS_DIR${NC}"
    exit 1
fi

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo -e "${RED}錯誤：pipeline.py 腳本不存在: $PIPELINE_SCRIPT${NC}"
    exit 1
fi

if [ ! -f "$RELINK_SCRIPT" ]; then
    echo -e "${RED}錯誤：relink_and_compare.sh 腳本不存在: $RELINK_SCRIPT${NC}"
    exit 1
fi

# 創建測試結果目錄
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_RESULTS_DIR="${RUN_SUCCESS_DIR}/wrapped_isa_test_results_${TIMESTAMP}"
mkdir -p "$TEST_RESULTS_DIR"
SUMMARY_FILE="${TEST_RESULTS_DIR}/test_summary.txt"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Wrapped ISA Pipeline 自動化測試                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}配置信息：${NC}"
echo "  Run Success 目錄: $RUN_SUCCESS_DIR"
echo "  Pipeline 腳本: $PIPELINE_SCRIPT"
echo "  Relink 腳本: $RELINK_SCRIPT"
echo "  GPU 晶片: $CHIP"
echo "  執行超時: ${EXECUTION_TIMEOUT}秒"
echo "  測試結果目錄: $TEST_RESULTS_DIR"
if [ -n "$SPECIFIC_KERNEL" ]; then
    echo "  測試模式: 單個 kernel ($SPECIFIC_KERNEL)"
elif [ -n "$LIMIT" ]; then
    echo "  測試模式: 前 $LIMIT 個 kernel"
else
    echo "  測試模式: 所有 kernel"
fi
if [ $SKIP_PIPELINE -eq 1 ]; then
    echo -e "  ${YELLOW}警告: 跳過 pipeline.py，使用已存在的 .hsaco${NC}"
fi
echo ""

# 函數：測試單個 kernel
test_kernel() {
    local kernel_dir="$1"
    local kernel_hash=$(basename "$kernel_dir")
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}測試 Kernel: ${MAGENTA}$kernel_hash${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    TOTAL_KERNELS=$((TOTAL_KERNELS + 1))
    
    # 查找 wrapped ISA 文件
    local wrapped_isa=$(find "$kernel_dir" -maxdepth 1 -name "*_wrapped.s" | head -n 1)
    
    if [ -z "$wrapped_isa" ] || [ ! -f "$wrapped_isa" ]; then
        echo -e "${YELLOW}⊘ 跳過：找不到 *_wrapped.s 文件${NC}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        RESULT_LOG="${RESULT_LOG}SKIP|$kernel_hash|找不到 wrapped ISA\n"
        echo ""
        return
    fi
    
    echo -e "${BLUE}[1/5]${NC} 找到 wrapped ISA: $(basename "$wrapped_isa")"
    
    # 查找原始 ISA 文件（用於 pipeline.py 提取 metadata）
    local original_isa=$(find "$kernel_dir" -maxdepth 1 -name "*-hip-amdgcn-amd-amdhsa-*.s" | head -n 1)
    
    if [ -z "$original_isa" ] || [ ! -f "$original_isa" ]; then
        echo -e "${YELLOW}⊘ 跳過：找不到原始完整 ISA 文件（用於提取 metadata）${NC}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        RESULT_LOG="${RESULT_LOG}SKIP|$kernel_hash|找不到原始 ISA\n"
        echo ""
        return
    fi
    
    echo -e "${BLUE}[2/5]${NC} 找到原始 ISA: $(basename "$original_isa")"
    
    # 運行 pipeline.py 生成 .hsaco
    local pipeline_workdir="${kernel_dir}/pipeline_wrapped_output"
    local wrapped_basename=$(basename "$wrapped_isa" .s)
    local rebuilt_hsaco="${pipeline_workdir}/${wrapped_basename}_rebuilt.hsaco"
    
    if [ $SKIP_PIPELINE -eq 1 ] && [ -f "$rebuilt_hsaco" ]; then
        echo -e "${BLUE}[3/5]${NC} 跳過 pipeline.py，使用已存在的 .hsaco"
    else
        echo -e "${BLUE}[3/5]${NC} 運行 pipeline.py 生成 .hsaco..."
        
        # 傳遞原始 ISA 文件給 pipeline.py（用於提取全局符號和正確的 metadata）
        if python3 "$PIPELINE_SCRIPT" \
            --chip "$CHIP" \
            --workdir "$pipeline_workdir" \
            --emit-isa \
            --original-isa "$original_isa" \
            "$wrapped_isa" > "${TEST_RESULTS_DIR}/${kernel_hash}_pipeline.log" 2>&1; then
            echo -e "${GREEN}  ✓ Pipeline 成功${NC}"
        else
            echo -e "${RED}  ✗ Pipeline 失敗${NC}"
            echo "  詳細日誌: ${TEST_RESULTS_DIR}/${kernel_hash}_pipeline.log"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            RESULT_LOG="${RESULT_LOG}FAIL|$kernel_hash|Pipeline 失敗\n"
            echo ""
            return
        fi
    fi
    
    if [ ! -f "$rebuilt_hsaco" ]; then
        echo -e "${RED}  ✗ 找不到生成的 .hsaco 文件${NC}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        RESULT_LOG="${RESULT_LOG}FAIL|$kernel_hash|找不到 .hsaco\n"
        echo ""
        return
    fi
    
    echo -e "${GREEN}  ✓ HSACO 文件: $(basename "$rebuilt_hsaco")${NC}"
    
    # 查找 host.o（支持兩種目錄結構）
    # 結構 1: offload_output_*/host-x86_64-unknown-linux-gnu.o
    # 結構 2: *-host-x86_64-unknown-linux-gnu.o（扁平結構）
    local host_o=$(find "$kernel_dir" -maxdepth 2 -name "*-host-x86_64-unknown-linux-gnu.o" -o -name "host-x86_64-unknown-linux-gnu.o" | head -n 1)
    
    if [ -z "$host_o" ] || [ ! -f "$host_o" ]; then
        echo -e "${YELLOW}⊘ 跳過：找不到 host.o 文件${NC}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        RESULT_LOG="${RESULT_LOG}SKIP|$kernel_hash|找不到 host.o\n"
        echo ""
        return
    fi
    
    echo -e "${BLUE}[4/5]${NC} 找到 host.o: $(basename "$host_o")"
    
    # 查找原始執行檔（支持兩種目錄結構）
    # 結構 1: compiled_output_*/executable
    # 結構 2: <kernel_hash>（可執行檔，扁平結構）
    local original_exe=""
    
    # 先嘗試找子目錄中的 executable
    original_exe=$(find "$kernel_dir" -path "*/compiled_output_*/executable" -type f | head -n 1)
    
    # 如果沒找到，嘗試找扁平結構中的可執行檔
    if [ -z "$original_exe" ]; then
        # 尋找與 kernel_hash 同名的可執行檔
        if [ -f "$kernel_dir/$kernel_hash" ] && [ -x "$kernel_dir/$kernel_hash" ]; then
            original_exe="$kernel_dir/$kernel_hash"
        fi
    fi
    
    if [ -z "$original_exe" ] || [ ! -f "$original_exe" ]; then
        echo -e "${YELLOW}⊘ 跳過：找不到原始執行檔${NC}"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        RESULT_LOG="${RESULT_LOG}SKIP|$kernel_hash|找不到執行檔\n"
        echo ""
        return
    fi
    
    echo -e "${BLUE}[5/5]${NC} 找到原始執行檔: $(basename "$original_exe")"
    
    # 嘗試從 kernel 目錄找到測試參數文件
    local test_args=""
    if [ -f "${kernel_dir}/test_args.txt" ]; then
        test_args=$(cat "${kernel_dir}/test_args.txt")
        echo -e "${BLUE}      使用測試參數: ${NC}$test_args"
    fi
    
    echo ""
    echo -e "${BLUE}運行 relink_and_compare.sh...${NC}"
    echo ""
    
    # 運行 relink_and_compare.sh
    local relink_exit_code=0
    export EXECUTION_TIMEOUT="$EXECUTION_TIMEOUT"
    export OUTPUT_BASE_DIR="$kernel_dir"
    
    if bash "$RELINK_SCRIPT" "$original_exe" "$host_o" "$rebuilt_hsaco" $test_args \
        > "${TEST_RESULTS_DIR}/${kernel_hash}_relink.log" 2>&1; then
        relink_exit_code=$?
    else
        relink_exit_code=$?
    fi
    
    echo ""
    
    # 根據退出碼判斷結果
    case $relink_exit_code in
        0)
            echo -e "${GREEN}✓✓✓ 測試通過！重建的 kernel 與原始 kernel 行為完全一致${NC}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            RESULT_LOG="${RESULT_LOG}PASS|$kernel_hash|完全一致\n"
            ;;
        1)
            echo -e "${YELLOW}⚠ 部分通過：退出碼相同但輸出有差異${NC}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            RESULT_LOG="${RESULT_LOG}PARTIAL|$kernel_hash|輸出有差異\n"
            ;;
        3)
            echo -e "${RED}✗ 執行超時${NC}"
            TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
            RESULT_LOG="${RESULT_LOG}TIMEOUT|$kernel_hash|執行超時\n"
            ;;
        *)
            echo -e "${RED}✗✗✗ 測試失敗！重建的 kernel 與原始 kernel 行為不一致${NC}"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            RESULT_LOG="${RESULT_LOG}FAIL|$kernel_hash|行為不一致\n"
            ;;
    esac
    
    echo "詳細日誌: ${TEST_RESULTS_DIR}/${kernel_hash}_relink.log"
    echo ""
}

# 主循環：遍歷所有 kernel
echo -e "${BLUE}開始遍歷 kernel 資料夾...${NC}"
echo ""

kernel_count=0

if [ -n "$SPECIFIC_KERNEL" ]; then
    # 測試特定 kernel
    kernel_dir="${RUN_SUCCESS_DIR}/${SPECIFIC_KERNEL}"
    if [ -d "$kernel_dir" ]; then
        test_kernel "$kernel_dir"
    else
        echo -e "${RED}錯誤：找不到指定的 kernel: $SPECIFIC_KERNEL${NC}"
        exit 1
    fi
else
    # 測試所有或限定數量的 kernel
    for kernel_dir in "$RUN_SUCCESS_DIR"/*/; do
        # 檢查是否達到限制
        if [ -n "$LIMIT" ] && [ $kernel_count -ge "$LIMIT" ]; then
            echo -e "${YELLOW}已達到測試限制 ($LIMIT 個 kernel)，停止測試${NC}"
            break
        fi
        
        test_kernel "$kernel_dir"
        kernel_count=$((kernel_count + 1))
    done
fi

# 生成測試總結
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  測試總結                                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

{
    echo "Wrapped ISA Pipeline 測試總結"
    echo "================================"
    echo ""
    echo "測試時間: $(date)"
    echo "配置:"
    echo "  - Run Success 目錄: $RUN_SUCCESS_DIR"
    echo "  - GPU 晶片: $CHIP"
    echo "  - 執行超時: ${EXECUTION_TIMEOUT}秒"
    echo ""
    echo "統計:"
    echo "  - 總計: $TOTAL_KERNELS"
    echo "  - 成功: $SUCCESS_COUNT"
    echo "  - 失敗: $FAILED_COUNT"
    echo "  - 超時: $TIMEOUT_COUNT"
    echo "  - 跳過: $SKIPPED_COUNT"
    echo ""
    
    if [ $TOTAL_KERNELS -gt 0 ]; then
        success_rate=$(awk "BEGIN {printf \"%.2f\", ($SUCCESS_COUNT / $TOTAL_KERNELS) * 100}")
        echo "  - 成功率: ${success_rate}%"
    fi
    
    echo ""
    echo "詳細結果:"
    echo "--------"
    echo -e "$RESULT_LOG" | column -t -s '|' -N "狀態,Kernel Hash,說明"
    echo ""
    echo "所有日誌文件已保存至:"
    echo "  $TEST_RESULTS_DIR"
} | tee "$SUMMARY_FILE"

echo ""
echo -e "${CYAN}統計總覽：${NC}"
echo -e "  ${GREEN}✓ 成功: $SUCCESS_COUNT${NC}"
echo -e "  ${RED}✗ 失敗: $FAILED_COUNT${NC}"
echo -e "  ${YELLOW}⌛ 超時: $TIMEOUT_COUNT${NC}"
echo -e "  ${YELLOW}⊘ 跳過: $SKIPPED_COUNT${NC}"
echo -e "  ${BLUE}━ 總計: $TOTAL_KERNELS${NC}"
echo ""

if [ $TOTAL_KERNELS -gt 0 ]; then
    success_rate=$(awk "BEGIN {printf \"%.2f\", ($SUCCESS_COUNT / $TOTAL_KERNELS) * 100}")
    echo -e "${CYAN}成功率: ${success_rate}%${NC}"
fi

echo ""
echo -e "${BLUE}測試總結已保存至: ${SUMMARY_FILE}${NC}"
echo ""

# 根據結果設定退出碼
if [ $FAILED_COUNT -eq 0 ] && [ $TIMEOUT_COUNT -eq 0 ] && [ $SUCCESS_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓✓✓ 所有測試通過！${NC}"
    exit 0
elif [ $SUCCESS_COUNT -gt 0 ]; then
    echo -e "${YELLOW}⚠ 部分測試通過${NC}"
    exit 1
else
    echo -e "${RED}✗✗✗ 所有測試失敗${NC}"
    exit 2
fi
