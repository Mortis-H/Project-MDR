#!/bin/bash

# 批次測試 pipeline.py 工具的正確性
# 遍歷所有 kernel 資料夾，對每個 kernel 的 compiled_output 下的 .s 檔案進行測試
# 測試結果和 log 都放在各自的 kernel 資料夾底下
#
# 使用方式：
#   1. 測試所有 kernel：
#      ./rebuild_isa_all.sh
#   
#   2. 指定搜尋路徑：
#      ./rebuild_isa_all.sh --path deterministic_kernels
#
#   3. 自訂平行任務數：
#      JOBS=8 ./rebuild_isa_all.sh --path deterministic_kernels
#
#   4. 只測試特定 kernel：
#      KERNEL_FILTER="002_cuda_code*" ./rebuild_isa_all.sh --path deterministic_kernels
#
#   5. 設定 kernel 層級的平行數：
#      KERNEL_JOBS=4 JOBS=8 ./rebuild_isa_all.sh --path deterministic_kernels

# 設定顏色輸出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 獲取腳本所在目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 預設配置
DEFAULT_BASE_DIR="/home/andycha/workspaces/multi_kernel_testcases"
PIPELINE_SCRIPT="/home/andycha/workspaces/Project-MDR/Track_B/amdisa-toolkit/examples/pipeline.py"
CHIP="gfx950"
KERNEL_FILTER="${KERNEL_FILTER:-*}"  # 預設處理所有 kernel，可用環境變數指定過濾

# 解析參數
SEARCH_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --path)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                SEARCH_DIR="$2"
                shift 2
            else
                echo -e "${RED}錯誤: --path 需要指定路徑參數${NC}"
                exit 1
            fi
            ;;
        --path=*)
            SEARCH_DIR="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "用法: $0 [--path PATH]"
            echo ""
            echo "選項："
            echo "  --path PATH     指定搜尋目錄（預設：$DEFAULT_BASE_DIR）"
            echo "                  可以是絕對路徑或相對路徑"
            echo "  --help          顯示此說明"
            echo ""
            echo "環境變數："
            echo "  JOBS            .s 檔案層級的平行數（預設為 CPU 核心數）"
            echo "  KERNEL_JOBS     kernel 層級的平行數（預設 4）"
            echo "  KERNEL_FILTER   只處理符合模式的 kernel（預設 '*'）"
            echo ""
            echo "範例："
            echo "  $0                                         # 測試預設目錄下所有 kernel"
            echo "  $0 --path deterministic_kernels            # 測試指定目錄下所有 kernel"
            echo "  JOBS=8 $0 --path ./my_kernels              # 使用 8 個平行任務"
            echo "  KERNEL_FILTER=\"002_*\" $0 --path kernels  # 只處理 002_ 開頭的 kernel"
            exit 0
            ;;
        *)
            echo -e "${RED}未知參數: $1${NC}"
            echo "使用 --help 查看說明"
            exit 1
            ;;
    esac
done

# 設定搜尋目錄
if [ -z "$SEARCH_DIR" ]; then
    # 如果未指定，使用預設目錄
    BASE_DIR="$DEFAULT_BASE_DIR"
else
    # 轉換為絕對路徑
    if [[ "$SEARCH_DIR" = /* ]]; then
        # 已經是絕對路徑
        BASE_DIR="$SEARCH_DIR"
    else
        # 相對路徑，從當前目錄轉換
        BASE_DIR="$(cd "$(pwd)/$SEARCH_DIR" 2>/dev/null && pwd)"
        if [ -z "$BASE_DIR" ]; then
            echo -e "${RED}錯誤: 無效的路徑${NC}"
            exit 1
        fi
    fi
fi

# 檢查搜尋目錄是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}錯誤: 目錄不存在: $BASE_DIR${NC}"
    exit 1
fi

# 自動檢測 CPU 核心數
NPROC=$(nproc 2>/dev/null || echo "4")
JOBS=${JOBS:-$NPROC}  # .s 檔案層級的平行數
KERNEL_JOBS=${KERNEL_JOBS:-4}  # kernel 層級的平行數（預設 4 個 kernel 同時處理）

# 檢查必要的檔案和工具
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}錯誤: 找不到基本目錄: $BASE_DIR${NC}"
    exit 1
fi

if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo -e "${RED}錯誤: 找不到 pipeline.py: $PIPELINE_SCRIPT${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}錯誤: 找不到 python3${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo "Pipeline 工具批次測試腳本"
echo -e "${BLUE}========================================${NC}"
echo -e "${CYAN}搜尋目錄: $BASE_DIR${NC}"
echo "Pipeline 腳本: $PIPELINE_SCRIPT"
echo "目標架構: $CHIP"
echo "Kernel 過濾: $KERNEL_FILTER"
echo "Kernel 層級平行數: $KERNEL_JOBS (個 kernel 同時處理)"
echo ".s 檔案層級平行數: $JOBS (每個 kernel 內)"
echo -e "${BLUE}========================================${NC}"
echo ""

# 測試單個 .s 檔案的函數
test_one_asm() {
    local asm_file="$1"
    local kernel_dir="$2"
    local test_output_dir="$3"
    local base_name=$(basename "$asm_file" .s)
    
    # 為每個測試創建獨立目錄
    local work_dir="$test_output_dir/$base_name"
    mkdir -p "$work_dir"
    
    local log_file="$test_output_dir/logs/${base_name}.log"
    
    # 運行 pipeline.py
    echo "[開始測試] $base_name" > "$log_file"
    echo "輸入檔案: $asm_file" >> "$log_file"
    echo "工作目錄: $work_dir" >> "$log_file"
    echo "" >> "$log_file"
    
    if python3 "$PIPELINE_SCRIPT" \
        --chip "$CHIP" \
        --workdir "$work_dir" \
        "$asm_file" >> "$log_file" 2>&1; then
        
        # 檢查預期的輸出檔案是否存在
        local expected_files=(
            "${base_name}_rebuilt.amdisamlir"
            "${base_name}_rebuilt.gpumlir"
            "${base_name}_rebuilt.s"
            "${base_name}_rebuilt.hsaco"
        )
        
        local all_exist=true
        local missing_files=""
        
        for expected in "${expected_files[@]}"; do
            if [ ! -f "$work_dir/$expected" ]; then
                all_exist=false
                missing_files="$missing_files $expected"
            fi
        done
        
        if [ "$all_exist" = true ]; then
            # 統計檔案大小
            local amdisamlir_size=$(stat -c%s "$work_dir/${base_name}_rebuilt.amdisamlir" 2>/dev/null || echo "0")
            local gpumlir_size=$(stat -c%s "$work_dir/${base_name}_rebuilt.gpumlir" 2>/dev/null || echo "0")
            local s_size=$(stat -c%s "$work_dir/${base_name}_rebuilt.s" 2>/dev/null || echo "0")
            local hsaco_size=$(stat -c%s "$work_dir/${base_name}_rebuilt.hsaco" 2>/dev/null || echo "0")
            
            echo "" >> "$log_file"
            echo "[成功] 所有預期檔案都已生成" >> "$log_file"
            echo "  - ${base_name}_rebuilt.amdisamlir ($amdisamlir_size bytes)" >> "$log_file"
            echo "  - ${base_name}_rebuilt.gpumlir ($gpumlir_size bytes)" >> "$log_file"
            echo "  - ${base_name}_rebuilt.s ($s_size bytes)" >> "$log_file"
            echo "  - ${base_name}_rebuilt.hsaco ($hsaco_size bytes)" >> "$log_file"
            
            echo "SUCCESS|$base_name|$amdisamlir_size|$gpumlir_size|$s_size|$hsaco_size"
            return 0
        else
            echo "" >> "$log_file"
            echo "[失敗] 缺少預期的輸出檔案:$missing_files" >> "$log_file"
            echo "PARTIAL|$base_name|$missing_files"
            return 1
        fi
    else
        echo "" >> "$log_file"
        echo "[失敗] pipeline.py 執行失敗" >> "$log_file"
        echo "FAILED|$base_name|pipeline_execution_failed"
        return 1
    fi
}

# 處理單個 kernel 的函數
process_one_kernel() {
    local kernel_dir="$1"
    local kernel_name=$(basename "$kernel_dir")
    
    # 跳過不是 kernel 的目錄
    if [[ "$kernel_name" == compiled_output* ]] || \
       [[ "$kernel_name" == pipeline_test* ]] || \
       [[ "$kernel_name" == failed_kernels* ]] || \
       [[ "$kernel_name" == relink_test* ]] || \
       [[ "$kernel_name" == "logs" ]] || \
       [[ "$kernel_name" == pipeline_test_global_summary* ]]; then
        return 0
    fi
    
    # 尋找 compiled_output 開頭的子資料夾
    local compiled_dirs=("$kernel_dir"/compiled_output*)
    
    # 檢查是否找到 compiled_output 資料夾
    if [ ! -e "${compiled_dirs[0]}" ]; then
        echo -e "${YELLOW}[跳過] $kernel_name: 找不到 compiled_output 資料夾${NC}"
        return 0
    fi
    
    # 處理找到的 compiled_output 資料夾
    for compiled_dir in "${compiled_dirs[@]}"; do
        if [ ! -d "$compiled_dir" ]; then
            continue
        fi
        
        echo -e "${MAGENTA}========================================${NC}"
        echo -e "${MAGENTA}[處理] Kernel: $kernel_name${NC}"
        echo -e "${MAGENTA}       編譯輸出: $(basename "$compiled_dir")${NC}"
        echo -e "${MAGENTA}========================================${NC}"
        
        # 尋找 .s 檔案
        local asm_files=()
        while IFS= read -r -d '' file; do
            asm_files+=("$file")
        done < <(find "$compiled_dir" -name "*-hip-amdgcn-amd-amdhsa-${CHIP}.s" -type f -print0)
        
        local file_count=${#asm_files[@]}
        
        if [ $file_count -eq 0 ]; then
            echo -e "${YELLOW}  警告: 找不到任何 .s 檔案${NC}"
            continue
        fi
        
        echo "  找到 $file_count 個 .s 檔案"
        
        # 創建測試輸出目錄（在 kernel 資料夾底下）
        local TEST_OUTPUT_DIR="$kernel_dir/pipeline_test_results_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$TEST_OUTPUT_DIR"
        local LOG_DIR="$TEST_OUTPUT_DIR/logs"
        mkdir -p "$LOG_DIR"
        
        local SUMMARY_LOG="$LOG_DIR/summary.txt"
        
        echo "  測試輸出: $(basename "$TEST_OUTPUT_DIR")"
        echo ""
        echo "  開始時間: $(date)" | tee "$SUMMARY_LOG" > /dev/null
        
        # 匯出函數和變數供子進程使用
        export -f test_one_asm
        export PIPELINE_SCRIPT
        export CHIP
        
        # 平行執行測試
        if command -v xargs &> /dev/null; then
            printf "%s\n" "${asm_files[@]}" | xargs -P "$JOBS" -I {} bash -c \
                "test_one_asm \"{}\" \"$kernel_dir\" \"$TEST_OUTPUT_DIR\"" > "$LOG_DIR/results.txt"
        else
            rm -f "$LOG_DIR/results.txt"
            for asm_file in "${asm_files[@]}"; do
                test_one_asm "$asm_file" "$kernel_dir" "$TEST_OUTPUT_DIR" >> "$LOG_DIR/results.txt"
            done
        fi
        
        echo ""
        echo "  結束時間: $(date)" | tee -a "$SUMMARY_LOG" > /dev/null
        echo ""
        
        # 統計結果
        if [ -f "$LOG_DIR/results.txt" ]; then
            # 取得計數並清理空白字元
            local success=$(grep -c "^SUCCESS" "$LOG_DIR/results.txt" 2>/dev/null || echo "0")
            success=$(echo "$success" | tr -d '[:space:]')
            success=${success:-0}
            
            local partial=$(grep -c "^PARTIAL" "$LOG_DIR/results.txt" 2>/dev/null || echo "0")
            partial=$(echo "$partial" | tr -d '[:space:]')
            partial=${partial:-0}
            
            local failed=$(grep -c "^FAILED" "$LOG_DIR/results.txt" 2>/dev/null || echo "0")
            failed=$(echo "$failed" | tr -d '[:space:]')
            failed=${failed:-0}
            
            echo "  測試完成！" | tee -a "$SUMMARY_LOG" > /dev/null
            echo "" | tee -a "$SUMMARY_LOG" > /dev/null
            echo "  總計: $file_count 個檔案" | tee -a "$SUMMARY_LOG" > /dev/null
            echo "  完全成功: $success 個檔案" | tee -a "$SUMMARY_LOG" > /dev/null
            
            if [ $partial -gt 0 ]; then
                echo "  部分成功: $partial 個檔案" | tee -a "$SUMMARY_LOG" > /dev/null
            fi
            
            if [ $failed -gt 0 ]; then
                echo "  執行失敗: $failed 個檔案" | tee -a "$SUMMARY_LOG" > /dev/null
            fi
            
            # 成功率
            if [ $file_count -gt 0 ]; then
                local success_rate=$(awk "BEGIN {printf \"%.2f\", ($success / $file_count) * 100}")
                echo "" | tee -a "$SUMMARY_LOG" > /dev/null
                echo "  成功率: ${success_rate}%" | tee -a "$SUMMARY_LOG" > /dev/null
            fi
            
            # 輸出結果供全域統計使用
            # 增加一個標記：如果有失敗或部分成功，標記為 has_failures
            if [ $failed -gt 0 ] || [ $partial -gt 0 ]; then
                echo "KERNEL_RESULT|$kernel_name|$file_count|$success|$partial|$failed|HAS_FAILURES"
            else
                echo "KERNEL_RESULT|$kernel_name|$file_count|$success|$partial|$failed|ALL_SUCCESS"
            fi
        fi
    done
}

# 匯出函數供平行處理使用
export -f process_one_kernel
export -f test_one_asm
export PIPELINE_SCRIPT
export CHIP
export JOBS
export BASE_DIR

# 全域彙總日誌
GLOBAL_SUMMARY_LOG="$BASE_DIR/pipeline_test_global_summary_$(date +%Y%m%d_%H%M%S).txt"
GLOBAL_RESULTS_FILE="/tmp/pipeline_test_kernel_results_$$.txt"

echo "全域測試開始時間: $(date)" > "$GLOBAL_SUMMARY_LOG"
echo "" >> "$GLOBAL_SUMMARY_LOG"

echo "搜尋 kernel 資料夾..."

# 收集所有 kernel 資料夾
kernel_dirs=()
for kernel_dir in "$BASE_DIR"/$KERNEL_FILTER/; do
    if [ -d "$kernel_dir" ]; then
        kernel_name=$(basename "$kernel_dir")
        # 跳過不是 kernel 的目錄
        if [[ "$kernel_name" == compiled_output* ]] || \
           [[ "$kernel_name" == pipeline_test* ]] || \
           [[ "$kernel_name" == failed_kernels* ]] || \
           [[ "$kernel_name" == relink_test* ]] || \
           [[ "$kernel_name" == "logs" ]]; then
            continue
        fi
        # 檢查是否有 compiled_output 資料夾
        if ls "$kernel_dir"/compiled_output* >/dev/null 2>&1; then
            kernel_dirs+=("$kernel_dir")
        fi
    fi
done

echo "找到 ${#kernel_dirs[@]} 個 kernel 資料夾需要處理"
echo ""

# 平行處理所有 kernel
if command -v xargs &> /dev/null && [ ${#kernel_dirs[@]} -gt 0 ]; then
    printf "%s\n" "${kernel_dirs[@]}" | xargs -P "$KERNEL_JOBS" -I {} bash -c \
        "process_one_kernel \"{}\"" > "$GLOBAL_RESULTS_FILE"
else
    rm -f "$GLOBAL_RESULTS_FILE"
    for kernel_dir in "${kernel_dirs[@]}"; do
        process_one_kernel "$kernel_dir" >> "$GLOBAL_RESULTS_FILE"
    done
fi

# 輸出全域統計
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}全域測試彙總${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "全域測試結束時間: $(date)" >> "$GLOBAL_SUMMARY_LOG"
echo "" >> "$GLOBAL_SUMMARY_LOG"

# 統計全域結果
TOTAL_KERNELS=0
TOTAL_FILES=0
TOTAL_SUCCESS=0
TOTAL_PARTIAL=0
TOTAL_FAILED=0
FAILED_KERNELS=()  # 記錄失敗的 kernel 名稱

if [ -f "$GLOBAL_RESULTS_FILE" ]; then
    while IFS='|' read -r marker kernel_name file_count success partial failed status; do
        if [ "$marker" = "KERNEL_RESULT" ]; then
            TOTAL_KERNELS=$((TOTAL_KERNELS + 1))
            TOTAL_FILES=$((TOTAL_FILES + file_count))
            TOTAL_SUCCESS=$((TOTAL_SUCCESS + success))
            TOTAL_PARTIAL=$((TOTAL_PARTIAL + partial))
            TOTAL_FAILED=$((TOTAL_FAILED + failed))
            
            # 記錄失敗的 kernel
            if [ "$status" = "HAS_FAILURES" ]; then
                FAILED_KERNELS+=("$kernel_name")
            fi
            
            # 記錄到全域彙總
            echo "----------------------------------------" >> "$GLOBAL_SUMMARY_LOG"
            echo "Kernel: $kernel_name" >> "$GLOBAL_SUMMARY_LOG"
            echo "  檔案數: $file_count" >> "$GLOBAL_SUMMARY_LOG"
            echo "  成功: $success, 部分成功: $partial, 失敗: $failed" >> "$GLOBAL_SUMMARY_LOG"
            if [ $file_count -gt 0 ]; then
                success_rate=$(awk "BEGIN {printf \"%.2f\", ($success / $file_count) * 100}")
                echo "  成功率: ${success_rate}%" >> "$GLOBAL_SUMMARY_LOG"
            fi
            echo "" >> "$GLOBAL_SUMMARY_LOG"
        fi
    done < "$GLOBAL_RESULTS_FILE"
    
    rm -f "$GLOBAL_RESULTS_FILE"
fi

echo "========================================"  | tee -a "$GLOBAL_SUMMARY_LOG"
echo "全域統計："  | tee -a "$GLOBAL_SUMMARY_LOG"
echo "  處理的 Kernel 數量: $TOTAL_KERNELS"  | tee -a "$GLOBAL_SUMMARY_LOG"
echo "  處理的檔案總數: $TOTAL_FILES"  | tee -a "$GLOBAL_SUMMARY_LOG"
echo -e "${GREEN}  完全成功: $TOTAL_SUCCESS 個檔案${NC}"  | tee -a "$GLOBAL_SUMMARY_LOG"

if [ $TOTAL_PARTIAL -gt 0 ]; then
    echo -e "${YELLOW}  部分成功: $TOTAL_PARTIAL 個檔案${NC}"  | tee -a "$GLOBAL_SUMMARY_LOG"
fi

if [ $TOTAL_FAILED -gt 0 ]; then
    echo -e "${RED}  執行失敗: $TOTAL_FAILED 個檔案${NC}"  | tee -a "$GLOBAL_SUMMARY_LOG"
fi

if [ $TOTAL_FILES -gt 0 ]; then
    global_success_rate=$(awk "BEGIN {printf \"%.2f\", ($TOTAL_SUCCESS / $TOTAL_FILES) * 100}")
    echo ""  | tee -a "$GLOBAL_SUMMARY_LOG"
    echo "  全域成功率: ${global_success_rate}%"  | tee -a "$GLOBAL_SUMMARY_LOG"
fi

echo "========================================"  | tee -a "$GLOBAL_SUMMARY_LOG"
echo ""

# 複製失敗的 kernel 到 failed 資料夾
if [ ${#FAILED_KERNELS[@]} -gt 0 ]; then
    FAILED_DIR="$BASE_DIR/failed_kernels_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$FAILED_DIR"
    
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}複製失敗的 Kernel 資料夾${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo "目標目錄: $FAILED_DIR"
    echo "失敗的 Kernel 數量: ${#FAILED_KERNELS[@]}"
    echo ""
    
    # 記錄到全域彙總
    echo "========================================" >> "$GLOBAL_SUMMARY_LOG"
    echo "失敗的 Kernel 列表 (已複製到 $FAILED_DIR):" >> "$GLOBAL_SUMMARY_LOG"
    echo "" >> "$GLOBAL_SUMMARY_LOG"
    
    for kernel_name in "${FAILED_KERNELS[@]}"; do
        kernel_path="$BASE_DIR/$kernel_name"
        if [ -d "$kernel_path" ]; then
            # 複製整個 kernel 資料夾
            cp -r "$kernel_path" "$FAILED_DIR/" 2>/dev/null
            
        fi
    done
    
    echo ""
    echo -e "${GREEN}失敗的 Kernel 已複製到: $FAILED_DIR${NC}"
    echo ""
    echo "" >> "$GLOBAL_SUMMARY_LOG"
    echo "失敗的 Kernel 資料夾位置: $FAILED_DIR" >> "$GLOBAL_SUMMARY_LOG"
    echo "" >> "$GLOBAL_SUMMARY_LOG"
else
    echo -e "${GREEN}🎉 所有 Kernel 測試全部通過！${NC}"
    echo ""
    echo "所有 Kernel 測試全部通過！" >> "$GLOBAL_SUMMARY_LOG"
    echo "" >> "$GLOBAL_SUMMARY_LOG"
fi

echo "全域彙總報告已儲存至: $GLOBAL_SUMMARY_LOG"
echo ""
echo "提示："
echo "  - 每個 kernel 的測試結果位於: <kernel_name>/pipeline_test_results_*/"
echo "  - 每個 kernel 的日誌位於: <kernel_name>/pipeline_test_results_*/logs/"
if [ ${#FAILED_KERNELS[@]} -gt 0 ]; then
    echo -e "  - ${YELLOW}失敗的 kernel 已複製到: failed_kernels_*/${NC}"
fi
echo ""
