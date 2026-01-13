#!/bin/bash

# 編譯所有 HIP 檔案並保存組譯檔案的腳本（平行化版本）
# 每個 HIP 檔案的編譯輸出都放在獨立的資料夾中
# 使用 --save-temps 參數保存中間檔案（包括 .s 組譯檔案）
#
# 用法：
#   ./compile_all_hip.sh                          # 編譯當前目錄下所有 HIP 檔案
#   ./compile_all_hip.sh -p <目錄>                # 編譯指定目錄下所有 HIP 檔案
#   ./compile_all_hip.sh -p deterministic_kernels  # 編譯 deterministic_kernels 下的所有 HIP 檔案

# 設定顏色輸出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 獲取腳本所在目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 解析參數
SEARCH_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--path)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                SEARCH_DIR="$2"
                shift 2
            else
                echo -e "${RED}錯誤: -p 需要指定路徑參數${NC}"
                exit 1
            fi
            ;;
        -p=*|--path=*)
            SEARCH_DIR="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "用法: $0 [-p PATH]"
            echo ""
            echo "選項："
            echo "  -p, --path PATH 指定搜尋目錄（預設為當前目錄）"
            echo "                  可以是絕對路徑或相對路徑"
            echo "  --help          顯示此說明"
            echo ""
            echo "環境變數："
            echo "  JOBS            平行編譯的任務數（預設為 CPU 核心數）"
            echo ""
            echo "範例："
            echo "  $0                                    # 編譯當前目錄下所有 HIP"
            echo "  $0 -p deterministic_kernels           # 編譯指定目錄下所有 HIP"
            echo "  JOBS=8 $0 -p ./my_kernels             # 使用 8 個平行任務編譯"
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
    # 如果未指定，使用當前工作目錄
    SEARCH_DIR="$(pwd)"
else
    # 轉換為絕對路徑
    if [[ "$SEARCH_DIR" = /* ]]; then
        # 已經是絕對路徑
        :
    else
        # 相對路徑，從當前目錄轉換
        SEARCH_DIR="$(cd "$(pwd)/$SEARCH_DIR" 2>/dev/null && pwd)"
        if [ -z "$SEARCH_DIR" ]; then
            echo -e "${RED}錯誤: 無效的路徑${NC}"
            exit 1
        fi
    fi
fi

# 檢查搜尋目錄是否存在
if [ ! -d "$SEARCH_DIR" ]; then
    echo -e "${RED}錯誤: 目錄不存在: $SEARCH_DIR${NC}"
    exit 1
fi

# 輸出目錄名稱（將在各個 kernel 目錄下創建）
OUTPUT_BASE="compiled_output_$(date +%Y%m%d_%H%M%S)"

# 創建集中的日誌目錄（在搜尋目錄下）
LOG_DIR="$SEARCH_DIR/${OUTPUT_BASE}_logs"
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/summary.txt"
ERROR_LOG="$LOG_DIR/errors.txt"

# 自動檢測 CPU 核心數，預設使用所有核心
NPROC=$(nproc 2>/dev/null || echo "4")
JOBS=${JOBS:-$NPROC}

# 編譯單個檔案的函數
compile_one() {
    local hip_file="$1"           # 絕對路徑的 HIP 檔案
    local output_base="$2"        # 輸出目錄名稱 (compiled_output_*)
    local central_log_dir="$3"    # 集中日誌目錄的絕對路徑
    
    # 從路徑中提取檔案名稱和目錄
    local file_name=$(basename "$hip_file" .hip)
    local base_name="$file_name"
    local hip_dir=$(dirname "$hip_file")
    
    # 在 HIP 檔案所在目錄下創建輸出目錄
    local output_dir="$hip_dir/$output_base"
    mkdir -p "$output_dir"
    
    # 進入輸出目錄進行編譯
    cd "$output_dir"
    
    # 日誌檔案放在集中的日誌目錄
    local file_log="$central_log_dir/${base_name}.log"
    
    # 使用 hipcc 編譯並鏈接，--save-temps 會保存所有中間檔案（包括 .s）
    # 使用絕對路徑引用源檔案
    if hipcc --save-temps "$hip_file" -o "${base_name}" > "$file_log" 2>&1; then
        # 統計生成的檔案
        local s_files=$(ls *.s 2>/dev/null | wc -l)
        echo "SUCCESS|$hip_file|$output_dir|$s_files"
        return 0
    else
        echo "FAILED|$hip_file|$output_dir"
        return 1
    fi
}

# 匯出函數和變數供子 shell 使用
export -f compile_one
export OUTPUT_BASE
export LOG_DIR

echo -e "${BLUE}========================================${NC}"
echo "開始平行編譯 HIP 檔案"
echo -e "${CYAN}搜尋目錄: $SEARCH_DIR${NC}"
echo "輸出目錄模式: 每個 kernel 目錄下的 $OUTPUT_BASE/"
echo "集中日誌目錄: $LOG_DIR"
echo "使用 $JOBS 個平行任務"
echo -e "${BLUE}========================================${NC}"

# 遞迴查找所有 .hip 檔案（使用絕對路徑）
# 排除已編譯的輸出目錄和測試目錄
mapfile -t HIP_FILES < <(find "$SEARCH_DIR" -name "*.hip" -type f \
    ! -path "*/compiled_output_*/*" \
    ! -path "*/pipeline_test_*/*" \
    ! -path "*/relink_test_*/*" \
    ! -path "*/failed_kernels*/*" \
    ! -path "*/test_*/*" \
    | sort)

# 計算總檔案數
total=${#HIP_FILES[@]}
echo "找到 $total 個 HIP 檔案"
echo ""

if [ $total -eq 0 ]; then
    echo -e "${RED}錯誤: 未找到任何 .hip 檔案${NC}"
    exit 1
fi

# 檢查是否有 GNU parallel
if command -v parallel &> /dev/null; then
    echo "使用 GNU parallel 進行平行編譯..."
    echo "開始時間: $(date)" | tee "$SUMMARY_LOG"
    
    # 使用 GNU parallel，顯示進度條
    printf '%s\n' "${HIP_FILES[@]}" | parallel -j "$JOBS" --bar --eta \
        compile_one {} "$OUTPUT_BASE" "$LOG_DIR" > "$LOG_DIR/results.txt"
    
elif command -v xargs &> /dev/null; then
    echo "使用 xargs 進行平行編譯..."
    echo "開始時間: $(date)" | tee "$SUMMARY_LOG"
    
    # 使用 xargs -P 進行平行處理
    printf '%s\n' "${HIP_FILES[@]}" | xargs -P "$JOBS" -I {} bash -c \
        "compile_one {} \"$OUTPUT_BASE\" \"$LOG_DIR\"" > "$LOG_DIR/results.txt"
else
    echo -e "${YELLOW}警告: 未找到 parallel 或 xargs，使用序列編譯${NC}"
    echo "開始時間: $(date)" | tee "$SUMMARY_LOG"
    
    # 回退到序列編譯
    for hip_file in "${HIP_FILES[@]}"; do
        if [ -f "$hip_file" ]; then
            compile_one "$hip_file" "$OUTPUT_BASE" "$LOG_DIR" >> "$LOG_DIR/results.txt"
        fi
    done
fi

echo ""
echo "結束時間: $(date)" | tee -a "$SUMMARY_LOG"
echo -e "${BLUE}========================================${NC}"

# 統計結果
if [ -f "$LOG_DIR/results.txt" ]; then
    success=$(grep -c "^SUCCESS" "$LOG_DIR/results.txt" 2>/dev/null || echo "0")
    failed=$(grep -c "^FAILED" "$LOG_DIR/results.txt" 2>/dev/null || echo "0")
    
    echo "編譯完成！" | tee -a "$SUMMARY_LOG"
    echo "總計: $total 個檔案" | tee -a "$SUMMARY_LOG"
    echo -e "${GREEN}成功: $success 個檔案${NC}" | tee -a "$SUMMARY_LOG"
    
    if [ $failed -gt 0 ]; then
        echo -e "${RED}失敗: $failed 個檔案${NC}" | tee -a "$SUMMARY_LOG"
        echo "" > "$ERROR_LOG"
        echo "失敗的檔案:" | tee -a "$ERROR_LOG"
        grep "^FAILED" "$LOG_DIR/results.txt" | cut -d'|' -f2 | tee -a "$ERROR_LOG"
        echo "" | tee -a "$ERROR_LOG"
        echo "詳細錯誤訊息請查看 $LOG_DIR 目錄下的個別日誌檔案" | tee -a "$ERROR_LOG"
    fi
    
    # 統計成功編譯的檔案的輸出
    if [ $success -gt 0 ]; then
        echo "" | tee -a "$SUMMARY_LOG"
        echo "成功編譯的檔案輸出目錄:" | tee -a "$SUMMARY_LOG"
        grep "^SUCCESS" "$LOG_DIR/results.txt" | while IFS='|' read -r status hip_file output_dir s_count; do
            echo "  - $hip_file -> $output_dir (生成 $s_count 個 .s 檔案)" | tee -a "$SUMMARY_LOG"
        done
    fi
else
    echo -e "${RED}錯誤: 找不到結果檔案${NC}"
fi

echo -e "${BLUE}========================================${NC}"

# 顯示生成的檔案統計（在搜尋目錄下搜索）
total_s_files=$(find "$SEARCH_DIR" -path "*/$OUTPUT_BASE/*.s" -type f 2>/dev/null | wc -l)
total_o_files=$(find "$SEARCH_DIR" -path "*/$OUTPUT_BASE/*.o" -type f 2>/dev/null | wc -l)
total_exe_files=$(find "$SEARCH_DIR" -path "*/$OUTPUT_BASE/*" -type f -executable ! -name "*.out" ! -name "*.sh" 2>/dev/null | wc -l)
total_output_dirs=$(find "$SEARCH_DIR" -type d -name "$OUTPUT_BASE" 2>/dev/null | wc -l)

echo "統計資訊:" | tee -a "$SUMMARY_LOG"
echo "  創建的輸出目錄數: $total_output_dirs" | tee -a "$SUMMARY_LOG"
echo "  生成的 .s 組譯檔案總數: $total_s_files" | tee -a "$SUMMARY_LOG"
echo "  生成的 .o 目標檔案總數: $total_o_files" | tee -a "$SUMMARY_LOG"
echo "  生成的可執行檔總數: $total_exe_files" | tee -a "$SUMMARY_LOG"

echo ""
echo "輸出檔案位置: 各個 kernel 目錄下的 $OUTPUT_BASE/ 子目錄"
echo "集中日誌目錄: $LOG_DIR/"
echo "查看完整結果: cat $SUMMARY_LOG"
if [ $failed -gt 0 ]; then
    echo "查看錯誤詳情: cat $ERROR_LOG"
fi
echo ""
echo "提示: 每個 HIP 檔案的編譯輸出都在其所在目錄的 $OUTPUT_BASE/ 子目錄中"
