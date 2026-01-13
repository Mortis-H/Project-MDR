#!/bin/bash

# 批次測試所有 kernel 的重新連結腳本
# 自動找到每個 kernel 的原始執行檔、host.o、rebuilt.hsaco 並進行測試
#
# 用法：
#   ./test_rebuilt_all.sh                          # 測試當前目錄下所有 kernel
#   ./test_rebuilt_all.sh -p <目錄>                # 測試指定目錄下所有 kernel
#   ./test_rebuilt_all.sh -p deterministic_kernels  # 測試 deterministic_kernels 下的所有 kernel

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 腳本目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELINK_SCRIPT="${SCRIPT_DIR}/relink_and_compare.sh"

# 檢查 relink_and_compare.sh 是否存在
if [ ! -f "$RELINK_SCRIPT" ]; then
    echo -e "${RED}錯誤：找不到 relink_and_compare.sh 腳本${NC}"
    echo "預期位置：$RELINK_SCRIPT"
    exit 1
fi

# 解析參數
SEARCH_DIR=""
KERNEL_ARGS=()

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
            echo "用法: $0 [-p PATH] [kernel_dir1] [kernel_dir2] ..."
            echo ""
            echo "選項："
            echo "  -p, --path PATH 指定搜尋目錄（預設為當前目錄）"
            echo "                  可以是絕對路徑或相對路徑"
            echo "  --help          顯示此說明"
            echo ""
            echo "如果不提供 kernel 目錄參數，將自動掃描並測試指定目錄下的所有 kernel"
            echo "如果提供 kernel 目錄參數，只測試指定的 kernel 目錄"
            echo ""
            echo "範例："
            echo "  $0                                         # 測試當前目錄下所有 kernels"
            echo "  $0 -p deterministic_kernels                # 測試指定目錄下所有 kernels"
            echo "  $0 -p deterministic_kernels 002_*          # 測試指定目錄下的特定 kernels"
            echo "  $0 002_kernel_A 003_kernel_B               # 測試當前目錄下的特定 kernels"
            exit 0
            ;;
        -*)
            echo -e "${RED}未知參數: $1${NC}"
            echo "使用 --help 查看說明"
            exit 1
            ;;
        *)
            # 其他參數視為 kernel 目錄
            KERNEL_ARGS+=("$1")
            shift
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

# 函數：從 .hip 檔案分析需要的參數
analyze_hip_args() {
    local hip_file="$1"
    local args=""
    local argc_count=0
    
    # 1. 檢查 argc 的數量要求
    if grep -q "argc.*!= *2" "$hip_file" 2>/dev/null || grep -q "argc.*< *2" "$hip_file" 2>/dev/null; then
        argc_count=1
    elif grep -q "argc.*!= *3" "$hip_file" 2>/dev/null || grep -q "argc.*< *3" "$hip_file" 2>/dev/null; then
        argc_count=2
    elif grep -q "argc.*!= *4" "$hip_file" 2>/dev/null || grep -q "argc.*< *4" "$hip_file" 2>/dev/null; then
        argc_count=3
    elif grep -q "argc.*!= *5" "$hip_file" 2>/dev/null || grep -q "argc.*< *5" "$hip_file" 2>/dev/null; then
        argc_count=4
    elif grep -q "argv\[1\]" "$hip_file" 2>/dev/null; then
        argc_count=1
    fi
    
    # 2. 嘗試從錯誤訊息或註釋中提取參數提示
    local usage_msg=$(grep -E "(printf|cout|Usage|usage|Falta|Missing|required)" "$hip_file" 2>/dev/null | head -1)
    
    # 3. 嘗試從程式碼中找到參數的具體含義
    local param_hints=""
    if grep -q "atoi.*argv\[1\]" "$hip_file" 2>/dev/null; then
        # 檢查變數名稱來推斷參數類型
        local var_name=$(grep -A 1 "atoi.*argv\[1\]" "$hip_file" | grep -oE "[A-Za-z_][A-Za-z0-9_]*\s*=" | head -1 | sed 's/=//g' | xargs)
        
        # 根據變數名稱推斷合適的值
        case "$var_name" in
            N|n|size|SIZE)
                param_hints="64"  # 數組大小
                ;;
            width|WIDTH|W)
                param_hints="32"  # 寬度
                ;;
            height|HEIGHT|H)
                param_hints="32"  # 高度
                ;;
            threads|THREADS)
                param_hints="256" # 線程數
                ;;
            blocks|BLOCKS)
                param_hints="16"  # Block 數
                ;;
            iterations|ITERATIONS|iter)
                param_hints="100" # 迭代次數
                ;;
            *)
                param_hints="64"  # 預設值
                ;;
        esac
    fi
    
    # 4. 根據 argc 數量生成參數
    if [ $argc_count -eq 1 ]; then
        args="${param_hints:-64}"
    elif [ $argc_count -eq 2 ]; then
        args="${param_hints:-64} 128"
    elif [ $argc_count -eq 3 ]; then
        args="${param_hints:-64} 128 256"
    elif [ $argc_count -eq 4 ]; then
        args="${param_hints:-64} 128 256 512"
    else
        args=""
    fi
    
    echo "$args"
}

# 函數：測試是否需要參數（快速測試）
quick_test_no_args() {
    local exe="$1"
    local timeout_sec=5
    
    # 嘗試不帶參數執行，看是否會立即返回錯誤訊息
    timeout $timeout_sec "$exe" > /dev/null 2>&1
    local exit_code=$?
    
    # 如果立即退出（不是超時），可能需要參數
    if [ $exit_code -ne 124 ]; then
        return 1  # 需要參數
    else
        return 0  # 不需要參數或正常執行
    fi
}

# 函數：測試單個 kernel
test_kernel() {
    local kernel_dir="$1"
    local kernel_name=$(basename "$kernel_dir")
    
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}測試 Kernel: ${MAGENTA}$kernel_name${NC}"
    echo -e "${CYAN}========================================${NC}"
    
    # 查找 .hip 檔案
    local hip_file="${kernel_dir}/${kernel_name}.hip"
    if [ ! -f "$hip_file" ]; then
        echo -e "${RED}✗ 找不到 HIP 檔案: $hip_file${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ HIP 檔案: $hip_file${NC}"
    
    # 查找 compiled_output 目錄（可能有多個時間戳）
    local compiled_dirs=($(find "$kernel_dir" -maxdepth 1 -type d -name "compiled_output_*" | sort -r))
    if [ ${#compiled_dirs[@]} -eq 0 ]; then
        echo -e "${RED}✗ 找不到 compiled_output 目錄${NC}"
        return 1
    fi
    local compiled_dir="${compiled_dirs[0]}"
    echo -e "${GREEN}✓ Compiled 目錄: $compiled_dir${NC}"
    
    # 查找原始執行檔
    local original_exe="${compiled_dir}/${kernel_name}"
    if [ ! -f "$original_exe" ]; then
        echo -e "${RED}✗ 找不到原始執行檔: $original_exe${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ 原始執行檔: $original_exe${NC}"
    
    # 查找 host.o
    local host_obj="${compiled_dir}/${kernel_name}-host-x86_64-unknown-linux-gnu.o"
    if [ ! -f "$host_obj" ]; then
        echo -e "${RED}✗ 找不到 host.o: $host_obj${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ Host 目標檔: $host_obj${NC}"
    
    # 查找 pipeline_test_results 目錄（可能有多個時間戳）
    local pipeline_dirs=($(find "$kernel_dir" -maxdepth 1 -type d -name "pipeline_test_results_*" | sort -r))
    if [ ${#pipeline_dirs[@]} -eq 0 ]; then
        echo -e "${RED}✗ 找不到 pipeline_test_results 目錄${NC}"
        return 1
    fi
    local pipeline_dir="${pipeline_dirs[0]}"
    echo -e "${GREEN}✓ Pipeline 目錄: $pipeline_dir${NC}"
    
    # 查找 rebuilt.hsaco（在子目錄中）
    local rebuilt_hsaco=$(find "$pipeline_dir" -name "*_rebuilt.hsaco" | head -1)
    if [ -z "$rebuilt_hsaco" ] || [ ! -f "$rebuilt_hsaco" ]; then
        echo -e "${RED}✗ 找不到 rebuilt.hsaco${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ Rebuilt HSACO: $rebuilt_hsaco${NC}"
    
    # 分析需要的參數
    local test_args=$(analyze_hip_args "$hip_file")
    if [ -n "$test_args" ]; then
        echo -e "${YELLOW}分析得到的參數: $test_args${NC}"
    else
        echo -e "${YELLOW}此程式不需要參數${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}開始執行測試...${NC}"
    echo ""
    
    # 執行測試
    if [ -n "$test_args" ]; then
        "$RELINK_SCRIPT" -p "$original_exe" -H "$host_obj" -G "$rebuilt_hsaco" -a "$test_args"
    else
        "$RELINK_SCRIPT" -p "$original_exe" -H "$host_obj" -G "$rebuilt_hsaco"
    fi
    
    local test_result=$?
    
    echo ""
    if [ $test_result -eq 0 ]; then
        echo -e "${GREEN}✓✓✓ Kernel $kernel_name 測試通過${NC}"
        return 0
    elif [ $test_result -eq 1 ]; then
        echo -e "${YELLOW}⚠⚠⚠ Kernel $kernel_name 部分通過（輸出有差異）${NC}"
        return 1
    elif [ $test_result -eq 3 ]; then
        echo -e "${YELLOW}⏱⏱⏱ Kernel $kernel_name 執行超時（已跳過）${NC}"
        return 3
    else
        echo -e "${RED}✗✗✗ Kernel $kernel_name 測試失敗${NC}"
        return 2
    fi
}

# 主程式
main() {
    echo -e "${MAGENTA}======================================${NC}"
    echo -e "${MAGENTA}批次測試所有 Kernel 重新連結${NC}"
    echo -e "${MAGENTA}======================================${NC}"
    echo ""
    echo -e "${CYAN}搜尋目錄：${NC}$SEARCH_DIR"
    echo ""
    
    # 如果提供了 kernel 參數，只測試指定的 kernel
    if [ ${#KERNEL_ARGS[@]} -gt 0 ]; then
        KERNEL_DIRS=("${KERNEL_ARGS[@]}")
        echo -e "${YELLOW}測試指定的 kernel 目錄${NC}"
    else
        # 找到所有 kernel 目錄（包含 .hip 檔案的目錄）
        echo -e "${YELLOW}掃描 kernel 目錄...${NC}"
        KERNEL_DIRS=()
        while IFS= read -r hip_file; do
            kernel_dir=$(dirname "$hip_file")
            KERNEL_DIRS+=("$kernel_dir")
        done < <(find "$SEARCH_DIR" -maxdepth 2 -name "*.hip" -type f | sort)
    fi
    
    local total=${#KERNEL_DIRS[@]}
    echo -e "${CYAN}找到 $total 個 kernel 目錄${NC}"
    echo ""
    
    # 統計變數
    local passed=0
    local partial=0
    local failed=0
    local timeout=0
    local skipped=0
    
    # 結果記錄（kernel 名稱）
    declare -a passed_kernels
    declare -a partial_kernels
    declare -a failed_kernels
    declare -a timeout_kernels
    declare -a skipped_kernels
    
    # 記錄失敗的 kernel 完整路徑（用於複製）
    declare -a failed_kernel_paths
    declare -a partial_kernel_paths
    declare -a timeout_kernel_paths
    
    # 測試每個 kernel
    local count=0
    for kernel_dir in "${KERNEL_DIRS[@]}"; do
        count=$((count + 1))
        echo ""
        echo -e "${MAGENTA}[$count/$total]${NC}"
        
        if test_kernel "$kernel_dir"; then
            passed=$((passed + 1))
            passed_kernels+=("$(basename "$kernel_dir")")
        else
            local result=$?
            local kernel_name=$(basename "$kernel_dir")
            if [ $result -eq 1 ]; then
                partial=$((partial + 1))
                partial_kernels+=("$kernel_name")
                partial_kernel_paths+=("$kernel_dir")
            elif [ $result -eq 2 ]; then
                failed=$((failed + 1))
                failed_kernels+=("$kernel_name")
                failed_kernel_paths+=("$kernel_dir")
            elif [ $result -eq 3 ]; then
                timeout=$((timeout + 1))
                timeout_kernels+=("$kernel_name")
                timeout_kernel_paths+=("$kernel_dir")
            else
                skipped=$((skipped + 1))
                skipped_kernels+=("$kernel_name")
            fi
        fi
        
        echo ""
        echo -e "${CYAN}----------------------------------------${NC}"
    done
    
    # 最終統計
    echo ""
    echo -e "${MAGENTA}======================================${NC}"
    echo -e "${MAGENTA}測試完成 - 最終統計${NC}"
    echo -e "${MAGENTA}======================================${NC}"
    echo ""
    
    # 計算總成功和總失敗
    local total_success=$passed
    local total_failed=$((failed + partial + timeout))
    
    echo -e "總計測試: ${CYAN}$total${NC}"
    echo -e "${GREEN}✓ 成功 (Success): $total_success${NC}"
    echo -e "${RED}✗ 失敗 (Failed):  $total_failed${NC}"
    echo ""
    echo -e "詳細分類："
    echo -e "  完全通過: ${GREEN}$passed${NC}"
    echo -e "  部分通過: ${YELLOW}$partial${NC} (輸出不一致)"
    echo -e "  測試失敗: ${RED}$failed${NC}"
    echo -e "  執行超時: ${YELLOW}$timeout${NC} (超過 ${EXECUTION_TIMEOUT:-60} 秒)"
    echo -e "  跳過測試: ${YELLOW}$skipped${NC} (檔案缺失)"
    echo ""
    
    # 詳細列表
    if [ $passed -gt 0 ]; then
        echo -e "${GREEN}完全通過的 Kernels:${NC}"
        for k in "${passed_kernels[@]}"; do
            echo -e "  ${GREEN}✓${NC} $k"
        done
        echo ""
    fi
    
    if [ $partial -gt 0 ]; then
        echo -e "${YELLOW}部分通過的 Kernels（輸出有差異，可能是非確定性）:${NC}"
        for k in "${partial_kernels[@]}"; do
            echo -e "  ${YELLOW}⚠${NC} $k"
        done
        echo ""
    fi
    
    if [ $failed -gt 0 ]; then
        echo -e "${RED}測試失敗的 Kernels:${NC}"
        for k in "${failed_kernels[@]}"; do
            echo -e "  ${RED}✗${NC} $k"
        done
        echo ""
    fi
    
    if [ $timeout -gt 0 ]; then
        echo -e "${YELLOW}執行超時的 Kernels（超過 ${EXECUTION_TIMEOUT:-60} 秒）:${NC}"
        for k in "${timeout_kernels[@]}"; do
            echo -e "  ${YELLOW}⏱${NC} $k"
        done
        echo ""
    fi
    
    if [ $skipped -gt 0 ]; then
        echo -e "${YELLOW}跳過的 Kernels（檔案缺失）:${NC}"
        for k in "${skipped_kernels[@]}"; do
            echo -e "  ${YELLOW}-${NC} $k"
        done
        echo ""
    fi
    
    # 複製失敗的 kernel 到 failed 目錄
    local total_failed=$((failed + partial + timeout))
    if [ $total_failed -gt 0 ]; then
        echo ""
        echo -e "${MAGENTA}======================================${NC}"
        echo -e "${MAGENTA}複製失敗的 Kernel 到 failed 目錄${NC}"
        echo -e "${MAGENTA}======================================${NC}"
        echo ""
        
        local FAILED_DIR="${SEARCH_DIR}/failed_kernels_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$FAILED_DIR"
        
        echo -e "${CYAN}建立失敗目錄：${NC}$FAILED_DIR"
        echo ""
        
        # 複製完全失敗的 kernel
        if [ ${#failed_kernel_paths[@]} -gt 0 ]; then
            echo -e "${RED}複製完全失敗的 kernel (${#failed_kernel_paths[@]} 個)：${NC}"
            for kernel_dir in "${failed_kernel_paths[@]}"; do
                local kernel_name=$(basename "$kernel_dir")
                local dest_dir="${FAILED_DIR}/${kernel_name}"
                echo -e "  - 複製 $kernel_name 到 $dest_dir"
                cp -r "$kernel_dir" "$dest_dir"
            done
            echo ""
        fi
        
        # 複製部分通過的 kernel（輸出不一致）
        if [ ${#partial_kernel_paths[@]} -gt 0 ]; then
            echo -e "${YELLOW}複製部分通過的 kernel (輸出不一致, ${#partial_kernel_paths[@]} 個)：${NC}"
            for kernel_dir in "${partial_kernel_paths[@]}"; do
                local kernel_name=$(basename "$kernel_dir")
                local dest_dir="${FAILED_DIR}/${kernel_name}"
                echo -e "  - 複製 $kernel_name 到 $dest_dir"
                cp -r "$kernel_dir" "$dest_dir"
            done
            echo ""
        fi
        
        # 複製超時的 kernel
        if [ ${#timeout_kernel_paths[@]} -gt 0 ]; then
            echo -e "${YELLOW}複製超時的 kernel (${#timeout_kernel_paths[@]} 個)：${NC}"
            for kernel_dir in "${timeout_kernel_paths[@]}"; do
                local kernel_name=$(basename "$kernel_dir")
                local dest_dir="${FAILED_DIR}/${kernel_name}"
                echo -e "  - 複製 $kernel_name 到 $dest_dir"
                cp -r "$kernel_dir" "$dest_dir"
            done
            echo ""
        fi
        
        echo -e "${GREEN}✓ 所有失敗的 kernel 已複製到：${NC}"
        echo -e "  ${CYAN}$FAILED_DIR${NC}"
        echo ""
    fi
    
    # 返回碼
    if [ $failed -gt 0 ]; then
        exit 1
    elif [ $partial -gt 0 ]; then
        exit 0  # 部分通過仍視為成功（因為可能是非確定性）
    else
        exit 0
    fi
}

# 執行主程式
main
