#!/bin/bash
#
# 批量 ISA 包裝腳本
# 用於批量處理多個 LLM 生成的 ISA 文件
# 預設遞迴處理所有子資料夾
#

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 使用說明
usage() {
    echo "用法: $0 -p PATH [選項]"
    echo ""
    echo "這個腳本會遞迴搜尋指定目錄及其子資料夾，尋找成對的 ISA 文件："
    echo "  - *-hip-amdgcn-amd-amdhsa-*.s (舊的完整 ISA)"
    echo "  - *_func.s (新的純指令 ISA)"
    echo ""
    echo "並自動生成包裝後的完整 ISA 文件（放在各自的資料夾中）。"
    echo ""
    echo "選項:"
    echo "  -p PATH              指定目標目錄路徑（必需）"
    echo "  -k, --kernel HASH    只處理指定的 kernel（kernel hash）"
    echo "  -h, --help           顯示此說明"
    echo ""
    echo "範例:"
    echo "  $0 -p /path/to/isa/directory                    # 處理所有 kernel"
    echo "  $0 -p /path/to/run_success                      # 處理所有 kernel"
    echo "  $0 -p /path/to/run_success -k 0a32840d...       # 只處理指定的 kernel"
    exit 1
}

# 解析參數
TARGET_DIR=""
SPECIFIC_KERNEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -p)
            TARGET_DIR="$2"
            shift 2
            ;;
        -k|--kernel)
            SPECIFIC_KERNEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo -e "${RED}未知選項: $1${NC}"
            usage
            ;;
        *)
            echo -e "${RED}錯誤: 未知參數 $1${NC}"
            echo "請使用 -p 指定目錄路徑"
            usage
            ;;
    esac
done

# 檢查是否指定了目錄
if [ -z "$TARGET_DIR" ]; then
    echo -e "${RED}錯誤: 必須指定目錄${NC}"
    usage
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${RED}錯誤: 目錄不存在: $TARGET_DIR${NC}"
    exit 1
fi

# 腳本路徑
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRAP_SCRIPT="$SCRIPT_DIR/wrap_isa.py"

if [ ! -f "$WRAP_SCRIPT" ]; then
    echo -e "${RED}錯誤: 找不到 wrap_isa.py 腳本: $WRAP_SCRIPT${NC}"
    exit 1
fi

echo -e "${GREEN}=== ISA 批量包裝工具（遞迴模式）===${NC}"
echo "目標目錄: $TARGET_DIR"
echo "包裝腳本: $WRAP_SCRIPT"
echo ""

# 統計
total=0
success=0
failed=0

# 記錄失敗的 kernel 目錄（使用關聯陣列）
declare -a failed_kernels

# 確定搜尋路徑
if [ -n "$SPECIFIC_KERNEL" ]; then
    # 指定了特定 kernel，只在該目錄下搜尋
    SEARCH_PATH="${TARGET_DIR}/${SPECIFIC_KERNEL}"
    if [ ! -d "$SEARCH_PATH" ]; then
        echo -e "${RED}錯誤: 找不到指定的 kernel 目錄: $SEARCH_PATH${NC}"
        exit 1
    fi
    echo -e "${YELLOW}只處理指定的 kernel: $SPECIFIC_KERNEL${NC}"
    echo ""
else
    # 處理整個目錄
    SEARCH_PATH="$TARGET_DIR"
fi

# 遞迴尋找所有 *_func.s 文件
while IFS= read -r -d '' new_isa; do
    total=$((total + 1))
    
    # 取得檔案所在的目錄
    file_dir=$(dirname "$new_isa")
    
    # 取得基礎名稱（移除 _func.s）
    base_name=$(basename "$new_isa" _func.s)
    
    # 在同一目錄下尋找對應的舊 ISA 文件
    old_isa=$(find "$file_dir" -maxdepth 1 -name "${base_name}-hip-amdgcn-amd-amdhsa-*.s" | head -n 1)
    
    if [ -z "$old_isa" ] || [ ! -f "$old_isa" ]; then
        echo -e "${YELLOW}[跳過]${NC} $base_name ($(realpath --relative-to="$TARGET_DIR" "$file_dir"))"
        echo "  原因: 找不到對應的舊 ISA 文件"
        failed=$((failed + 1))
        failed_kernels+=("$file_dir")
        echo ""
        continue
    fi
    
    # 輸出文件名（放在同一目錄）
    output_isa="$file_dir/${base_name}_wrapped.s"
    
    # 顯示相對路徑
    rel_path=$(realpath --relative-to="$TARGET_DIR" "$file_dir")
    if [ "$rel_path" = "." ]; then
        echo -e "${GREEN}[處理]${NC} $base_name"
    else
        echo -e "${GREEN}[處理]${NC} $base_name ($rel_path)"
    fi
    echo "  舊 ISA: $(basename "$old_isa")"
    echo "  新 ISA: $(basename "$new_isa")"
    echo "  輸出: $(basename "$output_isa")"
    
    # 執行包裝
    if python3 "$WRAP_SCRIPT" -o "$old_isa" -n "$new_isa" -O "$output_isa" > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ 完成${NC}"
        success=$((success + 1))
    else
        echo -e "${RED}  ✗ 失敗${NC}"
        failed=$((failed + 1))
        failed_kernels+=("$file_dir")
    fi
    echo ""
done < <(find "$SEARCH_PATH" -type f -name "*_func.s" -print0)

# 顯示統計
echo -e "${GREEN}=== 處理完成 ===${NC}"
echo "總計: $total"
echo -e "${GREEN}成功: $success${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}失敗: $failed${NC}"
else
    echo "失敗: $failed"
fi
echo ""

# 複製失敗的 kernel 資料夾
if [ ${#failed_kernels[@]} -gt 0 ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    FAIL_DIR="${TARGET_DIR}/fail_wrap_${TIMESTAMP}"
    
    echo -e "${YELLOW}=== 複製失敗的 Kernel 資料夾 ===${NC}"
    echo "目標目錄: $FAIL_DIR"
    echo "失敗數量: ${#failed_kernels[@]}"
    echo ""
    
    mkdir -p "$FAIL_DIR"
    
    # 去重並複製（使用 associative array 去重）
    declare -A seen
    for kernel_dir in "${failed_kernels[@]}"; do
        # 獲取 kernel 資料夾的絕對路徑
        kernel_abs_path=$(realpath "$kernel_dir")
        
        # 如果已經處理過這個目錄，跳過
        if [ -n "${seen[$kernel_abs_path]}" ]; then
            continue
        fi
        seen[$kernel_abs_path]=1
        
        # 獲取 kernel 資料夾名稱
        kernel_name=$(basename "$kernel_abs_path")
        
        # 獲取相對路徑（用於顯示）
        rel_path=$(realpath --relative-to="$TARGET_DIR" "$kernel_abs_path")
        
        echo -e "${YELLOW}複製:${NC} $rel_path"
        
        # 複製整個 kernel 資料夾
        cp -r "$kernel_abs_path" "$FAIL_DIR/" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✓ 已複製到: fail_wrap_${TIMESTAMP}/${kernel_name}${NC}"
        else
            echo -e "${RED}  ✗ 複製失敗${NC}"
        fi
    done
    
    echo ""
    echo -e "${GREEN}失敗的 Kernel 已複製到: $FAIL_DIR${NC}"
    echo ""
fi