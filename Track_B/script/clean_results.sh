#!/bin/bash

# 清除所有 kernel 資料夾底下的測試結果資料夾
# 使用方式：
#   1. 清除預設目錄的 pipeline 輸出（自動確認）：
#      ./clean_results.sh --type pipeline
#   
#   2. 清除預設目錄的 compile 輸出（自動確認）：
#      ./clean_results.sh --type compile
#
#   3. 清除預設目錄的 relink 輸出（自動確認）：
#      ./clean_results.sh --type relink
#
#   4. 清除預設目錄的 failed_kernels 資料夾（自動確認）：
#      ./clean_results.sh --type failed
#
#   5. 清除指定目錄的 relink 輸出：
#      ./clean_results.sh --type relink --path deterministic_kernels
#
#   6. 清除所有輸出：
#      ./clean_results.sh --type all
#
#   7. 互動確認模式：
#      ./clean_results.sh --type pipeline --no
#
#   8. 只列出不刪除：
#      ./clean_results.sh --type pipeline --dry-run

# 設定顏色輸出
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DEFAULT_BASE_DIR="/home/andycha/workspaces/multi_kernel_testcases"

# 解析參數
AUTO_YES=true  # 預設為自動確認
DRY_RUN=false
CLEAN_TYPE=""  # pipeline, compile, relink, failed, all
BASE_DIR=""    # 將透過參數或預設值設定

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                CLEAN_TYPE="$2"
                shift 2
            else
                echo -e "${RED}錯誤: --type 需要指定類型參數${NC}"
                exit 1
            fi
            ;;
        --type=*)
            CLEAN_TYPE="${1#*=}"
            shift
            ;;
        --path)
            if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
                BASE_DIR="$2"
                shift 2
            else
                echo -e "${RED}錯誤: --path 需要指定路徑參數${NC}"
                exit 1
            fi
            ;;
        --path=*)
            BASE_DIR="${1#*=}"
            shift
            ;;
        --yes|-y)
            AUTO_YES=true
            shift
            ;;
        --no|-n)
            AUTO_YES=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "使用方式："
            echo "  $0 --type TYPE [--path PATH] [選項]"
            echo ""
            echo "清除類型 (TYPE)："
            echo "  pipeline        清除 pipeline_test* 輸出"
            echo "  compile         清除 compiled_output* 輸出"
            echo "  relink          清除 relink_test_results* 輸出"
            echo "  failed          清除 failed_kernels* 資料夾"
            echo "  all             清除以上所有輸出"
            echo ""
            echo "選項："
            echo "  --path PATH     指定目標目錄路徑（預設：$DEFAULT_BASE_DIR）"
            echo "                  可以是絕對路徑或相對路徑"
            echo "  --yes           自動確認（預設）"
            echo "  --no            詢問確認"
            echo "  --dry-run       只列出不刪除"
            echo "  --help          顯示此說明"
            echo ""
            echo "範例："
            echo "  $0 --type pipeline                           # 清除預設目錄的 pipeline 輸出"
            echo "  $0 --type relink --path deterministic_kernels  # 清除特定目錄的 relink 輸出"
            echo "  $0 --type all --path ./my_kernels            # 清除指定目錄的所有輸出"
            echo "  $0 --type all --dry-run                      # 列出所有要清除的資料夾"
            echo "  $0 --type relink --path deterministic_kernels --no  # 互動確認"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}未知參數: $1${NC}"
            echo "使用 --help 查看說明"
            exit 1
            ;;
    esac
done

# 檢查是否指定清除類型
if [ -z "$CLEAN_TYPE" ]; then
    echo -e "${RED}錯誤: 必須指定清除類型${NC}"
    echo "使用 --type [pipeline|compile|relink|failed|all]"
    echo "使用 --help 查看完整說明"
    exit 1
fi

# 驗證清除類型
if [[ ! "$CLEAN_TYPE" =~ ^(pipeline|compile|relink|failed|all)$ ]]; then
    echo -e "${RED}錯誤: 無效的清除類型: $CLEAN_TYPE${NC}"
    echo "有效類型: pipeline, compile, relink, failed, all"
    exit 1
fi

# 處理目標目錄路徑
if [ -z "$BASE_DIR" ]; then
    # 如果未指定，使用預設目錄
    BASE_DIR="$DEFAULT_BASE_DIR"
else
    # 轉換為絕對路徑
    if [[ "$BASE_DIR" = /* ]]; then
        # 已經是絕對路徑
        :
    else
        # 相對路徑，轉換為絕對路徑
        BASE_DIR="$(cd "$(pwd)/$BASE_DIR" 2>/dev/null && pwd)"
        if [ -z "$BASE_DIR" ]; then
            echo -e "${RED}錯誤: 無效的路徑${NC}"
            exit 1
        fi
    fi
fi

# 檢查基本目錄是否存在
if [ ! -d "$BASE_DIR" ]; then
    echo -e "${RED}錯誤: 找不到目標目錄: $BASE_DIR${NC}"
    exit 1
fi

# 設定搜尋模式
declare -a SEARCH_PATTERNS
case "$CLEAN_TYPE" in
    pipeline)
        SEARCH_PATTERNS=("pipeline_test*")
        TYPE_DESC="pipeline_test 輸出"
        ;;
    compile)
        SEARCH_PATTERNS=("compiled_output*")
        TYPE_DESC="compiled_output 輸出"
        ;;
    relink)
        SEARCH_PATTERNS=("relink_test_results*")
        TYPE_DESC="relink_test_results 輸出"
        ;;
    failed)
        SEARCH_PATTERNS=("failed_kernels*")
        TYPE_DESC="failed_kernels 資料夾"
        ;;
    all)
        SEARCH_PATTERNS=("pipeline_test*" "compiled_output*" "relink_test_results*" "failed_kernels*")
        TYPE_DESC="pipeline_test, compiled_output, relink_test_results 和 failed_kernels 輸出"
        ;;
esac

echo -e "${BLUE}========================================${NC}"
echo "清理 $TYPE_DESC"
echo -e "${BLUE}========================================${NC}"
echo "基本目錄: $BASE_DIR"
echo "清除類型: $CLEAN_TYPE"
echo "搜尋模式: ${SEARCH_PATTERNS[*]}"
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}模式: 模擬執行（不會真的刪除）${NC}"
else
    if [ "$AUTO_YES" = true ]; then
        echo -e "${YELLOW}模式: 自動確認（不詢問直接刪除）${NC}"
    else
        echo -e "${YELLOW}模式: 互動確認（會詢問是否刪除）${NC}"
    fi
fi
echo -e "${BLUE}========================================${NC}"
echo ""

# 尋找所有符合模式的資料夾
echo "搜尋符合條件的資料夾..."
dirs_to_delete=()

# 分別處理不同類型的搜尋
for pattern in "${SEARCH_PATTERNS[@]}"; do
    if [[ "$pattern" == "failed_kernels"* ]]; then
        # failed_kernels 在頂層
        for dir in "$BASE_DIR"/$pattern/; do
            if [ -d "$dir" ]; then
                dirs_to_delete+=("$dir")
            fi
        done
    else
        # pipeline_test 和 compiled_output 在各個 kernel 資料夾內
        for kernel_dir in "$BASE_DIR"/*/; do
            # 檢查是否為目錄
            if [ ! -d "$kernel_dir" ]; then
                continue
            fi
            
            # 跳過 failed_kernels 資料夾本身
            kernel_name=$(basename "$kernel_dir")
            if [[ "$kernel_name" == failed_kernels* ]]; then
                continue
            fi
            
            for test_dir in "$kernel_dir"$pattern/; do
                if [ -d "$test_dir" ]; then
                    dirs_to_delete+=("$test_dir")
                fi
            done
        done
    fi
done

# 檢查是否找到任何資料夾
if [ ${#dirs_to_delete[@]} -eq 0 ]; then
    echo -e "${GREEN}沒有找到任何符合條件的資料夾${NC}"
    exit 0
fi

# 顯示找到的資料夾
echo -e "${YELLOW}找到 ${#dirs_to_delete[@]} 個資料夾：${NC}"
echo ""

total_size=0
for dir in "${dirs_to_delete[@]}"; do
    # 計算資料夾大小
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    size_bytes=$(du -sb "$dir" 2>/dev/null | cut -f1)
    total_size=$((total_size + size_bytes))
    
    # 提取相對路徑
    rel_path=${dir#$BASE_DIR/}
    echo "  [$size] $rel_path"
done

# 顯示總大小
total_size_human=$(numfmt --to=iec-i --suffix=B $total_size 2>/dev/null || echo "$total_size bytes")
echo ""
echo "總大小: $total_size_human"
echo ""

# 如果是 dry-run 模式，只列出不刪除
if [ "$DRY_RUN" = true ]; then
    echo -e "${GREEN}模擬執行完成（未刪除任何檔案）${NC}"
    exit 0
fi

# 詢問確認
if [ "$AUTO_YES" = false ]; then
    echo -e "${RED}警告: 這將永久刪除以上資料夾！${NC}"
    read -p "確定要繼續嗎? (yes/no): " response
    
    if [ "$response" != "yes" ]; then
        echo "取消操作"
        exit 0
    fi
fi

# 執行刪除
echo ""
echo "開始刪除..."
deleted_count=0
failed_count=0

for dir in "${dirs_to_delete[@]}"; do
    rel_path=${dir#$BASE_DIR/}
    
    if rm -rf "$dir" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} 已刪除: $rel_path"
        deleted_count=$((deleted_count + 1))
    else
        echo -e "${RED}✗${NC} 刪除失敗: $rel_path"
        failed_count=$((failed_count + 1))
    fi
done

# 顯示結果
echo ""
echo -e "${BLUE}========================================${NC}"
echo "清理完成！"
echo "  成功刪除: $deleted_count 個資料夾"
if [ $failed_count -gt 0 ]; then
    echo -e "  ${RED}刪除失敗: $failed_count 個資料夾${NC}"
fi
echo "  釋放空間: $total_size_human"
echo -e "${BLUE}========================================${NC}"
