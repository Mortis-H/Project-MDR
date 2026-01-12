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
    echo "用法: $0 <目錄>"
    echo ""
    echo "這個腳本會遞迴搜尋指定目錄及其子資料夾，尋找成對的 ISA 文件："
    echo "  - *-hip-amdgcn-amd-amdhsa-*.s (舊的完整 ISA)"
    echo "  - *_func.s (新的純指令 ISA)"
    echo ""
    echo "並自動生成包裝後的完整 ISA 文件（放在各自的資料夾中）。"
    echo ""
    echo "範例:"
    echo "  $0 /path/to/isa/directory"
    echo "  $0 /path/to/run_success"
    exit 1
}

# 檢查參數
if [ $# -ne 1 ]; then
    usage
fi

TARGET_DIR="$1"

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
    fi
    echo ""
done < <(find "$TARGET_DIR" -type f -name "*_func.s" -print0)

# 顯示統計
echo -e "${GREEN}=== 處理完成 ===${NC}"
echo "總計: $total"
echo -e "${GREEN}成功: $success${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}失敗: $failed${NC}"
else
    echo "失敗: $failed"
fi
