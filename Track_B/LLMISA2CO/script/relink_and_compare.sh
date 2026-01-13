#!/bin/bash

# 腳本：重新連結 GPU 代碼並比較執行結果
# 用法：./relink_and_compare.sh -p <原始執行檔> -H <host.o> -G <rebuilt.hsaco> [-a "測試參數"]

set -e  # 遇到錯誤立即退出

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 超時設定（秒）
EXECUTION_TIMEOUT=${EXECUTION_TIMEOUT:-60}  # 預設 60 秒，可透過環境變數覆蓋

# 函數：顯示使用方法
usage() {
    echo "用法: $0 -p <原始執行檔> [選項]"
    echo ""
    echo "必需參數:"
    echo "  -p PATH         原始編譯的可執行檔路徑"
    echo ""
    echo "可選參數:"
    echo "  -H PATH         主機端目標檔案路徑 (host-x86_64-unknown-linux-gnu.o)"
    echo "  -G PATH         Pipeline 重建的 GPU 代碼檔案 (.hsaco)"
    echo "  -a ARGS         執行檔所需的測試參數（用引號包起來）"
    echo "  -h, --help      顯示此說明"
    echo ""
    echo "環境變數:"
    echo "  EXECUTION_TIMEOUT  - 執行超時時間（秒），預設 60"
    echo "  OUTPUT_BASE_DIR    - 輸出目錄基礎路徑，預設為 kernel 根目錄"
    echo ""
    echo "範例:"
    echo "  $0 -p ./original_exe -H ./host.o -G ./rebuilt.hsaco -a \"64\""
    echo "  $0 -p ./original_exe -H ./host.o -G ./rebuilt.hsaco -a \"128 extra_arg\""
    echo ""
    echo "  # 自訂輸出目錄"
    echo "  OUTPUT_BASE_DIR=/tmp/test_results $0 -p ./exe -H ./host.o -G ./hsaco -a \"64\""
    echo ""
    echo "  # 自訂超時時間"
    echo "  EXECUTION_TIMEOUT=120 $0 -p ./exe -H ./host.o -G ./hsaco -a \"64\""
    exit 1
}

# 解析參數
ORIGINAL_EXE=""
HOST_OBJ=""
REBUILT_HSACO=""
TEST_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -p)
            ORIGINAL_EXE="$2"
            shift 2
            ;;
        -H)
            HOST_OBJ="$2"
            shift 2
            ;;
        -G)
            REBUILT_HSACO="$2"
            shift 2
            ;;
        -a)
            TEST_ARGS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}未知參數: $1${NC}"
            usage
            ;;
    esac
done

# 檢查必需參數
if [ -z "$ORIGINAL_EXE" ]; then
    echo -e "${RED}錯誤：必須使用 -p 指定原始執行檔${NC}"
    usage
fi

if [ -z "$HOST_OBJ" ]; then
    echo -e "${RED}錯誤：必須使用 -H 指定 host.o 文件${NC}"
    usage
fi

if [ -z "$REBUILT_HSACO" ]; then
    echo -e "${RED}錯誤：必須使用 -G 指定 rebuilt.hsaco 文件${NC}"
    usage
fi

# 檢查檔案是否存在
if [ ! -f "$ORIGINAL_EXE" ]; then
    echo -e "${RED}錯誤：原始執行檔不存在: $ORIGINAL_EXE${NC}"
    exit 1
fi

if [ ! -f "$HOST_OBJ" ]; then
    echo -e "${RED}錯誤：Host 目標檔案不存在: $HOST_OBJ${NC}"
    exit 1
fi

if [ ! -f "$REBUILT_HSACO" ]; then
    echo -e "${RED}錯誤：Rebuilt HSACO 檔案不存在: $REBUILT_HSACO${NC}"
    exit 1
fi

# 檢查原始執行檔是否可執行
if [ ! -x "$ORIGINAL_EXE" ]; then
    echo -e "${YELLOW}警告：原始執行檔沒有執行權限，正在添加...${NC}"
    chmod +x "$ORIGINAL_EXE"
fi

# 建立工作目錄
# 優先使用環境變數指定的輸出目錄，否則自動推斷 kernel 根目錄
if [ -n "$OUTPUT_BASE_DIR" ]; then
    # 用戶指定了輸出基礎目錄
    KERNEL_ROOT="$OUTPUT_BASE_DIR"
else
    # 自動推斷 kernel 根目錄
    # 從原始執行檔路徑推斷（exe 在 compiled_output_* 目錄中）
    KERNEL_ROOT=$(dirname "$(dirname "$ORIGINAL_EXE")")
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${KERNEL_ROOT}/relink_test_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}GPU 代碼重新連結與比較測試${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "${YELLOW}Kernel 根目錄：${NC}$KERNEL_ROOT"
echo -e "${YELLOW}輸出目錄：${NC}$OUTPUT_DIR"
echo -e "${YELLOW}原始執行檔：${NC}$ORIGINAL_EXE"
echo -e "${YELLOW}Host 目標檔：${NC}$HOST_OBJ"
echo -e "${YELLOW}Rebuilt HSACO：${NC}$REBUILT_HSACO"
echo -e "${YELLOW}測試參數：${NC}${TEST_ARGS:-（無）}"
echo -e "${YELLOW}執行超時：${NC}${EXECUTION_TIMEOUT}秒"
echo ""

# 步驟 1：創建新的 HIP Fat Binary
echo -e "${BLUE}[步驟 1/5]${NC} 創建 HIP Fat Binary..."
HIPFB_FILE="${OUTPUT_DIR}/relinked.hipfb"

clang-offload-bundler \
    --type=o \
    --bundle-align=4096 \
    --targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx950 \
    --input="$HOST_OBJ" \
    --input="$REBUILT_HSACO" \
    --output="$HIPFB_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Fat Binary 創建成功：$HIPFB_FILE${NC}"
else
    echo -e "${RED}✗ Fat Binary 創建失敗${NC}"
    exit 1
fi

# 步驟 2：連結成可執行檔
echo -e "${BLUE}[步驟 2/5]${NC} 連結新的可執行檔..."
RELINKED_EXE="${OUTPUT_DIR}/relinked_executable"

hipcc -o "$RELINKED_EXE" "$HIPFB_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 可執行檔連結成功：$RELINKED_EXE${NC}"
    ls -lh "$RELINKED_EXE"
else
    echo -e "${RED}✗ 可執行檔連結失敗${NC}"
    exit 1
fi

# 步驟 3：驗證可執行檔
echo -e "${BLUE}[步驟 3/5]${NC} 驗證可執行檔..."
echo "原始執行檔："
file "$ORIGINAL_EXE"
echo ""
echo "重新連結的執行檔："
file "$RELINKED_EXE"
echo ""

# 比較檔案大小
ORIG_SIZE=$(stat -c%s "$ORIGINAL_EXE")
RELINK_SIZE=$(stat -c%s "$RELINKED_EXE")
echo "檔案大小比較："
echo "  原始：$ORIG_SIZE bytes"
echo "  重連結：$RELINK_SIZE bytes"
echo ""

# 步驟 4：執行原始執行檔
echo -e "${BLUE}[步驟 4/5]${NC} 執行原始執行檔（超時：${EXECUTION_TIMEOUT}秒）..."
ORIGINAL_OUTPUT="${OUTPUT_DIR}/original_output.txt"
ORIGINAL_STDERR="${OUTPUT_DIR}/original_stderr.txt"
ORIGINAL_EXIT_CODE=0

if [ -n "$TEST_ARGS" ]; then
    echo "執行命令：timeout ${EXECUTION_TIMEOUT}s $ORIGINAL_EXE $TEST_ARGS"
    timeout ${EXECUTION_TIMEOUT}s $ORIGINAL_EXE $TEST_ARGS > "$ORIGINAL_OUTPUT" 2> "$ORIGINAL_STDERR" || ORIGINAL_EXIT_CODE=$?
else
    echo -e "${YELLOW}警告：沒有提供測試參數，某些程式可能需要參數才能正常執行${NC}"
    echo "嘗試執行：timeout ${EXECUTION_TIMEOUT}s $ORIGINAL_EXE"
    timeout ${EXECUTION_TIMEOUT}s $ORIGINAL_EXE > "$ORIGINAL_OUTPUT" 2> "$ORIGINAL_STDERR" || ORIGINAL_EXIT_CODE=$?
fi

# 檢查是否超時
if [ $ORIGINAL_EXIT_CODE -eq 124 ]; then
    echo -e "${RED}✗ 原始執行檔執行超時（超過 ${EXECUTION_TIMEOUT} 秒）${NC}"
    echo "建議："
    echo "  1. 增加超時時間：export EXECUTION_TIMEOUT=120"
    echo "  2. 使用更小的測試參數"
    echo "  3. 跳過此 kernel"
    exit 3  # 特殊退出碼表示超時
else
    echo -e "${GREEN}✓ 原始執行檔執行完成（退出碼：$ORIGINAL_EXIT_CODE）${NC}"
fi
echo "輸出已保存至：$ORIGINAL_OUTPUT"
echo "錯誤輸出已保存至：$ORIGINAL_STDERR"
echo ""

# 步驟 5：執行重新連結的執行檔
echo -e "${BLUE}[步驟 5/5]${NC} 執行重新連結的執行檔（超時：${EXECUTION_TIMEOUT}秒）..."
RELINKED_OUTPUT="${OUTPUT_DIR}/relinked_output.txt"
RELINKED_STDERR="${OUTPUT_DIR}/relinked_stderr.txt"
RELINKED_EXIT_CODE=0

if [ -n "$TEST_ARGS" ]; then
    echo "執行命令：timeout ${EXECUTION_TIMEOUT}s $RELINKED_EXE $TEST_ARGS"
    timeout ${EXECUTION_TIMEOUT}s $RELINKED_EXE $TEST_ARGS > "$RELINKED_OUTPUT" 2> "$RELINKED_STDERR" || RELINKED_EXIT_CODE=$?
else
    echo "執行：timeout ${EXECUTION_TIMEOUT}s $RELINKED_EXE"
    timeout ${EXECUTION_TIMEOUT}s $RELINKED_EXE > "$RELINKED_OUTPUT" 2> "$RELINKED_STDERR" || RELINKED_EXIT_CODE=$?
fi

# 檢查是否超時
if [ $RELINKED_EXIT_CODE -eq 124 ]; then
    echo -e "${RED}✗ 重新連結執行檔執行超時（超過 ${EXECUTION_TIMEOUT} 秒）${NC}"
    echo "建議："
    echo "  1. 增加超時時間：export EXECUTION_TIMEOUT=120"
    echo "  2. 使用更小的測試參數"
    echo "  3. 跳過此 kernel"
    exit 3  # 特殊退出碼表示超時
else
    echo -e "${GREEN}✓ 重新連結執行檔執行完成（退出碼：$RELINKED_EXIT_CODE）${NC}"
fi
echo "輸出已保存至：$RELINKED_OUTPUT"
echo "錯誤輸出已保存至：$RELINKED_STDERR"
echo ""

# 比較結果
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}結果比較${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# 比較退出碼
echo "退出碼比較："
if [ $ORIGINAL_EXIT_CODE -eq $RELINKED_EXIT_CODE ]; then
    echo -e "  ${GREEN}✓ 相同${NC}（原始：$ORIGINAL_EXIT_CODE，重連結：$RELINKED_EXIT_CODE）"
    EXIT_CODE_MATCH=1
else
    echo -e "  ${RED}✗ 不同${NC}（原始：$ORIGINAL_EXIT_CODE，重連結：$RELINKED_EXIT_CODE）"
    EXIT_CODE_MATCH=0
fi
echo ""

# 比較標準輸出
echo "標準輸出比較："
DIFF_OUTPUT="${OUTPUT_DIR}/output_diff.txt"
if diff -u "$ORIGINAL_OUTPUT" "$RELINKED_OUTPUT" > "$DIFF_OUTPUT" 2>&1; then
    echo -e "  ${GREEN}✓ 標準輸出完全相同${NC}"
    STDOUT_MATCH=1
else
    echo -e "  ${RED}✗ 標準輸出有差異${NC}"
    echo "  差異已保存至：$DIFF_OUTPUT"
    echo ""
    echo "差異內容預覽（前 20 行）："
    head -20 "$DIFF_OUTPUT"
    STDOUT_MATCH=0
fi
echo ""

# 比較標準錯誤輸出
echo "標準錯誤輸出比較："
DIFF_STDERR="${OUTPUT_DIR}/stderr_diff.txt"
if diff -u "$ORIGINAL_STDERR" "$RELINKED_STDERR" > "$DIFF_STDERR" 2>&1; then
    echo -e "  ${GREEN}✓ 標準錯誤輸出完全相同${NC}"
    STDERR_MATCH=1
else
    echo -e "  ${YELLOW}⚠ 標準錯誤輸出有差異${NC}"
    echo "  差異已保存至：$DIFF_STDERR"
    echo ""
    echo "差異內容預覽（前 20 行）："
    head -20 "$DIFF_STDERR"
    STDERR_MATCH=0
fi
echo ""

# 顯示輸出內容
echo -e "${BLUE}原始執行檔輸出內容：${NC}"
echo "----------------------------------------"
cat "$ORIGINAL_OUTPUT"
echo "----------------------------------------"
echo ""

echo -e "${BLUE}重新連結執行檔輸出內容：${NC}"
echo "----------------------------------------"
cat "$RELINKED_OUTPUT"
echo "----------------------------------------"
echo ""

# 最終結論
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}測試總結${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

if [ $EXIT_CODE_MATCH -eq 1 ] && [ $STDOUT_MATCH -eq 1 ]; then
    echo -e "${GREEN}✓✓✓ 測試通過！重新連結的執行檔與原始執行檔行為完全一致${NC}"
    TEST_RESULT=0
elif [ $EXIT_CODE_MATCH -eq 1 ] && [ $STDOUT_MATCH -eq 0 ]; then
    echo -e "${YELLOW}⚠ 部分通過：退出碼相同但輸出有差異${NC}"
    TEST_RESULT=1
else
    echo -e "${RED}✗✗✗ 測試失敗！重新連結的執行檔與原始執行檔行為不一致${NC}"
    TEST_RESULT=2
fi

echo ""
echo -e "${CYAN}結果檔案位置：${NC}"
echo -e "  ${YELLOW}Kernel 目錄：${NC}$KERNEL_ROOT"
echo -e "  ${YELLOW}輸出目錄：${NC}$(basename "$OUTPUT_DIR")"
echo ""
echo "所有結果檔案已保存至："
echo "  $OUTPUT_DIR"
echo ""
echo "檔案清單："
echo "  - 原始輸出：$(basename "$ORIGINAL_OUTPUT")"
echo "  - 重連結輸出：$(basename "$RELINKED_OUTPUT")"
echo "  - 輸出差異：$(basename "$DIFF_OUTPUT")"
echo "  - Fat Binary：$(basename "$HIPFB_FILE")"
echo "  - 重連結執行檔：$(basename "$RELINKED_EXE")"
echo ""

exit $TEST_RESULT
