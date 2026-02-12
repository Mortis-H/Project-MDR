#!/bin/bash
# =============================================================================
# MDR @TIMESTAMP 快速測試腳本
# =============================================================================
#
# 用法：
#   ./quick_start.sh
#
# 需求：
#   - mdr_printf.py 在 PATH 或指定路徑
#   - universal_hsaco_runner 已編譯
#   - AMD GPU (gfx950)
#
# =============================================================================

set -e

# ===== 設定路徑（請根據實際環境修改）=====
MDR_PRINTF="${MDR_PRINTF:-../../mdr_printf.py}"
HSACO_RUNNER="${HSACO_RUNNER:-../../Track_B/kernel_testcases/universal_hsaco_runner}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
EXAMPLE_S="${SCRIPT_DIR}/example.s"

# =============================================================================
echo "=============================================="
echo "MDR @TIMESTAMP 快速測試"
echo "=============================================="
echo ""

# 檢查必要檔案
if [ ! -f "${MDR_PRINTF}" ]; then
    echo "❌ 找不到 mdr_printf.py: ${MDR_PRINTF}"
    echo "   請設定 MDR_PRINTF 環境變數"
    exit 1
fi

if [ ! -f "${HSACO_RUNNER}" ]; then
    echo "❌ 找不到 universal_hsaco_runner: ${HSACO_RUNNER}"
    echo "   請設定 HSACO_RUNNER 環境變數"
    exit 1
fi

# Step 1: 編譯
echo "[Step 1] 編譯 example.s ..."
mkdir -p "${OUTPUT_DIR}"
python3 "${MDR_PRINTF}" "${EXAMPLE_S}" --output-dir "${OUTPUT_DIR}"

if [ $? -ne 0 ]; then
    echo "❌ 編譯失敗"
    exit 1
fi
echo "✅ 編譯成功"
echo ""

# Step 2: 執行
HSACO="${OUTPUT_DIR}/example_debug_injected.hsaco"
KERNEL_NAME="_Z9vectorAddPKfS0_Pfi"

echo "[Step 2] 執行 kernel ..."
echo "命令: ${HSACO_RUNNER} ${HSACO} ${KERNEL_NAME} float_add 64"
echo ""
echo "--- 輸出開始 ---"
"${HSACO_RUNNER}" "${HSACO}" "${KERNEL_NAME}" float_add 64
echo "--- 輸出結束 ---"
echo ""

# 完成
echo "=============================================="
echo "測試完成！"
echo ""
echo "預期輸出應包含："
echo "  [Timestamp kernel_total] elapsed = XXXX ticks"
echo "  ✅ PASS: All 64 elements correct"
echo ""
echo "如果看到以上輸出，表示 @TIMESTAMP 功能正常。"
echo "=============================================="
