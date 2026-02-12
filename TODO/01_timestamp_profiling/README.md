# MDR @TIMESTAMP Handoff Package

這是 MDR `@TIMESTAMP` 功能開發的經驗報告與範例程式碼。

## 檔案說明

| 檔案 | 說明 |
|------|------|
| **MDR_TIMESTAMP_REPORT.md** | 完整技術報告（關鍵發現、實驗結果、使用建議）|
| **example.s** | 使用範例（vectorAdd kernel 加上 @TIMESTAMP）|
| **quick_start.sh** | 快速測試腳本 |

## 快速開始

```bash
# 設定路徑（根據新 repo 位置調整）
export MDR_PRINTF="/path/to/mdr_printf.py"
export HSACO_RUNNER="/path/to/universal_hsaco_runner"

# 執行測試
chmod +x quick_start.sh
./quick_start.sh
```

## 核心發現

1. **`s_memtime` 後必須加 `s_waitcnt lgkmcnt(0)`** - 否則會讀到錯誤值
2. **快照機制有效隔離 printf 開銷** - 測量精確度與 rocprofv2 吻合
3. **`s_memtime` 是 per-CU 計數器** - 跨 workgroup 要用 `s_memrealtime`

## 詳細內容

請閱讀 **MDR_TIMESTAMP_REPORT.md**

---

*2026-01-29*
