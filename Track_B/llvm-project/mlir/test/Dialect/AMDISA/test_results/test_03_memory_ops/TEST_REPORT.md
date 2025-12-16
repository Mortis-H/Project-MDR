# test_03_memory_ops 測試報告

## 測試日期
2025-12-16 05:24:11

## 測試結果
**狀態**: ✅ 通過

## Pipeline 流程

```
bundled.s (573 行) - hipcc 輸出
    ↓ extract device assembly
original.s (215 行)
    ↓ amdisa-translate -x s -emit mlir
stage1_amdisa.mlir (24 行)
    ↓ amdisa-translate -x mlir -emit gpuinlineasm
stage2_gpu.mlir (9 行)
    ↓ amdisa-translate -x mlir -emit s
stage3_rebuilt.s (215 行)
```

## 編譯驗證

| 檔案 | 狀態 | 大小 |
|------|------|------|
| original.o | ✅ | 5976 bytes |
| rebuilt.o | ✅ | 5976 bytes |

**結果**: 檔案大小完全一致

## 耗時
1 秒
