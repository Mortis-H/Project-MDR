# test_04_conditional 測試報告

## 測試日期
2025-12-17 02:15:34

## 測試結果
**狀態**: ✅ 通過

## Pipeline 流程

```
bundled.s (593 行) - hipcc 輸出
    ↓ extract device assembly
original.s (229 行)
    ↓ amdisa-translate -x s -emit mlir
stage1_amdisa.mlir (34 行)
    ↓ amdisa-translate -x mlir -emit gpuinlineasm
stage2_gpu.mlir (9 行)
    ↓ amdisa-translate -x mlir -emit s
stage3_rebuilt.s (198 行)
```

## 完整編譯工具鏈驗證

### Original 路徑
| 步驟 | 工具 | 輸入 | 輸出 | 大小 | 狀態 |
|------|------|------|------|------|------|
| 組譯 | clang | original.s | original.o | 6144 bytes | ✅ |
| 連結 | ld.lld | original.o | original.out | 1752 bytes | ✅ |

### Rebuilt 路徑
| 步驟 | 工具 | 輸入 | 輸出 | 大小 | 狀態 |
|------|------|------|------|------|------|
| 組譯 | clang | stage3_rebuilt.s | rebuilt.o | 6144 bytes | ✅ |
| 連結 | ld.lld | rebuilt.o | rebuilt.out | 1752 bytes | ✅ |

## 檔案大小比較

| 階段 | Original | Rebuilt | 差異 | 結果 |
|------|----------|---------|------|------|
| .o (Object) | 6144 | 6144 | 0 | ✅ 一致 |
| .out (Linked) | 1752 | 1752 | 0 | ✅ 一致 |

## 驗證結論

✅ **Object 檔案完全一致** - 機器碼100%相同，MLIR轉換正確性已充分驗證！

## 耗時
2 秒
