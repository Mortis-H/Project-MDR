mdr_cap.py 使用說明
===================

此工具會在 AMD ISA `.s` 檔案中插入 `@CAPTURE` 指令，將暫存器或表達式結果寫回指定的 register，並自動生成對應的 HSACO 與 mapping 檔。

基本流程
--------
1. 在 `.s` 檔中加入 `@CAPTURE` 註解（建議使用 f-string）。
2. 執行 `mdr_cap.py` 產生注入後的 ISA 與 HSACO。
3. 依輸出檔案觀察注入結果與 mapping。

@CAPTURE 輸入範例（i64 / SGPR pair）
---------------------------------
```
; @CAPTURE f"{(s[12:13] + s[16:17]*s4*4):ld}" dst=s[12:13]
; @CAPTURE f"{(s[14:15] + s[18:19]*s4*4):ld}" dst=s[14:15]
```

指令範例
--------
以下範例假設在含有 `cap.s` 的目錄執行：

```
../../mdr_capture/mdr_cap.py cap.s --chip=gfx942
```

常見輸出
--------
- `cap_output/<input>_injected.s`：插入 capture 指令後的 ISA
- `cap_output/<input>_clobber.gpumlir`：含 clobber 的 GPU MLIR
- `cap_output/<input>_final.s`：最終 ISA
- `cap_output/<input>.hsaco`：可執行 HSACO
- `cap_output/capture_mapping.txt`：capture 映射資訊

Demo
----
- https://asciinema.org/a/hdQWGlGlU2aiqxjX
