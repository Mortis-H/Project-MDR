# SP3 Printf Debug 工具鏈

本專案提供一套從 SP3 原始碼中標註 `@PRINT` 指令、自動注入 GPU printf 除錯碼、產生可執行 HSACO 的工具鏈。

---

## 目錄

1. [Docker 環境設定](#docker-環境設定)
2. [範例一：INT32 Kernel](#範例一int32-kernel)
3. [範例二：INT8 Kernel](#範例二int8-kernel)
4. [補充：為個別 SP3 生成 Host Code](#補充為個別-sp3-生成-host-code)

---

## Docker 環境設定

### 啟動 Docker 容器

```bash
sg docker -c "docker run -d --name uc_moe_INT32_env_andy \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --shm-size 64G \
  -v /home/root123/andy-mdr/models:/models \
  -v /home/root123/andy-mdr:/workspace/andy \
  mortis_moe_int32:v2.3-scale32768 \
  sleep infinity"
```

### 進入容器

```bash
sudo docker exec -it uc_moe_INT32_env_andy /bin/bash
```

### 容器內環境變數

進入容器後，需設定以下環境變數：

```bash
export LD_LIBRARY_PATH="/workspace/poc_kl/scripts/common:$LD_LIBRARY_PATH"
export PATH=/workspace/andy/Project-MDR/Track_B/amdisa-toolkit/build/bin:$PATH
export PATH=/workspace/andy/llvm-install/bin:$PATH
```

---

## 範例一：INT32 Kernel

以 `FMOE_192_int32_full_scale32768.sp3` 為例，展示從 SP3 源碼中的 `@PRINT` 標註到最終執行輸出的完整流程。

### 1. SP3 源碼中的 @PRINT 標註

在 SP3 原始碼 `FMOE_192_int32_full_scale32768.sp3` (第 1654-1657 行) 中，於 `s_addk_i32` 之後加入 `@PRINT` 指令：

```
s_addk_i32        s_loop_idx,    gemm0_SUB_K
// @PRINT max=16 if $tgid_x == 0 && $tgid_y == 0 && $tgid_z == 0 && $tid == 0: f"s80={s80:d}"
s_cmp_lt_i32      s_loop_idx,    s_loop_cnt
s_cbranch_scc0    label_actv[cl_p]
```

**說明：**
- `max=16`：最多輸出 16 次（用於迴圈中避免輸出過多）
- `if $tgid_x == 0 && ... && $tid == 0`：僅當 thread group ID 和 thread ID 都為 0 時才印出
- `f"s80={s80:d}"`：以十進位格式印出 sgpr s80（即 `s_loop_idx`）的值

### 2. 執行 sp3_printf_wrap.py

```bash
python3 sp3_printf_wrap.py FMOE_192_int32_full_scale32768.sp3 --output-dir kernel_INT32 --insert-mode all
```

此命令執行 8 個步驟的管線：
1. **SP3 -> Binary**：使用 SP3 compiler 編譯為二進位
2. **Binary -> Disasm SP3**：反組譯回 SP3 格式
3. **Inject directives**：將 `@PRINT` 指令注入到反組譯的 SP3 中
4. **SP3 -> LLVM Assembly**：轉換為 `.s` 組合語言
5. **Fix Atomic Add**：修復 atomic add 語法
6. **Rename Symbol**：將符號重新命名
7. **Normalize directive comments**：統一 `@PRINT` 註解格式
8. **gpr_printf_tool**：注入 GPU printf 程式碼，產生最終 HSACO

### 3. @PRINT 在生成的 .s 中的對應位置

在 `kernel_INT32/kernel_fixed_print.s` (第 569-572 行) 中，`@PRINT` 被保留為註解，對應到具體的機器碼指令旁：

```
s_addk_i32    s80, 0x0100                             // 000000000D84: B7500100
; @PRINT max=16 if $tgid_x == 0 && $tgid_y == 0 && $tgid_z == 0 && $tid == 0: f"s80={s80:d}"
s_cmp_lt_i32  s80, s81                                // 000000000D88: BF045150
s_cbranch_scc0  label_049B                            // 000000000D8C: BF840137
```

可以看到 SP3 源碼中的 `s_loop_idx` 被解析為實際的 sgpr `s80`，`gemm0_SUB_K` 被解析為立即值 `0x0100`（256）。

經過 `gpr_printf_tool` 處理後，在最終的 `kernel_INT32/gpr_printf_out/kernel_fixed_print_debug_injected.s` 中，`@PRINT` 位置被注入了實際的 printf 呼叫程式碼（使用 `__ockl_printf_begin` / `__ockl_printf_append_args` 等 ROCm runtime 函式），以及用 `v_cndmask_b32` 實現的 max=16 計數邏輯。

### 4. 執行結果

```bash
python3 run_custom_kernel_moe.py --kernel kernel_INT32/gpr_printf_out/kernel_fixed_print_debug_injected.hsaco
```

執行後，GPU printf 會輸出迴圈中 `s80` 的值（每次遞增 256 = 0x100），最多 16 次：

```
s80=256
s80=768
s80=1280
s80=1792
s80=2304
s80=2816
s80=3328
s80=3840
Output (token 0, first 8 values) per run:
  Run  1: [-2032.0, -2416.0, -2176.0, 11392.0, -2976.0, 556.0, -6400.0, -4080.0]
```

可以觀察到 `s80`（`s_loop_idx`）的迭代過程：256, 768, 1280, ... 每次增加 512（兩次 `s_addk_i32 s80, 0x0100`），直到 kernel 計算完成並輸出最終結果。

---

## 範例二：INT8 Kernel

以 `FMOE_INT8_G1U1_1TG_4W_32mx1_128nx4.sp3` 為例，展示在迴圈展開的 memory load 指令旁加入多個 `@PRINT`。

### 1. SP3 源碼中的 @PRINT 標註

在 `FMOE_INT8_G1U1_1TG_4W_32mx1_128nx4.sp3` (第 378-381 行) 的 `buffer_load_dword` 迴圈中：

```
for var i=0; i < inst_cnt; i++
    var i_idx = s+i
    buffer_load_dword  v0,  v_regs(_v_X_addr, i_idx),  s_regs(_s_X_buf, 0), 0 lds:1 offen:1 
    // @PRINT if $tgid_x == 0 && $tgid_y == 0 && $tgid_z == 0 && $tid == 129: f"X_mem_load v_regs={v_regs:d} tid={$tid:d}"
```

**說明：**
- 這是一個 SP3 巨集迴圈，`v_regs` 是巨集參數，在展開時會被替換為具體的暫存器
- `$tid == 129`：指定印出特定 thread 的值
- 每次迴圈迭代的 `@PRINT` 會在展開後對應到不同的 vgpr

### 2. @PRINT 在生成的 .s 中的對應位置

在 `kernel_INT8/kernel_fixed_print.s` (第 285-307 行) 中，SP3 巨集迴圈被展開，每個 `buffer_load_dword` 後都有對應的 `@PRINT`，`v_regs` 被替換為實際暫存器：

```
buffer_load_dword   v26, s[20:23], 0 offen lds     // 000000000644: ...
; @PRINT if ... && $tid == 129: f"X_mem_load v26={v26:d} tid={$tid:d}"
s_add_u32     m0, 0x00000100, s50
buffer_load_dword   v27, s[20:23], 0 offen lds     // 000000000654: ...
; @PRINT if ... && $tid == 129: f"X_mem_load v27={v27:d} tid={$tid:d}"
s_add_u32     m0, 0x00000200, s50
buffer_load_dword   v28, s[20:23], 0 offen lds     // 000000000664: ...
; @PRINT if ... && $tid == 129: f"X_mem_load v28={v28:d} tid={$tid:d}"
...
buffer_load_dword   v33, s[20:23], 0 offen lds     // 0000000006B4: ...
; @PRINT if ... && $tid == 129: f"X_mem_load v33={v33:d} tid={$tid:d}"
```

可以看到 SP3 巨集中的單一 `@PRINT` 被展開為 8 個獨立的 `@PRINT`（v26 ~ v33），每個都對應到展開後的具體 vgpr。

### 3. 執行

先生成含有 printf 注入的 HSACO：

```bash
python3 sp3_printf_wrap.py FMOE_INT8_G1U1_1TG_4W_32mx1_128nx4.sp3 --output-dir kernel_INT8 --insert-mode all
```

使用自訂 host program 執行（需先透過 `sp3_host_gen.py` 產生 host code，見下方補充）：

```bash
./FMOE_INT8_test \
  --hsaco kernel_INT8/gpr_printf_out/kernel_fixed_print_debug_injected.hsaco \
  --kernel _ZN5aiter45fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_32x192E \
  --dim 512 --hidden-dim 512 --batch 64 --topk 8 --eprt 4
```

執行後會印出各 thread 的 memory load 暫存器值，例如：

```
X_mem_load v26=<value> tid=129
X_mem_load v27=<value> tid=129
...
```

---

## 補充：為個別 SP3 生成 Host Code

對於非 INT32 的 kernel（例如 INT8），由於沒有現成的 Python host runner，可以使用 `sp3_host_gen.py` 自動從 SP3 檔案生成 HIP C++ host program：

### 1. 生成 Host Code

```bash
python3 sp3_host_gen.py ../../poc_kl/mi300/fused_moe_asm/shaders/FMOE_INT8_G1U1_1TG_4W_32mx1_128nx4.sp3 -o FMOE_INT8_test.cpp
```

`sp3_host_gen.py` 會解析 SP3 檔頭的參數表與 grid 維度公式，自動生成一個 `.cpp` 檔案，包含：
- 透過 `hipModuleLoad` 載入 HSACO
- 為每個 pointer 參數分配 GPU buffer
- 按照 SP3 定義的 offset 組裝 kernarg buffer
- 以正確的 grid/block 維度啟動 kernel
- GPU printf 輸出會自動出現在 stderr

### 2. 編譯

```bash
hipcc --offload-arch=gfx942 -O0 -g FMOE_INT8_test.cpp -o FMOE_INT8_test
```

### 3. 執行

編譯完成後，即可使用 `--hsaco` 參數載入帶有 printf 注入的 kernel：

```bash
./FMOE_INT8_test \
  --hsaco kernel_INT8/gpr_printf_out/kernel_fixed_print_debug_injected.hsaco \
  --kernel _ZN5aiter45fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_32x192E \
  --dim 512 --hidden-dim 512 --batch 64 --topk 8 --eprt 4
```

---

## @PRINT 語法參考

```
// @PRINT [max=N] [if <condition>]: f"<format_string>"
```

| 欄位 | 說明 |
|------|------|
| `max=N` | 可選。限制最多印出 N 次（適用於迴圈內） |
| `if <condition>` | 可選。條件式，支援 `$tgid_x`, `$tgid_y`, `$tgid_z`, `$tid` 等內建變數 |
| `f"..."` | Python f-string 格式。支援 `{reg:d}`（十進位）等格式化語法 |
| `s80`, `v26` 等 | 直接引用 sgpr/vgpr 暫存器名稱 |
| `v_regs(...)`, `s_regs(...)` | SP3 巨集呼叫，展開時會替換為具體暫存器 |

### 完整流程圖

```
SP3 源碼 (含 @PRINT)
        │
        ▼
  sp3_printf_wrap.py
        │
        ├── Step 1: SP3 -> Binary
        ├── Step 2: Binary -> Disasm SP3
        ├── Step 3: 注入 @PRINT 到 Disasm
        ├── Step 4: SP3 -> LLVM .s
        ├── Step 5: Fix Atomic Add
        ├── Step 6: Symbol Rename
        ├── Step 7: Normalize Comments
        └── Step 8: gpr_printf_tool -> HSACO
                                         │
                                         ▼
                              GPU 執行 & printf 輸出
```
