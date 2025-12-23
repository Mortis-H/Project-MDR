# Track_A 快速入門指南（給 LLM 的摘要）

## 🎯 一句話總結
在 AMD GPU 的組合語言（ISA）中插入高階除錯邏輯（GPU Dialect），且不破壞原始計算。

---

## 🔑 核心技術：Register Clobbing（必讀！）

```mlir
// 步驟 1：保護原始邏輯使用的暫存器（v0-v31）
%reserved = llvm.inline_asm has_side_effects asm_dialect = att "", 
            "={v[0:31]}": () -> vector<32xi32>

// 步驟 2：從原始暫存器讀取值
%val_A = llvm.inline_asm has_side_effects asm_dialect = att 
         "v_mov_b32 $0, v6", "=v": () -> f32

// 步驟 3：插入 DSL 代碼（編譯器會使用 v32+ 暫存器）
%tid = gpu.thread_id x
gpu.printf "A[%d] = %4.3f\n", %tid, %val_A : index, f32

// 步驟 4：釋放保護
llvm.inline_asm has_side_effects asm_dialect = att "", 
                "{v[0:31]}" %reserved : (vector<32xi32>)-> ()
```

**原理**：告訴 LLVM「v0-v31 已被佔用」→ DSL 代碼使用 v32+ → 不會衝突

---

## 📁 關鍵檔案

| 檔案 | 用途 |
|------|------|
| `kernel_template.mlir` | MLIR 範本（展示 DSL 語法） |
| `mdr.py` | 編譯腳本（MLIR → ISA → HSACO） |
| `e2e_test/vec_add_kernel_after.s` | 插入 DSL 後的 ISA（驗證用） |
| `TECHNICAL_REPORT.md` | 完整技術文檔 |

---

## ⚡ 快速驗證

```bash
cd /home/morhuang/Project-MDR/Track_A/e2e_test
./vec_add_module
```

**預期輸出**：會看到 `gpu.printf` 的除錯訊息 + 原始計算結果正確（Result: OK）

---

## 🧩 支援的功能

- ✅ `gpu.thread_id`、`gpu.printf`
- ✅ `cf.cond_br`（條件分支）
- ✅ `arith.cmpi`、`arith.mulf`、`arith.subf`（算術運算）
- ✅ 混合 `llvm.inline_asm` 和高階 Dialect

---

## ⚠️ 重要限制

1. **目前是 PoC**：`mdr.py` 輸出硬編碼的 MLIR，不是真正的「ISA → Raising」
2. **需手動分析**：哪些暫存器被使用、在哪裡插入安全
3. **未來目標**：自動化 ISA 分析和 DSL 插入

---

## 🎓 給 LLM 的建議

- 需要詳細技術？→ 讀 `TECHNICAL_REPORT.md`
- 需要看 MLIR 語法？→ 讀 `kernel_template.mlir`
- 需要理解編譯流程？→ 讀 `mdr.py`
- 需要驗證結果？→ 運行 `./vec_add_module`

