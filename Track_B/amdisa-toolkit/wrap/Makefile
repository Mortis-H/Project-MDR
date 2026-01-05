# Makefile for GPU Kernel + Host Integration
# 整合 wrap.s (GPU kernel) 和 matrix_add_host.cpp (host program)

# 配置變量
CHIP := gfx950
WORKDIR := build_output
WRAP_S := 0b66f7d63892c8d65fd11a9251b84e0488021ea1_wrapped.s
HOST_CPP := matrix_add_host.cpp
PIPELINE_PY := ../examples/pipeline.py

# 輸出文件
HSACO := vec_add_kernel.hsaco
HOST_BIN := matrix_add_host
# 自動從輸入 .s 檔案名稱生成輸出檔案名稱
WRAP_S_BASE := $(basename $(WRAP_S))
GENERATED_HSACO := $(WORKDIR)/$(WRAP_S_BASE)_rebuilt.hsaco

# 編譯器和工具
HIPCC := hipcc
PYTHON := python3

# 顏色輸出（可選）
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

.PHONY: all clean run help check-tools

# 預設目標：構建所有
all: check-tools $(HSACO) $(HOST_BIN)
	@echo "$(GREEN)✓ 構建完成！$(NC)"
	@echo "使用 'make run' 執行程序"

# 檢查必要工具
check-tools:
	@echo "$(YELLOW)檢查必要工具...$(NC)"
	@which hipcc > /dev/null || (echo "錯誤: 找不到 hipcc" && exit 1)
	@which amdisa-translate > /dev/null || (echo "錯誤: 找不到 amdisa-translate" && exit 1)
	@which mlir-opt > /dev/null || (echo "錯誤: 找不到 mlir-opt" && exit 1)
	@which llvm-mc > /dev/null || (echo "錯誤: 找不到 llvm-mc" && exit 1)
	@which ld.lld > /dev/null || (echo "錯誤: 找不到 ld.lld" && exit 1)
	@which python3 > /dev/null || (echo "錯誤: 找不到 python3" && exit 1)
	@echo "$(GREEN)✓ 所有工具都可用$(NC)"

# 步驟 1: 將 wrap.s 轉換為 HSACO
$(HSACO): $(WRAP_S) $(PIPELINE_PY)
	@echo "$(YELLOW)使用 pipeline.py 生成 HSACO...$(NC)"
	@mkdir -p $(WORKDIR)
	$(PYTHON) $(PIPELINE_PY) --chip $(CHIP) --workdir $(WORKDIR) $(WRAP_S)
	@cp $(GENERATED_HSACO) $(HSACO)
	@echo "$(GREEN)✓ 生成 HSACO: $(HSACO)$(NC)"

# 步驟 2: 編譯 host 程序
$(HOST_BIN): $(HOST_CPP)
	@echo "$(YELLOW)編譯 host 程序...$(NC)"
	$(HIPCC) -o $(HOST_BIN) $(HOST_CPP)
	@echo "$(GREEN)✓ 編譯完成: $(HOST_BIN)$(NC)"

# 執行程序
run: all
	@echo "$(YELLOW)執行程序...$(NC)"
	@echo "================================"
	@./$(HOST_BIN)
	@echo "================================"
	@echo "$(GREEN)✓ 執行完成$(NC)"

# 清理生成的文件
clean:
	@echo "$(YELLOW)清理生成的文件...$(NC)"
	rm -rf $(WORKDIR)
	rm -f $(HSACO) $(HOST_BIN)
	@echo "$(GREEN)✓ 清理完成$(NC)"

# 顯示幫助信息
help:
	@echo "GPU Kernel + Host 整合 Makefile"
	@echo ""
	@echo "可用目標："
	@echo "  make          - 構建 HSACO 和 host 程序（預設）"
	@echo "  make all      - 同上"
	@echo "  make run      - 構建並執行程序"
	@echo "  make clean    - 清理所有生成的文件"
	@echo "  make help     - 顯示此幫助信息"
	@echo ""
	@echo "配置變量："
	@echo "  CHIP=$(CHIP)        - GPU 架構"
	@echo "  WORKDIR=$(WORKDIR)  - 中間文件目錄"
	@echo ""
	@echo "示例："
	@echo "  make run              - 一鍵構建並執行"
	@echo "  make CHIP=gfx908      - 為不同 GPU 構建"
	@echo "  make clean && make    - 重新構建"



