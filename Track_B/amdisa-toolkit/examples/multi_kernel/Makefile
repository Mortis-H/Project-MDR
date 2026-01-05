# Makefile for HIP Dynamic Loading Example (Simplified)

# 編譯器設定
HIPCC = hipcc
ARCH = gfx950

# 目標
.PHONY: all clean kernel host test

all: kernel host

# 編譯 kernel 到 HSACO
kernel: vector_add_kernel.hsaco

vector_add_kernel.hsaco: vector_add_kernel.hip
	@echo "=== Compiling kernel to HSACO ==="
	$(HIPCC) --genco --offload-arch=$(ARCH) --save-temps $< -o $@
	@echo "✓ Generated $@"

# 編譯 host 程式
host: host_program

host_program: host_program.cpp
	@echo "=== Compiling host program ==="
	$(HIPCC) --save-temps $< -o $@
	@echo "✓ Generated $@"

# 測試執行
test: all
	@echo ""
	@echo "=== Running test ==="
	./host_program vector_add_kernel.hsaco vectorAdd 1048576

# 清理
clean:
	rm -f host_program
	rm -f *.hsaco *.out *.o *.s *.bc *.ll *.hipi *.hipfb *.resolution.txt
	rm -rf pipeline_output

# 顯示幫助
help:
	@echo "Available targets:"
	@echo "  all     - Build kernel and host program"
	@echo "  kernel  - Compile kernel to HSACO"
	@echo "  host    - Compile host program"
	@echo "  test    - Build and run test"
	@echo "  clean   - Remove all generated files"
	@echo ""
	@echo "Quick start:"
	@echo "  make test    # Build and test"
