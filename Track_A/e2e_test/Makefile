HIPCC        ?= hipcc
OFFLOAD_ARCH ?= gfx950
CLANG        ?= clang
CLANG_OFFLOAD_BUNDLER ?= clang-offload-bundler
LLD          ?= ld.lld

TARGET      := vec_add_module
KERNEL_SRC  := vec_add_kernel.hip
KERNEL_S    := vec_add_kernel-$(OFFLOAD_ARCH).s
KERNEL_O    := vec_add_kernel-$(OFFLOAD_ARCH).o
KERNEL_OUT  := vec_add_kernel-$(OFFLOAD_ARCH).out
KERNEL_CO   := vec_add_kernel.hsaco
HOST_SRC    := main.cpp

MDR_BUILD   := build_mdr
MDR_S       := $(MDR_BUILD)/kernel_isa.s

.PHONY: all clean

all: $(TARGET)

# (1) 把 device-only kernel 編成 .hsaco code object
$(KERNEL_CO): $(KERNEL_SRC)
	$(HIPCC) --genco --offload-arch=$(OFFLOAD_ARCH) --save-temps $< -o $@

# (2) 編 host 程式（不需要帶 kernel source）
$(TARGET): $(HOST_SRC) $(KERNEL_CO)
	$(HIPCC) $(HOST_SRC) -o $@

# (3) Assemble kernels
assemble: $(KERNEL_SRC) $(MDR_S)
	cp $(MDR_S) $(KERNEL_S)
	$(CLANG) -cc1as -triple amdgcn-amd-amdhsa -filetype obj -main-file-name $(KERNEL_SRC) -target-cpu $(OFFLOAD_ARCH) -fdebug-compilation-dir=$(PWD) -dwarf-version=5 -mrelocation-model pic -o $(KERNEL_O) $(KERNEL_S)
	$(LLD) -flavor gnu -m elf64_amdgpu --no-undefined -shared -plugin-opt=-amdgpu-internalize-symbols --lto-partitions=8 -plugin-opt=mcpu=gfx950 -plugin-opt=O3 --lto-CGO3 --whole-archive -o $(KERNEL_OUT) $(KERNEL_O) --no-whole-archive
	$(CLANG_OFFLOAD_BUNDLER) -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--$(OFFLOAD_ARCH) -input=/dev/null -input=$(KERNEL_OUT) -output=$(KERNEL_CO) -verbose

clean:
	rm -f $(TARGET) $(KERNEL_CO) $(KERNEL_S) $(KERNEL_O) $(KERNEL_OUT)
