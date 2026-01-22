HIPCC ?= hipcc
ARCH ?= gfx950
OPT ?= -O3

KERNEL_SRC = batched_gemm.hip
HOST_SRC = batched_host.cpp
MODULE_HOST_SRC = batched_host_module.cpp
TARGET = batched_gemm.out
MODULE_TARGET = batched_gemm_module.out

.PHONY: all clean asm module hsaco

all: $(TARGET)

$(TARGET): $(KERNEL_SRC) $(HOST_SRC)
	$(HIPCC) --offload-arch=$(ARCH) $(OPT) $(KERNEL_SRC) $(HOST_SRC) -o $@

module: $(MODULE_TARGET)

$(MODULE_TARGET): $(MODULE_HOST_SRC)
	$(HIPCC) $(OPT) $(MODULE_HOST_SRC) -o $@

asm: $(KERNEL_SRC)
	$(HIPCC) --offload-arch=$(ARCH) $(OPT) --save-temps -c $(KERNEL_SRC) -o batched_gemm.o

hsaco:
	llvm-mc -triple amdgcn-amd-amdhsa -mcpu=$(ARCH) -filetype=obj \
		batched_gemm-hip-amdgcn-amd-amdhsa-$(ARCH).s \
		-o batched_gemm-hip-amdgcn-amd-amdhsa-$(ARCH).o
	lld -flavor gnu -m elf64_amdgpu --no-undefined -shared \
		-plugin-opt=-amdgpu-internalize-symbols -plugin-opt=mcpu=$(ARCH) \
		--whole-archive batched_gemm-hip-amdgcn-amd-amdhsa-$(ARCH).o \
		--no-whole-archive \
		-o batched_gemm-hip-amdgcn-amd-amdhsa-$(ARCH).hsaco

clean:
	rm -f $(TARGET) $(MODULE_TARGET) batched_gemm.o *.o *.bc *.ll *.hipi *.hsaco *.d *.obj *.out *.txt *.hipfb
	rm -f $(filter-out cap.s,$(wildcard *.s))
