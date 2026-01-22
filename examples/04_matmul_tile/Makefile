HIPCC ?= hipcc
ARCH ?= gfx950
OPT ?= -O3

KERNEL_SRC = matmul_tile.hip
HOST_SRC = matmul_host.cpp
TARGET = matmul_tile.out

.PHONY: all clean asm

all: $(TARGET)

$(TARGET): $(KERNEL_SRC) $(HOST_SRC)
	$(HIPCC) --offload-arch=$(ARCH) $(OPT) $(KERNEL_SRC) $(HOST_SRC) -o $@

asm: $(KERNEL_SRC)
	$(HIPCC) --offload-arch=$(ARCH) $(OPT) --save-temps -c $(KERNEL_SRC) -o matmul_tile.o

clean:
	rm -f $(TARGET) matmul_tile.o matmul_tile.s *.o *.s *.bc *.ll *.hipi *.hsaco *.d *.obj *.out *.txt *.hipfb
