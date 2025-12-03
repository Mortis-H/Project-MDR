./build/bin/llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx950 -filetype=asm --show-inst-operands ../bkwd.s -o output.s |& tee inst_operands.txt
