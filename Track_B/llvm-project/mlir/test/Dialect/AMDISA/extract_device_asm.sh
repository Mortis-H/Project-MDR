#!/bin/bash

# 從 hipcc 生成的 .s 文件中提取 device assembly

INPUT_FILE="$1"
OUTPUT_FILE="$2"

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "用法: $0 <input.s> <output.s>"
    exit 1
fi

# 提取 hip-amdgcn-amd-amdhsa bundle
sed -n '/^# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa/,/^# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa/p' "$INPUT_FILE" | \
  grep -v "^# __CLANG_OFFLOAD_BUNDLE" > "$OUTPUT_FILE"

echo "已提取 device assembly: $OUTPUT_FILE ($(wc -l < "$OUTPUT_FILE") 行)"

