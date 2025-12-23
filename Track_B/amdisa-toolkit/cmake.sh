#!/bin/bash

# 設定 LLVM 安裝路徑
LLVM_INSTALL_DIR=/tmp/llvm-install

cmake -B build -G Ninja \
  -DMLIR_DIR=$LLVM_INSTALL_DIR/lib/cmake/mlir \
  -DLLVM_DIR=$LLVM_INSTALL_DIR/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release