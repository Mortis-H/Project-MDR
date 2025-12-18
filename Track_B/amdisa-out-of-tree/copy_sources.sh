#!/bin/bash
# Script to copy AMDISA sources from in-tree to out-of-tree project

# Set the source directory (in-tree llvm-project)
LLVM_SRC="/home/morhuang/Project-MDR/Track_B/llvm-project"
MLIR_SRC="$LLVM_SRC/mlir"

# Set the destination directory (out-of-tree project)
DEST_DIR="/home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree"

echo "=========================================="
echo "Copying AMDISA sources to out-of-tree project"
echo "=========================================="

# ============================================================================
# 1. Copy Dialect Headers
# ============================================================================
echo "Copying Dialect headers..."
mkdir -p "$DEST_DIR/include/AMDISA/IR"

# Copy TableGen files
cp "$MLIR_SRC/include/mlir/Dialect/AMDISA/IR/AMDISAOps.td" \
   "$DEST_DIR/include/AMDISA/IR/AMDISAOps.td"

# Copy C++ headers
cp "$MLIR_SRC/include/mlir/Dialect/AMDISA/IR/AMDISAOps.h" \
   "$DEST_DIR/include/AMDISA/IR/AMDISAOps.h"

# Copy Pass headers
cp "$MLIR_SRC/include/mlir/Dialect/AMDISA/Passes.h" \
   "$DEST_DIR/include/AMDISA/Passes.h"

cp "$MLIR_SRC/include/mlir/Dialect/AMDISA/Passes.td" \
   "$DEST_DIR/include/AMDISA/Passes.td"

# ============================================================================
# 2. Copy Dialect Implementation
# ============================================================================
echo "Copying Dialect implementation..."
mkdir -p "$DEST_DIR/lib/AMDISA/IR"

cp "$MLIR_SRC/lib/Dialect/AMDISA/IR/AMDISAOps.cpp" \
   "$DEST_DIR/lib/AMDISA/IR/AMDISAOps.cpp"

# ============================================================================
# 3. Copy Transforms/Passes
# ============================================================================
echo "Copying Transforms..."
mkdir -p "$DEST_DIR/lib/AMDISA/Transforms"

cp "$MLIR_SRC/lib/Dialect/AMDISA/Transforms/LowerToGPUInlineAsm.cpp" \
   "$DEST_DIR/lib/AMDISA/Transforms/LowerToGPUInlineAsm.cpp"

# ============================================================================
# 4. Copy amdisa-translate tool
# ============================================================================
echo "Copying amdisa-translate tool..."
mkdir -p "$DEST_DIR/tools/amdisa-translate"

cp "$MLIR_SRC/tools/amdisa-translate/amdisa-translate.cpp" \
   "$DEST_DIR/tools/amdisa-translate/"

cp "$MLIR_SRC/tools/amdisa-translate/AMDISAAsmParser.cpp" \
   "$DEST_DIR/tools/amdisa-translate/"

cp "$MLIR_SRC/tools/amdisa-translate/AMDISAAsmParser.h" \
   "$DEST_DIR/tools/amdisa-translate/"

cp "$MLIR_SRC/tools/amdisa-translate/AMDGCNAssembly.cpp" \
   "$DEST_DIR/tools/amdisa-translate/"

cp "$MLIR_SRC/tools/amdisa-translate/AMDGCNAssembly.h" \
   "$DEST_DIR/tools/amdisa-translate/"

cp "$MLIR_SRC/tools/amdisa-translate/AMDGPUMetadata.cpp" \
   "$DEST_DIR/tools/amdisa-translate/"

cp "$MLIR_SRC/tools/amdisa-translate/AMDGPUMetadata.h" \
   "$DEST_DIR/tools/amdisa-translate/"

cp "$MLIR_SRC/tools/amdisa-translate/parse_utils.cpp" \
   "$DEST_DIR/tools/amdisa-translate/"

cp "$MLIR_SRC/tools/amdisa-translate/parse_utils.h" \
   "$DEST_DIR/tools/amdisa-translate/"

cp "$MLIR_SRC/tools/amdisa-translate/ParsedProgram.h" \
   "$DEST_DIR/tools/amdisa-translate/"

# Copy source directory if it exists
if [ -d "$MLIR_SRC/tools/amdisa-translate/source" ]; then
    echo "Copying source directory..."
    cp -r "$MLIR_SRC/tools/amdisa-translate/source" \
       "$DEST_DIR/tools/amdisa-translate/"
fi

echo "=========================================="
echo "Copy completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update include paths in copied files:"
echo "   - Change 'mlir/Dialect/AMDISA/...' to 'AMDISA/...'"
echo ""
echo "2. Build the out-of-tree project:"
echo "   cd $DEST_DIR"
echo "   mkdir build && cd build"
echo "   cmake -G Ninja .. -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir"
echo "   cmake --build ."
echo ""

