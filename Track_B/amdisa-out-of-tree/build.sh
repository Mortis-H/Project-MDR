#!/bin/bash
# Build script for AMDISA out-of-tree project

# Configuration
PROJECT_DIR="/home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree"
BUILD_DIR="$PROJECT_DIR/build"

# LLVM/MLIR paths - UPDATE THESE TO MATCH YOUR INSTALLATION
LLVM_BUILD_DIR="/home/morhuang/llvm-project/build"
MLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"
LLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm"
LLVM_EXTERNAL_LIT="$LLVM_BUILD_DIR/bin/llvm-lit"

echo "=========================================="
echo "Building AMDISA Out-of-Tree Project"
echo "=========================================="
echo "Project Dir: $PROJECT_DIR"
echo "Build Dir: $BUILD_DIR"
echo "MLIR Dir: $MLIR_DIR"
echo "LLVM Dir: $LLVM_DIR"
echo "=========================================="

# Check if MLIR_DIR exists
if [ ! -d "$MLIR_DIR" ]; then
    echo "ERROR: MLIR_DIR not found at: $MLIR_DIR"
    echo ""
    echo "Please update the MLIR_DIR variable in this script to point to your"
    echo "LLVM/MLIR installation. It should contain MLIRConfig.cmake."
    echo ""
    echo "You can find it with:"
    echo "  find /path/to/llvm-project/build -name MLIRConfig.cmake"
    exit 1
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo ""
echo "Configuring..."
cmake -G Ninja .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DLLVM_EXTERNAL_LIT="$LLVM_EXTERNAL_LIT" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed!"
    exit 1
fi

# Build
echo ""
echo "Building..."
cmake --build . -j$(nproc)

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Executable location: $BUILD_DIR/bin/amdisa-translate"
echo ""
echo "To run:"
echo "  $BUILD_DIR/bin/amdisa-translate --help"
echo ""

