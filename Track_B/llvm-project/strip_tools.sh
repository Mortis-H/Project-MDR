#!/bin/bash
# Strip LLVM/MLIR tools to reduce size
# This script removes debug symbols from compiled binaries

set -e

BUILD_DIR="/home/morhuang/Project-MDR/Track_B/llvm-project/build/bin"
BACKUP_DIR="/home/morhuang/Project-MDR/Track_B/llvm-project/build/bin_debug_backup"

echo "=========================================="
echo "LLVM/MLIR Tools Size Optimization"
echo "=========================================="
echo ""

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ Error: Build directory not found: $BUILD_DIR"
    exit 1
fi

# Create backup directory
echo "📦 Creating backup directory..."
mkdir -p "$BACKUP_DIR"

# Tools to strip
TOOLS=(amdisa-translate mlir-opt llvm-mc)

# Backup and show original sizes
echo ""
echo "📊 Original sizes:"
for tool in "${TOOLS[@]}"; do
    if [ -f "$BUILD_DIR/$tool" ]; then
        SIZE=$(du -h "$BUILD_DIR/$tool" | cut -f1)
        echo "  - $tool: $SIZE"
        cp "$BUILD_DIR/$tool" "$BACKUP_DIR/"
    else
        echo "  - $tool: ⚠️  Not found"
    fi
done

# Strip debug symbols
echo ""
echo "✂️  Stripping debug symbols..."
for tool in "${TOOLS[@]}"; do
    if [ -f "$BUILD_DIR/$tool" ]; then
        echo "  Processing $tool..."
        strip --strip-debug "$BUILD_DIR/$tool"
    fi
done

# Show new sizes
echo ""
echo "📊 Optimized sizes:"
TOTAL_SAVED=0
for tool in "${TOOLS[@]}"; do
    if [ -f "$BUILD_DIR/$tool" ]; then
        SIZE=$(du -h "$BUILD_DIR/$tool" | cut -f1)
        echo "  - $tool: $SIZE"
        
        # Calculate saved space (rough estimate)
        if [ -f "$BACKUP_DIR/$tool" ]; then
            BEFORE=$(stat -c%s "$BACKUP_DIR/$tool")
            AFTER=$(stat -c%s "$BUILD_DIR/$tool")
            SAVED=$((BEFORE - AFTER))
            TOTAL_SAVED=$((TOTAL_SAVED + SAVED))
        fi
    fi
done

# Show total saved space
if [ $TOTAL_SAVED -gt 0 ]; then
    SAVED_MB=$((TOTAL_SAVED / 1024 / 1024))
    SAVED_GB=$((SAVED_MB / 1024))
    echo ""
    echo "💾 Total space saved: ${SAVED_MB} MB (${SAVED_GB} GB)"
fi

echo ""
echo "=========================================="
echo "✅ Optimization complete!"
echo "=========================================="
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""
echo "To restore original versions:"
echo "  cp $BACKUP_DIR/* $BUILD_DIR/"
echo ""

