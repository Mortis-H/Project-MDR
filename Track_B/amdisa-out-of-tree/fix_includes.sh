#!/bin/bash
# Script to fix include paths for out-of-tree build

DEST_DIR="/home/morhuang/Project-MDR/Track_B/amdisa-out-of-tree"

echo "=========================================="
echo "Fixing include paths for out-of-tree build"
echo "=========================================="

# Find all .h, .cpp, and .td files and update include paths
find "$DEST_DIR" -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.td" \) | while read file; do
    if grep -q "mlir/Dialect/AMDISA/" "$file"; then
        echo "Fixing: $file"
        sed -i 's|mlir/Dialect/AMDISA/|AMDISA/|g' "$file"
    fi
done

echo "=========================================="
echo "Include paths fixed!"
echo "=========================================="

