#!/bin/bash
# Compile HIP kernel to assembly with AGPR usage

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "Compiling HIP kernel with AGPR usage"
echo "═══════════════════════════════════════════════════════════════"

# Target architecture (gfx950 for MI300)
ARCH="gfx950"

# Compile to assembly
echo "[1/3] Compiling to assembly..."
hipcc mfma_simple.hip \
    --offload-arch=${ARCH} \
    -S -o mfma_simple_full.s \
    -O2 \
    --save-temps

echo "[2/3] Extracting kernel assembly..."
# Extract just the kernel assembly (between .text and .section)
awk '/\.text/,/^\.section/ {print}' mfma_simple_full.s > mfma_simple_kernel_tmp.s

# Or try to extract more cleanly
if [ -f "mfma_simple-hip-amdgcn-amd-amdhsa-${ARCH}.s" ]; then
    echo "[INFO] Found device-specific assembly"
    cp "mfma_simple-hip-amdgcn-amd-amdhsa-${ARCH}.s" original.s
else
    echo "[INFO] Using extracted assembly"
    # Clean up and extract kernel portion
    grep -A 99999 "\.text" mfma_simple_full.s | grep -B 99999 "\.section" | head -n -1 > original.s
fi

echo "[3/3] Verifying AGPR usage..."
if grep -q "\.agpr_count:" original.s; then
    AGPR_COUNT=$(grep "\.agpr_count:" original.s | awk '{print $2}')
    echo "✅ Found AGPR usage: agpr_count = ${AGPR_COUNT}"
else
    echo "⚠️  Warning: No .agpr_count found in assembly"
fi

if grep -qE '\ba[0-9]+\b|\ba\[[0-9]+:[0-9]+\]' original.s; then
    echo "✅ Found AGPR register usage in instructions"
    echo ""
    echo "Sample AGPR instructions:"
    grep -E '\ba[0-9]+\b|\ba\[[0-9]+:[0-9]+\]' original.s | head -5
else
    echo "⚠️  Warning: No AGPR register references found"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Compilation complete!"
echo "Output: original.s"
echo "═══════════════════════════════════════════════════════════════"

# Show metadata
echo ""
echo "Kernel Metadata:"
grep -E "vgpr_count|sgpr_count|agpr_count|kernarg_segment_size" original.s || echo "No metadata found"

