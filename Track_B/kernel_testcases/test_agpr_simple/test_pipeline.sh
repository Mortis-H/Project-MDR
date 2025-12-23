#!/bin/bash
# Test pipeline.py with AGPR-enabled kernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PIPELINE_DIR="../../llvm-project/my_test"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Testing Pipeline.py with AGPR Kernel                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if original.s exists
if [ ! -f "original.s" ]; then
    echo "❌ Error: original.s not found"
    echo "Please run ./compile.sh first"
    exit 1
fi

# Show original metadata
echo "═══════════════════════════════════════════════════════════════"
echo "Original Kernel Metadata"
echo "═══════════════════════════════════════════════════════════════"
ORIG_VGPR=$(grep "\.vgpr_count:" original.s | awk '{print $2}')
ORIG_SGPR=$(grep "\.sgpr_count:" original.s | awk '{print $2}')
ORIG_AGPR=$(grep "\.agpr_count:" original.s | awk '{print $2}')
ORIG_KERNARG=$(grep "kernarg_segment_size:" original.s | head -1 | awk '{print $2}')

echo "VGPR Count:    ${ORIG_VGPR}"
echo "SGPR Count:    ${ORIG_SGPR}"
echo "AGPR Count:    ${ORIG_AGPR}"
echo "Kernarg Size:  ${ORIG_KERNARG}"
echo ""

# Show AGPR usage in instructions
echo "═══════════════════════════════════════════════════════════════"
echo "AGPR Usage in Instructions"
echo "═══════════════════════════════════════════════════════════════"
grep -E '\ba[0-9]+\b|\ba\[[0-9]+:[0-9]+\]' original.s | head -10
echo ""

# Run pipeline.py
echo "═══════════════════════════════════════════════════════════════"
echo "Running Pipeline.py"
echo "═══════════════════════════════════════════════════════════════"
cd "$PIPELINE_DIR"
python3 pipeline.py "$SCRIPT_DIR/original.s" \
    --workdir "$SCRIPT_DIR/pipeline_output" \
    --output-prefix "agpr_test" 2>&1 | tee "$SCRIPT_DIR/pipeline.log"

cd "$SCRIPT_DIR"

# Check if pipeline succeeded
if [ ! -f "pipeline_output/agpr_test.s" ]; then
    echo ""
    echo "❌ Pipeline failed - no output assembly generated"
    exit 1
fi

# Compare metadata
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Comparing Metadata (Original vs Rebuilt)"
echo "═══════════════════════════════════════════════════════════════"

REBUILT_VGPR=$(grep "\.vgpr_count:" pipeline_output/agpr_test.s | awk '{print $2}')
REBUILT_SGPR=$(grep "\.sgpr_count:" pipeline_output/agpr_test.s | awk '{print $2}')
REBUILT_AGPR=$(grep "\.agpr_count:" pipeline_output/agpr_test.s | awk '{print $2}')
REBUILT_KERNARG=$(grep "kernarg_segment_size:" pipeline_output/agpr_test.s | head -1 | awk '{print $2}')

printf "%-20s %-10s %-10s %-10s\n" "Metric" "Original" "Rebuilt" "Match"
printf "%-20s %-10s %-10s %-10s\n" "--------------------" "----------" "----------" "----------"

# Check VGPR
VGPR_MATCH="❌"
if [ "$ORIG_VGPR" = "$REBUILT_VGPR" ]; then
    VGPR_MATCH="✅"
fi
printf "%-20s %-10s %-10s %-10s\n" "VGPR Count" "$ORIG_VGPR" "$REBUILT_VGPR" "$VGPR_MATCH"

# Check SGPR
SGPR_MATCH="❌"
if [ "$ORIG_SGPR" = "$REBUILT_SGPR" ]; then
    SGPR_MATCH="✅"
fi
printf "%-20s %-10s %-10s %-10s\n" "SGPR Count" "$ORIG_SGPR" "$REBUILT_SGPR" "$SGPR_MATCH"

# Check AGPR
AGPR_MATCH="❌"
if [ "$ORIG_AGPR" = "$REBUILT_AGPR" ]; then
    AGPR_MATCH="✅"
fi
printf "%-20s %-10s %-10s %-10s\n" "AGPR Count" "$ORIG_AGPR" "$REBUILT_AGPR" "$AGPR_MATCH"

# Check Kernarg
KERNARG_MATCH="❌"
if [ "$ORIG_KERNARG" = "$REBUILT_KERNARG" ]; then
    KERNARG_MATCH="✅"
fi
printf "%-20s %-10s %-10s %-10s\n" "Kernarg Size" "$ORIG_KERNARG" "$REBUILT_KERNARG" "$KERNARG_MATCH"

echo ""

# Final result
ALL_MATCH=true
if [ "$VGPR_MATCH" = "❌" ] || [ "$SGPR_MATCH" = "❌" ] || [ "$AGPR_MATCH" = "❌" ] || [ "$KERNARG_MATCH" = "❌" ]; then
    ALL_MATCH=false
fi

echo "═══════════════════════════════════════════════════════════════"
echo "Result Summary"
echo "═══════════════════════════════════════════════════════════════"

if [ "$ALL_MATCH" = true ]; then
    echo "✅ All metadata matches!"
    echo ""
    echo "Key observations:"
    echo "  • Pipeline successfully processed AGPR kernel"
    if [ "$AGPR_MATCH" = "✅" ]; then
        echo "  • AGPR count preserved correctly (${ORIG_AGPR})"
    fi
    echo "  • VGPR/SGPR clobber mechanism working"
    echo "  • Ready for DSL insertion"
else
    echo "⚠️  Some metadata mismatches detected"
    echo ""
    if [ "$AGPR_MATCH" = "❌" ]; then
        echo "  ⚠️  AGPR count mismatch (expected, see AGPR_LIMITATION.md)"
    fi
    if [ "$VGPR_MATCH" = "❌" ]; then
        echo "  ❌ VGPR count mismatch (unexpected!)"
    fi
    if [ "$SGPR_MATCH" = "❌" ]; then
        echo "  ❌ SGPR count mismatch (unexpected!)"
    fi
fi

echo ""
echo "📁 Output files:"
echo "  • pipeline_output/agpr_test.gpumlir  - GPU MLIR with clobber"
echo "  • pipeline_output/agpr_test.s        - Rebuilt assembly"
echo "  • pipeline_output/agpr_test.hsaco    - Executable binary"
echo "  • pipeline.log                       - Pipeline execution log"

if [ "$ALL_MATCH" = true ]; then
    exit 0
else
    exit 1
fi

