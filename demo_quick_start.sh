#!/bin/bash
# demo_quick_start.sh
# Quick start script for Advanced Vietnamese Spell Corrector

echo "=========================================="
echo "Advanced Vietnamese Spell Corrector"
echo "Quick Start Demo"
echo "=========================================="
echo ""

# Step 1: Create sample lexicon
echo "[Step 1/3] Creating sample lexicon..."
python prepare_data.py --create_sample

if [ $? -ne 0 ]; then
    echo "Error: Failed to create lexicon"
    exit 1
fi

echo "✓ Lexicon created at: data/vi_lexicon.txt"
echo ""

# Step 2: Check if detector model exists
echo "[Step 2/3] Checking detector model..."
if [ ! -d "outputs/detector" ]; then
    echo "⚠ Detector model not found at: outputs/detector"
    echo ""
    echo "To train detector, run:"
    echo "  python vi_spell_pipeline_plus.py --do_train_detector"
    echo ""
    echo "Continuing with limited functionality..."
else
    echo "✓ Detector model found"
fi
echo ""

# Step 3: Run tests
echo "[Step 3/3] Running tests..."
python test_advanced_corrector.py

echo ""
echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start API server:"
echo "   uvicorn app:app --host 0.0.0.0 --port 8000"
echo ""
echo "2. Test API:"
echo "   curl -X POST http://localhost:8000/correct_v2 \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"text\": \"Tôii đangg họcc tiếng Việt\"}'"
echo ""
echo "3. Read documentation:"
echo "   cat ADVANCED_CORRECTOR_README.md"
echo ""

