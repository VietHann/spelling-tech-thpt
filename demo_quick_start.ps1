# demo_quick_start.ps1
# Quick start script for Advanced Vietnamese Spell Corrector (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Advanced Vietnamese Spell Corrector" -ForegroundColor Cyan
Write-Host "Quick Start Demo" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create sample lexicon
Write-Host "[Step 1/3] Creating sample lexicon..." -ForegroundColor Yellow
python prepare_data.py --create_sample

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to create lexicon" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Lexicon created at: data/vi_lexicon.txt" -ForegroundColor Green
Write-Host ""

# Step 2: Check if detector model exists
Write-Host "[Step 2/3] Checking detector model..." -ForegroundColor Yellow
if (-Not (Test-Path "outputs/detector")) {
    Write-Host "⚠ Detector model not found at: outputs/detector" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To train detector, run:"
    Write-Host "  python vi_spell_pipeline_plus.py --do_train_detector"
    Write-Host ""
    Write-Host "Continuing with limited functionality..." -ForegroundColor Yellow
} else {
    Write-Host "✓ Detector model found" -ForegroundColor Green
}
Write-Host ""

# Step 3: Run tests
Write-Host "[Step 3/3] Running tests..." -ForegroundColor Yellow
python test_advanced_corrector.py

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Quick Start Complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Start API server:"
Write-Host "   uvicorn app:app --host 0.0.0.0 --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "2. Test API (in another terminal):"
Write-Host "   curl -X POST http://localhost:8000/correct_v2 ``" -ForegroundColor White
Write-Host "     -H 'Content-Type: application/json' ``" -ForegroundColor White
Write-Host "     -d '{\"text\": \"Tôii đangg họcc tiếng Việt\"}'" -ForegroundColor White
Write-Host ""
Write-Host "3. Read documentation:"
Write-Host "   Get-Content ADVANCED_CORRECTOR_README.md" -ForegroundColor White
Write-Host ""

