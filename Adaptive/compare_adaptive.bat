@echo off
echo Starting Baseline vs Adaptive Model Comparison...
echo.

REM Set the parent directory in Python path
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Default parameters - UPDATE THESE PATHS TO YOUR ACTUAL CHECKPOINTS
set BASELINE_CHECKPOINT=./checkpoints/best_model.pth
set ADAPTIVE_CHECKPOINT=./checkpoints_adaptive/best_model.pth
set DATA_PATH=./Datasets
set RESULTS_DIR=./comparison_results
set BATCH_SIZE=1
set DIM=48
set NUM_SAMPLES=10

REM Adaptive parameters
set CONTRAST_METHOD=combined
set MIN_DENSITY=0.1
set MAX_DENSITY=1.0

echo Configuration:
echo   Baseline checkpoint: %BASELINE_CHECKPOINT%
echo   Adaptive checkpoint: %ADAPTIVE_CHECKPOINT%
echo   Data path: %DATA_PATH%
echo   Results directory: %RESULTS_DIR%
echo   Batch size: %BATCH_SIZE%
echo   Model dimension: %DIM%
echo   Sample images: %NUM_SAMPLES%
echo   Adaptive contrast method: %CONTRAST_METHOD%
echo   Sampling density: [%MIN_DENSITY%, %MAX_DENSITY%]
echo.

REM Check if checkpoints exist
if not exist "%BASELINE_CHECKPOINT%" (
    echo Error: Baseline checkpoint not found: %BASELINE_CHECKPOINT%
    echo Please update the BASELINE_CHECKPOINT path in this script.
    pause
    exit /b 1
)

if not exist "%ADAPTIVE_CHECKPOINT%" (
    echo Error: Adaptive checkpoint not found: %ADAPTIVE_CHECKPOINT%
    echo Please ensure the adaptive model has been trained.
    pause
    exit /b 1
)

REM Create results directory
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

echo Starting comparison...
python Adaptive/compare_adaptive.py ^
    --baseline_checkpoint "%BASELINE_CHECKPOINT%" ^
    --adaptive_checkpoint "%ADAPTIVE_CHECKPOINT%" ^
    --data_path "%DATA_PATH%" ^
    --results_dir "%RESULTS_DIR%" ^
    --batch_size %BATCH_SIZE% ^
    --num_samples %NUM_SAMPLES% ^
    --dim %DIM% ^
    --contrast_method %CONTRAST_METHOD% ^
    --min_density %MIN_DENSITY% ^
    --max_density %MAX_DENSITY%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Comparison completed successfully!
    echo Results saved to: %RESULTS_DIR%
    echo.
    echo Generated files:
    echo   - comparison_results.json: Detailed comparison metrics
    echo   - comparison_plots.png: Performance comparison charts
    echo   - improvement_distribution.png: Improvement distribution histograms
    echo   - sample_comparisons/: Side-by-side image comparisons
) else (
    echo.
    echo Comparison failed with error code %ERRORLEVEL%
    echo Please check the error messages above.
)

pause