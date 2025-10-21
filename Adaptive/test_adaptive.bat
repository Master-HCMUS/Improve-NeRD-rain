@echo off
echo Starting Adaptive NeRD-Rain Testing...
echo.

REM Set the parent directory in Python path
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Default parameters
set CHECKPOINT=./checkpoints_adaptive/best_model.pth
set DATA_PATH=./Datasets
set RESULTS_DIR=./test_results_adaptive
set BATCH_SIZE=1
set DIM=48

REM Adaptive parameters
set USE_ADAPTIVE_INR=--use_adaptive_inr
set CONTRAST_METHOD=combined
set MIN_DENSITY=0.1
set MAX_DENSITY=1.0

REM Testing options
set SAVE_RESULTS=--save_results
set ANALYZE_SAMPLING=--analyze_sampling

echo Configuration:
echo   Checkpoint: %CHECKPOINT%
echo   Data path: %DATA_PATH%
echo   Results directory: %RESULTS_DIR%
echo   Batch size: %BATCH_SIZE%
echo   Model dimension: %DIM%
echo   Adaptive INR: Enabled
echo   Contrast method: %CONTRAST_METHOD%
echo   Sampling density: [%MIN_DENSITY%, %MAX_DENSITY%]
echo   Save results: Yes
echo   Analyze sampling: Yes
echo.

REM Check if checkpoint exists
if not exist "%CHECKPOINT%" (
    echo Error: Checkpoint file not found: %CHECKPOINT%
    echo Please ensure the model has been trained and the checkpoint exists.
    pause
    exit /b 1
)

REM Create results directory
if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

echo Starting testing...
python Adaptive/test_adaptive.py ^
    --checkpoint "%CHECKPOINT%" ^
    --data_path "%DATA_PATH%" ^
    --results_dir "%RESULTS_DIR%" ^
    --batch_size %BATCH_SIZE% ^
    --dim %DIM% ^
    %USE_ADAPTIVE_INR% ^
    --contrast_method %CONTRAST_METHOD% ^
    --min_density %MIN_DENSITY% ^
    --max_density %MAX_DENSITY% ^
    %SAVE_RESULTS% ^
    %ANALYZE_SAMPLING%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Testing completed successfully!
    echo Results saved to: %RESULTS_DIR%
    echo.
    echo Generated files:
    echo   - test_results.json: Detailed metrics
    echo   - sampling_statistics.json: Adaptive sampling analysis
    echo   - Individual result images and comparisons
    echo   - Sampling analysis visualizations
) else (
    echo.
    echo Testing failed with error code %ERRORLEVEL%
    echo Please check the error messages above.
)

pause