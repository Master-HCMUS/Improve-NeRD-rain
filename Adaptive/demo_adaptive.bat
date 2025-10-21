@echo off
echo Starting Adaptive NeRD-Rain Demo...
echo.

REM Set the parent directory in Python path
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Default parameters - UPDATE THESE PATHS
set CHECKPOINT=./checkpoints_adaptive/best_model.pth
set INPUT_IMAGE=./demo_input.jpg
set OUTPUT_DIR=./demo_results
set DIM=48

REM Adaptive parameters
set CONTRAST_METHOD=combined
set MIN_DENSITY=0.1
set MAX_DENSITY=1.0

REM Demo options
set ANALYZE_CONTRAST=--analyze_contrast
set ANALYZE_DENSITY=--analyze_density

echo Configuration:
echo   Checkpoint: %CHECKPOINT%
echo   Input image: %INPUT_IMAGE%
echo   Output directory: %OUTPUT_DIR%
echo   Model dimension: %DIM%
echo   Contrast method: %CONTRAST_METHOD%
echo   Sampling density: [%MIN_DENSITY%, %MAX_DENSITY%]
echo   Analyze contrast methods: Yes
echo   Analyze density configs: Yes
echo.

REM Check if checkpoint exists
if not exist "%CHECKPOINT%" (
    echo Error: Checkpoint file not found: %CHECKPOINT%
    echo Please ensure the adaptive model has been trained.
    pause
    exit /b 1
)

REM Check if input image exists
if not exist "%INPUT_IMAGE%" (
    echo Error: Input image not found: %INPUT_IMAGE%
    echo Please provide a valid input image path.
    echo.
    echo Usage examples:
    echo   demo_adaptive.bat "path\to\your\image.jpg"
    echo   demo_adaptive.bat "C:\Users\Username\Pictures\rainy_image.png"
    pause
    exit /b 1
)

REM Create output directory
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Starting demo...
python Adaptive/demo_adaptive.py ^
    --checkpoint "%CHECKPOINT%" ^
    --input_image "%INPUT_IMAGE%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --dim %DIM% ^
    --contrast_method %CONTRAST_METHOD% ^
    --min_density %MIN_DENSITY% ^
    --max_density %MAX_DENSITY% ^
    %ANALYZE_CONTRAST% ^
    %ANALYZE_DENSITY%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Demo completed successfully!
    echo Results saved to: %OUTPUT_DIR%
    echo.
    echo Generated files:
    echo   - input.png: Original input image
    echo   - output.png: Restored image
    echo   - adaptive_inference_analysis.png: Complete analysis
    echo   - contrast_analysis/: Contrast method comparisons
    echo   - density_analysis/: Sampling density comparisons
    echo.
    echo You can view the results by opening the files in %OUTPUT_DIR%
) else (
    echo.
    echo Demo failed with error code %ERRORLEVEL%
    echo Please check the error messages above.
)

pause