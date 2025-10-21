@echo off
echo Starting Adaptive NeRD-Rain Training...
echo.

REM Set the parent directory in Python path
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Default parameters
set DATA_PATH=./Datasets
set SAVE_DIR=./checkpoints_adaptive
set LOG_DIR=./logs_adaptive
set BATCH_SIZE=8
set NUM_EPOCHS=500
set LEARNING_RATE=3e-4
set PATCH_SIZE=256
set DIM=48

REM Adaptive parameters
set USE_ADAPTIVE_INR=--use_adaptive_inr
set CONTRAST_METHOD=combined
set MIN_DENSITY=0.1
set MAX_DENSITY=1.0

REM Training parameters
set WARMUP_EPOCHS=10
set SCHEDULER_TYPE=cosine
set SAVE_EVERY=50

echo Configuration:
echo   Data path: %DATA_PATH%
echo   Save directory: %SAVE_DIR%
echo   Log directory: %LOG_DIR%
echo   Batch size: %BATCH_SIZE%
echo   Epochs: %NUM_EPOCHS%
echo   Learning rate: %LEARNING_RATE%
echo   Patch size: %PATCH_SIZE%
echo   Model dimension: %DIM%
echo   Adaptive INR: Enabled
echo   Contrast method: %CONTRAST_METHOD%
echo   Sampling density: [%MIN_DENSITY%, %MAX_DENSITY%]
echo.

REM Create directories
if not exist "%SAVE_DIR%" mkdir "%SAVE_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo Starting training...
python Adaptive/train_adaptive.py ^
    --data_path "%DATA_PATH%" ^
    --save_dir "%SAVE_DIR%" ^
    --log_dir "%LOG_DIR%" ^
    --batch_size %BATCH_SIZE% ^
    --num_epochs %NUM_EPOCHS% ^
    --learning_rate %LEARNING_RATE% ^
    --patch_size %PATCH_SIZE% ^
    --dim %DIM% ^
    %USE_ADAPTIVE_INR% ^
    --contrast_method %CONTRAST_METHOD% ^
    --min_density %MIN_DENSITY% ^
    --max_density %MAX_DENSITY% ^
    --warmup_epochs %WARMUP_EPOCHS% ^
    --scheduler_type %SCHEDULER_TYPE% ^
    --save_every %SAVE_EVERY%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Training completed successfully!
    echo Best model saved to: %SAVE_DIR%\best_model.pth
    echo Final model saved to: %SAVE_DIR%\final_model.pth
    echo Logs available at: %LOG_DIR%
    echo.
    echo To view training progress, run:
    echo   tensorboard --logdir=%LOG_DIR%
) else (
    echo.
    echo Training failed with error code %ERRORLEVEL%
    echo Please check the error messages above.
)

pause