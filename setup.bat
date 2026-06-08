@echo off
REM ============================================================
REM AV-XAI Project - Environment Setup (Windows)
REM Creates conda env 'df', installs all dependencies, registers Jupyter kernel
REM
REM HOW TO RUN:
REM   1. Open Anaconda Prompt (NOT regular cmd)
REM   2. cd to this project folder
REM   3. Run: setup.bat
REM ============================================================

echo.
echo [AV-XAI] Starting environment setup...
echo.

REM ---------- Verify conda is available ----------
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda not found in PATH.
    echo Please run this from an Anaconda Prompt, not regular cmd.
    pause
    exit /b 1
)

REM ---------- Create environment ----------
echo [1/5] Creating conda environment 'df' with Python 3.11...
call conda create -n df python=3.11 -y
if errorlevel 1 goto error

REM ---------- Activate ----------
echo.
echo [2/5] Activating environment 'df'...
call conda activate df
if errorlevel 1 goto error

REM ---------- Install PyTorch with CUDA 12.1 ----------
REM Note: A6000 supports both CUDA 11.8 and 12.1. If install fails,
REM try changing cu121 to cu118 below.
echo.
echo [3/5] Installing PyTorch with CUDA 12.1 support (may take a few minutes)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto error

REM ---------- Install other dependencies ----------
echo.
echo [4/5] Installing remaining dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 goto error

REM ---------- Register Jupyter kernel ----------
echo.
echo [5/5] Registering Jupyter kernel for 'df'...
python -m ipykernel install --user --name df --display-name "Python (df)"
if errorlevel 1 goto error

REM ---------- Verify CUDA is working ----------
echo.
echo Verifying CUDA availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('PyTorch:', torch.__version__)"

echo.
echo ============================================================
echo [SUCCESS] Environment 'df' is ready to use.
echo.
echo NEXT STEPS:
echo   1. Get your Kaggle API token (kaggle.json) from kaggle.com/settings
echo   2. Place it in: C:\Users\%USERNAME%\.kaggle\kaggle.json
echo   3. Accept DFDC competition rules at:
echo      https://www.kaggle.com/c/deepfake-detection-challenge/rules
echo   4. Open Jupyter: jupyter notebook
echo   5. Open notebooks/01_setup_and_download.ipynb
echo   6. Select kernel: Python (df)
echo ============================================================
pause
exit /b 0

:error
echo.
echo [ERROR] Setup failed. Check messages above.
echo Common fixes:
echo   - If CUDA install failed, edit setup.bat: change cu121 to cu118
echo   - If conda activate fails, run: conda init cmd.exe (then restart prompt)
pause
exit /b 1
