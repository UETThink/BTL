# ===================================================================
# LPR Setup Script — cài tất cả dependency theo đúng thứ tự
# Chạy:  .\setup.ps1            (auto detect GPU)
#        .\setup.ps1 -Cpu       (ép CPU only)
# ===================================================================
param(
    [switch]$Cpu = $false
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " LPR Setup — License Plate Recognition" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# --- 1. Kiểm tra Python ---
Write-Host "`n[1/5] Kiểm tra Python..." -ForegroundColor Yellow
$pyVersion = python --version 2>&1
Write-Host "    $pyVersion"
if ($pyVersion -notmatch "Python 3\.(10|11|12)") {
    Write-Host "    [CẢNH BÁO] Khuyến nghị Python 3.10–3.12 (đang có $pyVersion)" -ForegroundColor Yellow
}

# --- 2. Detect GPU ---
Write-Host "`n[2/5] Kiểm tra GPU NVIDIA..." -ForegroundColor Yellow
$hasGpu = $false
if (-not $Cpu) {
    try {
        $nvidia = nvidia-smi 2>$null
        if ($LASTEXITCODE -eq 0) {
            $hasGpu = $true
            $gpuLine = ($nvidia | Select-String "GeForce|RTX|GTX|Tesla|Quadro" | Select-Object -First 1)
            Write-Host "    GPU phát hiện: $gpuLine" -ForegroundColor Green
        }
    } catch {}
}
if (-not $hasGpu) {
    Write-Host "    Không có GPU NVIDIA — sẽ cài bản CPU." -ForegroundColor Yellow
}

# --- 3. Upgrade pip ---
Write-Host "`n[3/5] Upgrade pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# --- 4. Cài PyTorch + PaddlePaddle (cần index URL riêng) ---
Write-Host "`n[4/5] Cài PyTorch + PaddlePaddle..." -ForegroundColor Yellow
if ($hasGpu) {
    Write-Host "    → PyTorch CUDA 11.8" -ForegroundColor Green
    python -m pip install "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" `
        --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

    Write-Host "    → PaddlePaddle GPU (CUDA 11.7/11.8)" -ForegroundColor Green
    python -m pip install "paddlepaddle-gpu==2.6.1.post117" `
        -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
    if ($LASTEXITCODE -ne 0) { throw "PaddlePaddle GPU install failed" }
} else {
    Write-Host "    → PyTorch CPU" -ForegroundColor Yellow
    python -m pip install "torch==2.7.1" "torchvision==0.22.1"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

    Write-Host "    → PaddlePaddle CPU" -ForegroundColor Yellow
    python -m pip install "paddlepaddle==2.6.2"
    if ($LASTEXITCODE -ne 0) { throw "PaddlePaddle install failed" }
}

# --- 5. Cài phần còn lại từ requirements.txt ---
Write-Host "`n[5/5] Cài các package còn lại từ requirements.txt..." -ForegroundColor Yellow
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements.txt install failed" }

# --- Verify ---
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host " VERIFY" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$verify = @"
import sys
print(f'Python: {sys.version.split()[0]}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

import paddle
print(f'Paddle: {paddle.__version__}  GPU: {paddle.is_compiled_with_cuda()}')

import cv2, ultralytics, fast_alpr, paddleocr, onnxruntime
print(f'OpenCV: {cv2.__version__}')
print(f'Ultralytics: {ultralytics.__version__}')
print(f'ONNX Runtime providers: {onnxruntime.get_available_providers()}')
print('All imports OK')
"@

python -c $verify
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[LỖI] Có package import thất bại — xem log trên." -ForegroundColor Red
    exit 1
}

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host " HOÀN TẤT! Có thể chạy:" -ForegroundColor Green
Write-Host "   python .\test_image_hybrid.py test_image --device cuda" -ForegroundColor Green
Write-Host "   python .\test_video.py video.mp4 --device cuda --half" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
