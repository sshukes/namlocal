# Create and activate a conda environment for NAM training
param(
    [string]$EnvName = "namlocal",
    [switch]$UseGpu
)

Write-Host "Creating conda env $EnvName (Python 3.11)..."
conda create -y -n $EnvName python=3.11
conda activate $EnvName

if ($UseGpu) {
    Write-Host "Installing CUDA-enabled torch..."
    pip install torch --index-url https://download.pytorch.org/whl/cu129
} else {
    Write-Host "Installing CPU torch..."
    pip install torch
}

pip install neural-amp-modeler PySide6 numpy soundfile pydantic

Write-Host "GPU availability check:"
python -c "import torch; print(torch.cuda.is_available())"
