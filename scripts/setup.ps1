# One-time setup on Windows PowerShell 5.1 (the equivalent of scripts/setup.sh).
# Usage: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 [folder]
# folder defaults to this repo; pass a worktree such as ..\fg-vision to set that one up instead.
param([string]$Folder = (Split-Path -Parent $PSScriptRoot))

function Assert-Ok($step) {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$step failed (exit $LASTEXITCODE)"
        exit 1
    }
}

Set-Location -LiteralPath $Folder
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        Write-Error "python not found on PATH: install Python 3.10-3.12"
        exit 1
    }
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)"
    if ($LASTEXITCODE -ne 0) { Write-Warning "FarmGuard expects Python 3.10-3.12; dependencies may not install" }
    python -m venv .venv
    Assert-Ok "venv"
}

& $py -m pip install --upgrade pip
Assert-Ok "pip upgrade"
& $py -m pip install -e ".[dev]"
Assert-Ok "core install"
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Warning "Some optional packages failed; core still works." }

if (-not (Test-Path .env)) { Copy-Item .env.example .env }

& $py -m farmguard.cli make-sounds
Assert-Ok "make-sounds"
& $py -m farmguard.cli make-demo
Assert-Ok "make-demo"
& $py -m pytest -q
Assert-Ok "pytest"

# Embed the detector vocabulary while online, so runtime needs no CLIP. Mock runs don't need this.
if (-not (Test-Path weights\yolov8s-worldv2-farmguard.pt)) {
    & $py scripts\bake_world_vocab.py
    if ($LASTEXITCODE -ne 0) { Write-Warning "bake_world_vocab failed: the real detector won't start until it succeeds; mock runs still work." }
}

Write-Host "Done. Try: $py -m farmguard.cli run --source data/demo/demo.mp4 --mock-annotations data/demo/demo_annotations.json --zones data/demo/demo_zones.json --provider mock"
