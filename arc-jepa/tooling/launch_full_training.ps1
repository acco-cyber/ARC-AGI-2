param([string]$Message = "ARC-JEPA code update", [switch]$SkipTrainPush)
# 1) stage the package as a Kaggle dataset folder, 2) publish a new version of poby7722/arc-jepa-code,
# 3) wait until the new version is ready, 4) push the FULL training kernel (kaggle/train_full).
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$src = "E:\Claude code\arc2\ARC-AGI-2\arc-jepa"
$dst = Join-Path $env:TEMP "arc-jepa-code-stage"
if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
New-Item -ItemType Directory -Force $dst | Out-Null
robocopy $src "$dst\arc-jepa" /E /XD __pycache__ .pytest_cache runs out /XF *.pyc *.log synth*.jsonl /NFL /NDL /NJH /NJS | Out-Null
'{"title": "ARC-JEPA code", "id": "poby7722/arc-jepa-code", "licenses": [{"name": "Apache 2.0"}]}' | Out-File -Encoding ascii "$dst\dataset-metadata.json"
$n = (Get-ChildItem "$dst\arc-jepa" -Recurse -File).Count
"staged $n files"
kaggle datasets version -p $dst -m $Message --dir-mode zip 2>&1 | Select-Object -Last 3
$deadline = (Get-Date).AddMinutes(15)
do { Start-Sleep -Seconds 20; $st = (kaggle datasets status poby7722/arc-jepa-code 2>$null | Out-String).Trim(); "dataset status: $st" } while ($st -notmatch 'ready' -and (Get-Date) -lt $deadline)
if ($SkipTrainPush) { "skip train push"; exit 0 }
kaggle kernels push -p "$src\kaggle\train_full" 2>&1 | Select-Object -Last 3
Start-Sleep -Seconds 45
kaggle kernels status poby7722/arc-jepa-train
