param([string]$Kernel = "poby7722/arc-jepa-infer", [int]$Version = 1, [string]$Message = "ARC-JEPA infer v1", [int]$MaxMinutes = 540)
# Wait for a kernel version to finish its dev run, then submit it if no submission exists for today (UTC).
$env:PYTHONIOENCODING = "utf-8"
$deadline = (Get-Date).AddMinutes($MaxMinutes)
$last = ""
while ((Get-Date) -lt $deadline) {
    $s = (kaggle kernels status $Kernel 2>$null | Out-String).Trim()
    if ($s -ne $last) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $s"; $last = $s }
    if ($s -match 'ERROR|CANCEL') { "kernel failed; not submitting"; exit 2 }
    if ($s -match 'COMPLETE') { break }
    Start-Sleep -Seconds 90
}
if ($s -notmatch 'COMPLETE') { "timed out waiting"; exit 3 }
$todayUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd")
$subs = kaggle competitions submissions arc-prize-2026-arc-agi-2 2>$null | Out-String
if ($subs -match [regex]::Escape($todayUtc)) { "a submission already exists for $todayUtc UTC; not submitting"; exit 4 }
kaggle competitions submit arc-prize-2026-arc-agi-2 -k $Kernel -v $Version -f submission.json -m $Message
Start-Sleep -Seconds 20
kaggle competitions submissions arc-prize-2026-arc-agi-2 2>$null | Select-Object -First 4
