param([string]$Kernel = "poby7722/arc-jepa-train", [int]$MaxMinutes = 600, [int]$Every = 120)
# Log every status change of a Kaggle kernel until it finishes (or the time limit passes).
$env:PYTHONIOENCODING = "utf-8"
$deadline = (Get-Date).AddMinutes($MaxMinutes)
$last = ""
while ((Get-Date) -lt $deadline) {
    $s = (kaggle kernels status $Kernel 2>$null | Out-String).Trim()
    if ($s -and $s -ne $last) { "$(Get-Date -Format 'MM-dd HH:mm:ss') $s"; $last = $s }
    if ($s -match 'COMPLETE|ERROR|CANCEL') { break }
    Start-Sleep -Seconds $Every
}
"final: $s"
