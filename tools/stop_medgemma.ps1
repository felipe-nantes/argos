$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PidFile = Join-Path $Root ".medgemma\server.pid"

if (-not (Test-Path $PidFile)) {
    Write-Output "Nenhum PID do MedGemma registrado."
    exit 0
}
$ServerPid = [int](Get-Content $PidFile -Raw)
$ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$ServerPid" -ErrorAction SilentlyContinue
if ($null -eq $ProcessInfo) {
    Remove-Item -LiteralPath $PidFile -Force
    Write-Output "Processo já estava encerrado."
    exit 0
}
if ($ProcessInfo.CommandLine -notmatch "medgemma_server\.py") {
    throw "PID $ServerPid não pertence ao backend MedGemma; nada foi encerrado."
}
Stop-Process -Id $ServerPid -Force
Remove-Item -LiteralPath $PidFile -Force
Write-Output "Backend MedGemma encerrado (PID=$ServerPid)."
