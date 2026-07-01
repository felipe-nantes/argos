param(
    [string]$Config = "configs\medgemma_local_4b.yaml",
    [int]$Port = 8001,
    [int]$WaitSeconds = 360
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$StateDir = Join-Path $Root ".medgemma"
$PidFile = Join-Path $StateDir "server.pid"
$OutLog = Join-Path $StateDir "server.out.log"
$ErrLog = Join-Path $StateDir "server.err.log"

if (-not (Test-Path $Python)) {
    throw "Python do projeto não encontrado: $Python"
}
if (-not (Test-Path (Join-Path $Root $Config))) {
    throw "Configuração MedGemma não encontrada: $Config"
}

Write-Output "Executando preflight local (GPU, dependências e pesos em cache)..."
& $Python "tools\setup_medgemma.py" --config $Config --local-only
if ($LASTEXITCODE -ne 0) {
    throw "Preflight MedGemma falhou; o backend não será iniciado."
}
New-Item -ItemType Directory -Path $StateDir -Force | Out-Null

if (Test-Path $PidFile) {
    $ExistingPid = [int](Get-Content $PidFile -Raw)
    if (Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue) {
        Write-Output "MedGemma já está em execução (PID=$ExistingPid)."
        exit 0
    }
    Remove-Item -LiteralPath $PidFile -Force
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "A porta $Port já está em uso."
}

$Arguments = @(
    "tools\medgemma_server.py",
    "--config", $Config,
    "--host", "127.0.0.1",
    "--port", "$Port"
)
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments `
    -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog
Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii
Write-Output "Carregando MedGemma (PID=$($Process.Id)); isso pode levar alguns minutos..."

$Deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $Deadline) {
    if (-not (Get-Process -Id $Process.Id -ErrorAction SilentlyContinue)) {
        Write-Output "O backend encerrou durante a inicialização."
        if (Test-Path $ErrLog) { Get-Content -Tail 30 $ErrLog }
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        exit 1
    }
    try {
        $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
        if ($Health.status -eq "ready" -and $Health.model_loaded -eq $true) {
            Write-Output "MedGemma PRONTO: $($Health.model_version) [$($Health.quantization)]"
            exit 0
        }
        if ($Health.status -eq "failed") {
            Write-Output "MedGemma falhou ao carregar: $($Health.load_error)"
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
            exit 1
        }
    } catch {
        # Durante o carregamento o servidor ainda pode não aceitar conexões.
    }
    Start-Sleep -Seconds 2
}
Write-Output "Timeout aguardando MedGemma. Consulte: $ErrLog"
Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
exit 1
