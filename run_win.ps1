<#
  run_win.ps1 - Sobe o ARGOS NESTE PC (Windows) com o MedGemma 4B local
  (transformers + CUDA), na ordem: Gateway MedGemma 4B (:8001) -> Webapp (:8080),
  com verificacao de saude entre as etapas. Ctrl+C encerra o webapp e o gateway.

  Contraparte do run_mac.sh (que usa o 27B via Ollama). Trocar de backend =
  rodar o launcher da maquina: run_win.ps1 aqui (4B), run_mac.sh no MAC (27B).

  Uso:  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_win.ps1
  Setup unico (se o preflight reclamar de dependencias). ORDEM IMPORTA: o extra
  [seg] (TotalSegmentator) resolve seu proprio torch/torchvision sem CUDA a
  partir do indice padrao do PyPI, entao o par CUDA precisa ser reinstalado
  POR CIMA, por ultimo:
        .\.venv-win\Scripts\python.exe -m pip install -e ".[medgemma,seg]"
        .\.venv-win\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
        .\.venv-win\Scripts\hf.exe auth login   # se o token nao estiver salvo
        .\.venv-win\Scripts\python.exe tools\setup_medgemma.py --config configs\medgemma_local_4b.yaml
#>
param(
  [string]$Venv = ".venv-win",
  [string]$Config = "configs\medgemma_local_4b.yaml",
  [string]$ExpectedModel = "google/medgemma-1.5-4b-it",
  [int]$GatewayPort = 8001,
  [int]$WebappPort = 8080,
  [int]$GatewayWaitSeconds = 600
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot ".")).Path
Set-Location $Repo
$Py = Join-Path $Repo "$Venv\Scripts\python.exe"
$LogDir = Join-Path $Repo "casos"; New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$GwOut = Join-Path $LogDir "run_gateway.out.log"
$GwErr = Join-Path $LogDir "run_gateway.err.log"
$gateway = $null

function Say($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok($m)  { Write-Host "    OK  $m" -ForegroundColor Green }
function Die($m) { Write-Host "`nERRO: $m" -ForegroundColor Red; exit 1 }

# --- 0) Preflight ----------------------------------------------------
Say "Preflight (MedGemma 4B local)"
if (-not (Test-Path $Py)) {
  Die "venv Windows nao encontrado: $Py`n    Crie e instale uma vez (ordem importa: extras primeiro, CUDA por cima):`n      py -3.13 -m venv $Venv`n      $Venv\Scripts\python.exe -m pip install -e `".[medgemma,seg]`"`n      $Venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124"
}
if (-not (Test-Path (Join-Path $Repo $Config))) { Die "config nao encontrada: $Config" }
$dependencyProbe = @"
import importlib.util
import sys
required = ('dtwin', 'uvicorn', 'fastapi', 'torch', 'transformers', 'bitsandbytes', 'totalsegmentator')
missing = [name for name in required if importlib.util.find_spec(name) is None]
print('Dependencias ausentes: ' + ', '.join(missing) if missing else 'Dependencias Python presentes.')
sys.exit(1 if missing else 0)
"@
& $Py -c $dependencyProbe
if ($LASTEXITCODE -ne 0) {
  Die "dependencias do 4B/segmentacao ausentes em $Venv. Rode (nesta ordem):`n      $Venv\Scripts\python.exe -m pip install -e `".[medgemma,seg]`"`n      $Venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124"
}
$cuda = (& $Py -c "import torch; print(torch.cuda.is_available())" 2>$null | Out-String).Trim()
if ($cuda -ne "True") {
  Die "CUDA indisponivel no torch deste venv (o 4B usa device=cuda). O extra [seg] costuma sobrescrever o torch com uma build sem CUDA; reinstale o par CUDA por cima:`n      $Venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124"
}
& $Py "tools\setup_medgemma.py" --config $Config --local-only
if ($LASTEXITCODE -ne 0) {
  Die "Pesos locais do MedGemma 4B nao estao completos. Aceite a licenca, execute '$Venv\Scripts\hf.exe auth login' e depois '$Venv\Scripts\python.exe tools\setup_medgemma.py --config $Config'."
}
Ok "venv, config, dependencias, CUDA e pesos do 4B presentes."

# --- 1) Gateway MedGemma 4B (:8001) ---------------------------------
Say "Gateway MedGemma 4B (:$GatewayPort)"
$already = $false
try {
  $h = Invoke-RestMethod -Uri "http://127.0.0.1:$GatewayPort/health" -TimeoutSec 3
  if ($h.status -eq "ready") {
    if ($h.model_id -ne $ExpectedModel) {
      Die "A porta $GatewayPort ja serve o modelo '$($h.model_id)', mas este launcher exige '$ExpectedModel'. Encerre o outro gateway antes de continuar."
    }
    $already = $true
  }
} catch {
  $already = $false
}
if ($already) {
  Ok "Gateway 4B ja estava pronto (modelo $ExpectedModel)."
} else {
  if (Get-NetTCPConnection -LocalPort $GatewayPort -State Listen -ErrorAction SilentlyContinue) { Die "Porta $GatewayPort ja esta em uso." }
  $srvArgs = @("tools\medgemma_server.py", "--config", $Config, "--host", "127.0.0.1", "--port", "$GatewayPort")
  $gateway = Start-Process -FilePath $Py -ArgumentList $srvArgs -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -RedirectStandardOutput $GwOut -RedirectStandardError $GwErr
  Write-Host "    carregando o 4B na GPU (pode levar alguns minutos)..."
  $deadline = (Get-Date).AddSeconds($GatewayWaitSeconds); $ready = $false
  while ((Get-Date) -lt $deadline) {
    if ($gateway.HasExited) { if (Test-Path $GwErr) { Get-Content -Tail 25 $GwErr }; Die "O gateway encerrou durante o carregamento." }
    try {
      $h = Invoke-RestMethod -Uri "http://127.0.0.1:$GatewayPort/health" -TimeoutSec 5
      if ($h.status -eq "ready" -and $h.model_loaded) {
        if ($h.model_id -ne $ExpectedModel) {
          Die "Gateway iniciou modelo inesperado '$($h.model_id)' (esperado: '$ExpectedModel')."
        }
        $ready = $true
        break
      }
      if ($h.status -eq "failed") { Die "MedGemma falhou ao carregar: $($h.load_error)" }
    } catch { }
    Start-Sleep -Seconds 3
  }
  if (-not $ready) { if (Test-Path $GwErr) { Get-Content -Tail 25 $GwErr }; Die "Timeout aguardando o gateway 4B (veja $GwErr)." }
  Ok "Gateway 4B pronto (PID=$($gateway.Id))."
}

# --- 2) Webapp (:8080) — primeiro plano -----------------------------
Say "Webapp (:$WebappPort)"
$busy = $false
try { Invoke-RestMethod -Uri "http://127.0.0.1:$WebappPort/api/health" -TimeoutSec 2 | Out-Null; $busy = $true } catch { $busy = $false }
if ($busy) { Die "Ja existe algo respondendo em :$WebappPort. Feche antes de subir o webapp." }
$env:WEBAPP_MEDGEMMA_CONFIG = $Config
$env:WEBAPP_MEDGEMMA_HEALTH = "http://127.0.0.1:$GatewayPort/health"
Write-Host "`n  ABRA NO NAVEGADOR:  http://127.0.0.1:$WebappPort" -ForegroundColor Green
Write-Host "  (Ctrl+C encerra o webapp e o gateway)`n"
Start-Job -ArgumentList "http://127.0.0.1:$WebappPort" -ScriptBlock { param($u) Start-Sleep 4; Start-Process $u } | Out-Null
try {
  & $Py -m uvicorn webapp.server:app --host 127.0.0.1 --port $WebappPort
} finally {
  if ($gateway -and -not $gateway.HasExited) { Stop-Process -Id $gateway.Id -Force -ErrorAction SilentlyContinue; Ok "Gateway encerrado." }
}
