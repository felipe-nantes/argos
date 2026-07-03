#!/usr/bin/env bash
# =====================================================================
# run_mac.sh — sobe o Digital Twin Cirúrgico no MAC na ordem correta:
#   Ollama (:11434) -> Gateway MedGemma (:8001) -> Webapp (:8080)
# com verificação de saúde entre cada etapa. Ctrl+C encerra o webapp e
# o gateway (deixa o Ollama no ar se ele já estava rodando).
#
#   Uso:  bash run_mac.sh
# =====================================================================
set -euo pipefail

# --- Parâmetros (só mexa se mudar as portas do runbook) -------------
GATEWAY_PORT=8001
WEBAPP_PORT=8080
OLLAMA_URL="http://127.0.0.1:11434"
MEDGEMMA_CONFIG="configs/medgemma_ollama_27b.yaml"
OLLAMA_TAG="medgemma:27b-it-bf16"

# --- Roda sempre a partir da raiz do repo ---------------------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOGDIR="$REPO/casos"; mkdir -p "$LOGDIR"
GW_LOG="$LOGDIR/run_gateway.log"
OL_LOG="$LOGDIR/run_ollama.log"

GATEWAY_PID=""; OLLAMA_PID=""; OLLAMA_STARTED=0

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()  { printf '    \033[1;32mOK\033[0m  %s\n' "$*"; }
die() { printf '\n\033[1;31mERRO:\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() {
  trap - INT TERM EXIT
  say "Encerrando"
  [ -n "$GATEWAY_PID" ] && kill "$GATEWAY_PID" 2>/dev/null || true
  if [ "$OLLAMA_STARTED" = "1" ] && [ -n "$OLLAMA_PID" ]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
    ok "Ollama parado (foi iniciado por este script)."
  else
    ok "Ollama deixado no ar."
  fi
  ok "Gateway parado."
}
trap cleanup INT TERM EXIT

# wait_http <url> <timeout_s> [regex_esperado_no_corpo]
wait_http() {
  local url="$1" timeout="$2" re="${3:-}" start body
  start=$(date +%s)
  while :; do
    body="$(curl -fsS --max-time 5 "$url" 2>/dev/null || true)"
    if [ -n "$body" ]; then
      if [ -z "$re" ] || printf '%s' "$body" | grep -Eq "$re"; then return 0; fi
    fi
    [ $(( $(date +%s) - start )) -ge "$timeout" ] && return 1
    sleep 2
  done
}

# --- 0) Preflight ----------------------------------------------------
say "Preflight"
command -v curl   >/dev/null 2>&1 || die "curl não encontrado."
command -v ollama >/dev/null 2>&1 || die "ollama não encontrado. Instale o Ollama."
[ -x "$PY" ] || die "venv não encontrado em $PY
    Crie com: python3.12 -m venv .venv && .venv/bin/python -m pip install -e '.[webapp,medgemma,seg]'"
[ -f "$MEDGEMMA_CONFIG" ] || die "config não encontrada: $MEDGEMMA_CONFIG (rode a partir da raiz do repo)"
"$PY" -c "import dtwin, uvicorn, fastapi, SimpleITK, pydicom, yaml" 2>/dev/null \
  || die "dependências ausentes no venv. Rode: $PY -m pip install -e '.[webapp,medgemma,seg]'"
ok "venv, config e dependências presentes."

# --- 1) Ollama (:11434) ---------------------------------------------
say "Ollama (:11434)"
if wait_http "$OLLAMA_URL/api/tags" 3 ""; then
  ok "Ollama já está no ar."
else
  ok "Ollama não respondeu; iniciando 'ollama serve'..."
  nohup ollama serve >"$OL_LOG" 2>&1 &
  OLLAMA_PID=$!; OLLAMA_STARTED=1
  wait_http "$OLLAMA_URL/api/tags" 30 "" || die "Ollama não subiu. Veja $OL_LOG"
  ok "Ollama iniciado (pid $OLLAMA_PID)."
fi
# Checa a tag pela API /api/tags (com retry via wait_http), não pelo CLI
# `ollama list` numa tentativa única: o CLI pode falhar transitoriamente
# enquanto o daemon está ocupado, derrubando o launch sem motivo real.
if wait_http "$OLLAMA_URL/api/tags" 10 "\"$OLLAMA_TAG\""; then
  ok "Modelo $OLLAMA_TAG disponível."
else
  die "Modelo '$OLLAMA_TAG' não encontrado no Ollama. Confira com: ollama list"
fi

# --- 2) Gateway MedGemma (:8001) ------------------------------------
say "Gateway MedGemma (:$GATEWAY_PORT)"
GW_READY_RE='"status"[[:space:]]*:[[:space:]]*"ready"'
if wait_http "http://127.0.0.1:$GATEWAY_PORT/health" 3 "$GW_READY_RE"; then
  ok "Gateway já estava pronto."
else
  "$PY" tools/medgemma_server.py --config "$MEDGEMMA_CONFIG" --port "$GATEWAY_PORT" >"$GW_LOG" 2>&1 &
  GATEWAY_PID=$!
  printf '    aguardando o gateway ficar pronto (até 180s)...\n'
  if wait_http "http://127.0.0.1:$GATEWAY_PORT/health" 180 "$GW_READY_RE"; then
    ok "Gateway pronto (pid $GATEWAY_PID)."
  else
    printf '    --- últimas linhas de %s ---\n' "$GW_LOG"
    tail -n 20 "$GW_LOG" 2>/dev/null || true
    die "Gateway não ficou pronto. Verifique: Ollama no ar? modelo com visão? torch instalado (.[seg])?"
  fi
fi

# --- 3) Webapp (:8080) — primeiro plano -----------------------------
say "Webapp (:$WEBAPP_PORT)"
if wait_http "http://127.0.0.1:$WEBAPP_PORT/api/health" 2 ""; then
  die "Já existe algo respondendo em :$WEBAPP_PORT. Feche antes de subir o webapp."
fi
printf '\n\033[1;32m  ABRA NO NAVEGADOR:  http://127.0.0.1:%s\033[0m\n' "$WEBAPP_PORT"
printf '  (Ctrl+C aqui encerra o webapp e o gateway)\n\n'
command -v open >/dev/null 2>&1 && ( sleep 4; open "http://127.0.0.1:$WEBAPP_PORT" >/dev/null 2>&1 || true ) &
WEBAPP_MEDGEMMA_CONFIG="$MEDGEMMA_CONFIG" \
WEBAPP_MEDGEMMA_HEALTH="http://127.0.0.1:$GATEWAY_PORT/health" \
  "$PY" -m uvicorn webapp.server:app --host 127.0.0.1 --port "$WEBAPP_PORT"
