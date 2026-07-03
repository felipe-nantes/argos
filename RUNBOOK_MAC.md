# Runbook — Rodar o projeto no MAC

Guia operacional para executar o Digital Twin Cirúrgico no MAC (única máquina de
execução: Ollama + MedGemma 27B + segmentação).
Caminho assumido: `/Users/sander_gurgel/Documents/projetos_sander/argos-main`.

## Serviços e ordem de subida

| # | Serviço | Porta | Terminal |
|---|---|---|---|
| 1 | Ollama (daemon + modelo `medgemma:27b-it-bf16`) | 11434 | aba 1 |
| 2 | Gateway MedGemma | 8001 | aba 2 |
| 3 | Webapp (upload + relatório + 3D) | 8080 | aba 3 |

Ordem **obrigatória**: Ollama → Gateway → Webapp. Cada serviço roda em primeiro
plano — use uma aba de terminal para cada. Rode **sempre a partir da raiz do repo**.

## Modo rápido (um comando)

O script `run_mac.sh` sobe tudo na ordem certa, com verificações de saúde entre
cada etapa, e encerra o gateway/webapp no Ctrl+C:

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main
bash run_mac.sh          # ou ./run_mac.sh
```

Abra `http://127.0.0.1:8080` quando ele indicar. Os passos manuais abaixo
continuam válidos para diagnóstico ou se preferir controlar cada serviço.

---

## 0. Atualizar o código (aplicar o bundle vindo do PC Windows)

Só quando houver mudança nova vinda do Windows. Transfira o `argos-main.bundle`
para o MAC e:

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main

BKP=~/argos_bkp_$(date +%Y%m%d_%H%M%S); mkdir -p "$BKP"
cp -R webapp viewer dtwin configs profiles tools tests docs contexto *.py *.md *.toml *.txt "$BKP" 2>/dev/null
echo "backup em: $BKP"

git init -b main 2>/dev/null || git init 2>/dev/null
git fetch ~/Downloads/argos-main.bundle main
git reset --hard FETCH_HEAD        # .venv/ e casos/ ficam intactos (gitignored)
git log --oneline -3
```

## 1. Verificar o ambiente

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main
.venv/bin/python digital_twin.py doctor
```
Esperado: "Núcleo completo" e uma linha `torch device: ...`. Só reinstale se o
doctor acusar falta de dependência (o pacote não mudou nesta atualização):
```bash
.venv/bin/python -m pip install -e ".[webapp,medgemma,seg]"
```

## 2. Aba 1 — Ollama

```bash
ollama serve                      # se ainda não estiver rodando como serviço
```
Confirme o modelo (em outra aba):
```bash
ollama list | grep medgemma       # deve listar medgemma:27b-it-bf16
```

## 3. Aba 2 — Gateway MedGemma (:8001)

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main
.venv/bin/python tools/medgemma_server.py --config configs/medgemma_ollama_27b.yaml --port 8001
```
Confirme:
```bash
curl -s http://127.0.0.1:8001/health
# esperado: "status":"ready" ... "model_id":"medgemma:27b-it-bf16"
```
`"status":"failed"` → leia o campo `load_error` (Ollama fora do ar ou modelo sem visão).

## 4. Aba 3 — Webapp (:8080)

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main
WEBAPP_MEDGEMMA_CONFIG=configs/medgemma_ollama_27b.yaml \
  .venv/bin/python -m uvicorn webapp.server:app --port 8080
```
Confirme:
```bash
curl -s http://127.0.0.1:8080/api/health
# esperado: {"backend":"pronto"}
```
`"desligado"` → o gateway/Ollama não estão prontos; volte aos passos 2 e 3.

## 5. Usar o fluxo

1. Abra `http://127.0.0.1:8080` no navegador.
2. Arraste a **pasta DICOM da RM** (ou um DICOM multi-frame).
3. Aguarde: des-identificação → segmentação do fígado → painel 2D → MedGemma. A
   **primeira** análise demora mais (Ollama carrega o 27B na memória).
4. Sai o **relatório** (sempre `pending_review`). Se algo falhar, aparece um
   cartão honesto "análise não concluída" — nunca um achado fabricado.
5. Clique em **"Visualizar fígado em 3D e revisar"**.
6. No visualizador, inspecione o contorno e registre **Aprovar segmentação** ou
   **Solicitar revisão** → salvo em `casos/webapp/<id>/case/outputs/approval.json`.

## 6. Encerrar

`Ctrl+C` nas abas do webapp e do gateway. O Ollama pode continuar no ar.

## Solução de problemas

- **Webapp `backend: desligado`** → gateway (:8001) ou Ollama fora do ar (`curl .../8001/health`).
- **Gateway `status: failed`** → `load_error`: rode `ollama serve`; confira a tag com `ollama list`.
- **`ModuleNotFoundError: torch` no gateway** → `pip install -e ".[seg]"` (o /health usa torch).
- **Segmentação cai da GPU para CPU** → normal no MAC (sem CUDA); mais lento (timeout até ~40 min).
- **Rode sempre da raiz do repo** — o webapp grava em `casos/webapp/` relativo ao diretório atual.
- **Nunca commite `casos/`** — é dado de paciente e já está no `.gitignore`.
