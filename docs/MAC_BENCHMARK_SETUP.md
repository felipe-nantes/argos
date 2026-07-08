# Preparar o Mac para rodar ARGOS e benchmarks

Este guia assume o Mac como a máquina de execução real do ARGOS com MedGemma 27B
via Ollama/Metal. O código deve ir pelo GitHub; artefatos privados ou ignorados
pelo Git devem ir em pacote separado.

## 1. O que vai por Git e o que vai por fora

Vai por Git:

- código do ARGOS;
- configs MedGemma;
- scripts;
- documentação;
- manifests públicos do RAG em `docs/rag/`;
- ferramentas de build do RAG.

Vai por fora do Git:

- datasets médicos, DICOMs, NIfTIs e labels locais;
- diretório `casos/`;
- diretório `dicoms/` ou equivalente;
- corpus/índice RAG já construído em `rag/`, se vocês quiserem evitar reconstrução;
- resultados anteriores de benchmark em `artifacts/`, apenas se forem necessários
  para auditoria.

Por segurança, não coloque DICOMs, NIfTIs, labels locais nem `casos/` no GitHub.

## 2. Atualizar o código no Mac

```bash
cd /Users/sander_gurgel/Documents/projetos_sander

if [ ! -d argos-main/.git ]; then
  git clone https://github.com/felipe-nantes/argos.git argos-main
fi

cd /Users/sander_gurgel/Documents/projetos_sander/argos-main
git checkout main
git pull origin main
git log --oneline -5
```

## 3. Criar ou atualizar o ambiente Python

Use Python 3.10 a 3.13. Se houver Python 3.12 disponível, prefira 3.12.

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main

python3.12 -m venv .venv || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[webapp,seg]"
```

Se também for rodar o servidor Python/Hugging Face local em vez de Ollama, instale
`.[medgemma]`. Para o fluxo recomendado do Mac com Ollama 27B, o gateway usa a API
do Ollama e não precisa carregar o 27B pelo Python.

## 4. Preparar o Ollama e o MedGemma 27B

```bash
ollama serve
```

Em outro terminal:

```bash
ollama list | grep medgemma
```

O modelo esperado pela config atual é:

```text
medgemma:27b-it-bf16
```

Se a tag for diferente, ajuste `model_id` em `configs/medgemma_ollama_27b.yaml`
antes de rodar benchmarks.

## 5. Restaurar artefatos enviados por fora

Se você recebeu o pacote `argos_mac_transfer.zip`, copie-o para o Mac e extraia:

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main
unzip ~/Downloads/argos_mac_transfer.zip -d /tmp/argos_mac_transfer
```

Copie o RAG pré-construído, se existir no pacote:

```bash
mkdir -p rag
rsync -a /tmp/argos_mac_transfer/argos_mac_transfer/rag/ ./rag/
```

Valide se o índice RAG existe:

```bash
test -f rag/index/liver_mri_rag_v1/bm25_index.json && echo "RAG OK"
```

Se o pacote não contiver `rag/`, reconstrua no Mac:

```bash
.venv/bin/python tools/build_rag_corpus.py --clean
.venv/bin/python tools/build_rag_index.py --clean
```

## 6. Subir ARGOS no Mac

Modo normal, baseline 27B via Ollama:

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main
bash run_mac.sh
```

Para benchmark pelo webapp com cenários Mac, suba manualmente informando as configs:

Terminal 1:

```bash
ollama serve
```

Terminal 2:

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main
.venv/bin/python tools/medgemma_server.py \
  --config configs/medgemma_ollama_27b.yaml \
  --port 8001
```

Terminal 3:

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main

WEBAPP_MEDGEMMA_CONFIG=configs/medgemma_ollama_27b.yaml \
WEBAPP_VOLUMETRIC_MEDGEMMA_CONFIG=configs/medgemma_ollama_27b_volumetric.yaml \
WEBAPP_RAG_MEDGEMMA_CONFIG=configs/medgemma_ollama_27b_rag.yaml \
WEBAPP_VOLUMETRIC_RAG_MEDGEMMA_CONFIG=configs/medgemma_ollama_27b_volumetric_rag.yaml \
WEBAPP_MEDGEMMA_HEALTH=http://127.0.0.1:8001/health \
.venv/bin/python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8080
```

Abra:

```text
http://127.0.0.1:8080/benchmark.html
```

## 7. Rodar benchmark por CLI

Primeiro rode um dry-run. Ele valida paths, labels, isolamento de ground truth,
hashes e configuração, sem chamar o MedGemma.

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main

.venv/bin/python -m dtwin.medgemma_benchmark \
  --datasets-manifest casos/benchmark_manifests/datasets.yaml \
  --medgemma-config configs/medgemma_ollama_27b_volumetric.yaml \
  --experiment-config benchmarks/experiments/current_panel.example.yaml \
  --dry-run
```

Depois rode um smoke pequeno:

```bash
.venv/bin/python -m dtwin.medgemma_benchmark \
  --datasets-manifest casos/benchmark_manifests/datasets.yaml \
  --medgemma-config configs/medgemma_ollama_27b_volumetric.yaml \
  --experiment-config benchmarks/experiments/current_panel.example.yaml \
  --out casos/webapp/benchmarks/runs \
  --limit 2
```

Só rode o lote completo depois que o smoke gerar `medgemma_report.json` válido nos
casos testados.

Benchmark completo:

```bash
.venv/bin/python -m dtwin.medgemma_benchmark \
  --datasets-manifest casos/benchmark_manifests/datasets.yaml \
  --medgemma-config configs/medgemma_ollama_27b_volumetric.yaml \
  --experiment-config benchmarks/experiments/current_panel.example.yaml \
  --out casos/webapp/benchmarks/runs
```

## 8. Conferências obrigatórias antes do benchmark real

```bash
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8080/api/health
.venv/bin/python digital_twin.py doctor
git status --short
```

Para avaliação final, use árvore Git limpa. Para desenvolvimento, o benchmark
registra o estado dirty no manifesto, mas isso não deve ser usado como resultado
final.

## 9. Cenários recomendados

Prioridade atual:

1. `configs/medgemma_ollama_27b_volumetric.yaml` — principal candidato.
2. `configs/medgemma_ollama_27b.yaml` — baseline reprodutível.
3. `configs/medgemma_ollama_27b_rag.yaml` — RAG textual, exploratório.
4. `configs/medgemma_ollama_27b_volumetric_rag.yaml` — exploratório; usar com
   cautela porque o RAG aumentou falhas de schema no lote anterior.

Não declare ganho de acurácia sem comparar positivos e negativos, com falhas e
inconclusivos contabilizados como erro na métrica principal.
