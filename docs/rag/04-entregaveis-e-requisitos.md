# 04 · Entregáveis, construção do zero e requisitos (Fase 4)

> Planejamento. Lista acionável do que precisa existir **antes** de qualquer
> código, o que criar/modificar, dependências (com verificação Apple Silicon),
> hardware, riscos e rollout. Ponto de partida: **NADA existe** (sem KB, sem golden
> set, sem validador clínico).

## 1. Construção da base de conhecimento do zero

**Antes de qualquer código de pipeline, é preciso reunir e curar o corpus.**

1. **Sourcing (inglês):** Opção B consolidada, **sem EASL** nesta etapa: PMC Open
   Access, StatPearls, anatomia de Couinaud e resumos autorais/seguros de
   LI-RADS/AASLD quando possível. A lista final de títulos/URLs ainda é insumo
   pendente.
2. **Licenciamento:** para cada fonte, registrar licença e decidir *texto integral*
   (redistribuível) vs *apenas resumo autoral* (restrito). Sem licença → não entra.
3. **Formato:** um arquivo **Markdown/JSON limpo por documento/seção**, com
   headings (o chunking adaptativo depende disso) e metadados obrigatórios
   (`source`, `license`, `url`, `retrieved_at`, `section`).
4. **Pipeline de ingestão (desenho, não implementar agora):**
   `carregar docs → chunk adaptativo (~480 tokens úteis; §01 §2) → embutir (embedding local) →
   gravar em Chroma + índice BM25 → emitir manifest do índice (corpus_version,
   hash, contagem de chunks, modelo de embedding)`.
5. **Localização dos artefatos:** corpus-fonte e índice **versionados fora de
   `casos/`** (que é dado de paciente, gitignored — [`RUNBOOK_MAC.md`](../../RUNBOOK_MAC.md)).
   Sugestão: `rag/corpus/` (fontes) e `rag/index/` (índice construído; pode ser
   git-LFS ou reconstruível por comando).

**Reunir antes de codar:** lista de fontes aprovada + licenças + ~dezenas de
documentos curados em Markdown. Sem isso, não há o que indexar.

## 2. Construção do golden set do zero

- **Abordagem:** synthetic-first (RAGAS `TestsetGenerator`/DeepEval `Synthesizer`
  com LLM-juiz **local**), ~100 QAs, a partir do corpus do §1. Ver Fase 2 §4.
- **Validação:** revisão humana amostral obrigatória; QA sintético é rascunho até
  aprovação. **Dono da validação = PA-3 (Felipe).**
- **Versionamento:** dataset versionado com `dataset_version` + hash do corpus.

## 3. Arquivos a criar (propostos) e a modificar

### A criar (novos)

| Arquivo/módulo | Responsabilidade |
|---|---|
| `configs/medgemma_local_4b_rag.yaml` | Config do modo RAG (bloco `rag:`, herda baseline) |
| `configs/medgemma_ollama_27b_rag.yaml` | Mesma config para o deploy Mac (Ollama 27B) |
| `dtwin/rag/__init__.py` | Pacote RAG isolado |
| `dtwin/rag/chunking.py` | Chunking adaptativo por seção (+ teto de tokens) |
| `dtwin/rag/embedding.py` | Cliente de embedding (Ollama/sentence-transformers) |
| `dtwin/rag/index.py` | Build/load do Chroma + BM25; manifest do índice + hash |
| `dtwin/rag/retriever.py` | BM25+denso → RRF → rerank → top-n; `min_score`/abstenção |
| `dtwin/rag/grounding.py` | Injeção de `[S#]`, validação de citação (fail-closed) |
| `tools/build_rag_index.py` | CLI: corpus → índice (ingestão auditável) |
| `tools/eval_rag.py` | CLI de avaliação RAGAS/DeepEval (juiz local), separado do gate clínico |
| `rag/corpus/…` | Documentos-fonte curados (Markdown/JSON + metadados) |
| `tests/test_rag_*.py` | Testes de chunking, RRF, abstenção, validação de citação, validação de config |

### A modificar (com motivo)

| Arquivo | Mudança | Motivo |
|---|---|---|
| `dtwin/medgemma_client.py` | Validar bloco `rag` em `_validate_config` | Regras em config, fail-closed (Fase 3 §4) |
| `dtwin/medgemma_screening.py` | Ramo RAG em `run_screening` | Ponto de inserção do retrieve-then-verify (Fase 3 §3) |
| `webapp/server.py` | +`"rag"` em `BENCHMARK_SCENARIOS` (+ env var) | Selecionável no benchmark como os outros (Fase 3 §2.1) |
| `webapp/static/benchmark.html` | 3º botão `data-scenario="rag"` + `SCENARIO_LABEL` | Expor o modo na UI |
| `webapp/static/argos.css` | Grid de cenário 2→3 colunas | Cosmético |
| `pyproject.toml` | Novo extra `[rag]` | Isolar deps do RAG (Fase 4 §4) |
| `docs/RUNNING.md` / `RUNBOOK_MAC.md` | Passo de build do índice + subir modo RAG | Operação |

> **Convenção respeitada:** o motor (`dtwin/`) ganha um **subpacote isolado**
> (`dtwin/rag/`), o modo é um **arquivo de config**, e o gate clínico
> (`metrics.py`, `validate_medgemma_report`) permanece intocado como autoridade
> final.

## 4. Dependências novas (com verificação Apple Silicon/MPS)

Proposta de extra isolado em `pyproject.toml`:

```toml
# ilustrativo
[project.optional-dependencies]
rag = [
    "chromadb>=0.5",            # vector store; CPU/ARM (pip puro) — OK Apple Silicon
    "bm25s>=0.2",               # BM25 numpy puro; CPU — OK Apple Silicon
    "sentence-transformers>=3", # MedCPT/BGE embedding + cross-encoder; MPS/CPU — OK
    "transformers>=4.44",       # MedCPT encoders; MPS/CPU — OK
    "ragas>=0.2",               # avaliação reference-free; juiz local via Ollama — OK
    # fallback via Ollama/nomic reusa o daemon já instalado (nenhuma dep nova)
]
```

| Dependência | Papel | Apple Silicon / MPS | CUDA-only? |
|---|---|---|---|
| chromadb | Vector store | **OK** (CPU/ARM, pip) | Não |
| bm25s | Esparso (BM25) | **OK** (numpy, CPU) | Não |
| sentence-transformers | Embedding + reranker | **OK** (usa PyTorch MPS/CPU) | Não |
| transformers | MedCPT encoders/cross-encoder | **OK** (MPS/CPU) | Não |
| ragas | Métricas RAG | **OK** (juiz = Ollama local) | Não |
| torch (já presente via `[medgemma]`) | Backend de emb/rerank | **OK** (MPS) | Não |

**Verificação de bloqueantes CUDA-only:** nenhuma das deps propostas exige CUDA.
**`bitsandbytes` NÃO entra no caminho RAG** (é usado só no `device: cuda` do
gateway — [`medgemma_server.py:88-96`](../../tools/medgemma_server.py)); no Mac o
embedding/reranker rodam em **MPS/CPU** e o gerador via **Ollama/Metal**. Se
alguma lib puxar `faiss-gpu` ou similar por engano → **BLOQUEANTE**, trocar por
`faiss-cpu`/Chroma. **Regra:** toda dep nova roda em MPS **ou** CPU; qualquer
requisito CUDA é bloqueante e deve ter alternativa MPS/CPU antes de entrar.

## 5. Requisitos de hardware (estimativa)

Premissa confirmada na tarefa: **o único peso real de memória é o LLM gerador (já
contabilizado)**; embedding (<1 GB) e reranker (<1 GB) são pequenos.

| Componente | Dev box (8 GB VRAM) | M5 Max (128 GB unificada) |
|---|---|---|
| LLM gerador (MedGemma) | ~3,6–4 GB VRAM (já em uso) | 27B via Ollama, folgado |
| Embedding (MedCPT; fallback nomic/BGE) | <1 GB — **rodar em CPU** p/ não tocar a VRAM | MPS, trivial |
| Reranker (MedCPT cross-encoder; fallback bge) | <1 GB — CPU/compartilhado | MPS, trivial |
| Índice Chroma + BM25 (corpus pequeno) | Disco: dezenas–centenas de MB; RAM: <1 GB | idem, irrelevante |
| RAGAS (avaliação, offline) | Juiz = Ollama; roda fora do horário de inferência | idem |

**Conclusão:** **uma única config roda nos dois** — no dev box, embedding/reranker
em CPU (não disputam a VRAM do LLM); no Mac, tudo em MPS/Metal com folga enorme.
Diferença entre ambientes fica **em config** (device do embedding/reranker,
`rerank_model`), não em código.

## 6. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| **Privacidade de dado hospitalar** | Corpus é **conhecimento público em inglês**, nunca dado de paciente; índice fora de `casos/`; tudo local (sem nuvem); juiz de avaliação **local** (Ollama). Mantém `regulatory_mode: RESEARCH`. |
| **Alucinação residual** | Grounding com `[S#]` verificável + abstenção → INCONCLUSIVA; gate clínico (`validate_medgemma_report`) segue como autoridade; faithfulness com piso (Fase 2 §7). |
| **Ausência de validação clínica** | Declarada como lacuna (Fase 2 §8); modo permanece RESEARCH; spot-check humano proposto; não prometer "confiabilidade clínica". |
| **Drift do índice** | `corpus_version` + hash no envelope/manifest; resultado rastreável ao conhecimento exato; reconstrução por comando versionada. |
| **1ª passagem polui a consulta (topologia B)** | Consulta-base fixa sempre ancora; `min_score` filtra recuperação fraca; opção de cair para topologia A (single-pass) via config. |
| **RAG piora vs baseline** | O gate exige **não regredir** sens/esp ≥ 0,75 e comparação com IC de Wilson; se não superar, o modo não é promovido. |
| **Corpus incompleto** | Abstenção em vez de invenção; cobertura menor é aceitável e medida. |

## 7. Rollout faseado

1. **F0 — Fundações (sem LLM):** curar corpus mínimo (§1) + `build_rag_index` +
   testar recuperação isolada (recall@k/MRR, Fase 2 §1), incluindo A/B MedCPT vs
   fallback leve. *Entrega:* índice + números de recuperação. Prova barata de que o
   retriever presta **antes** de integrar.
2. **F1 — MVP integrado:** config `rag`, ramo em `run_screening` com
   retrieve-then-verify sobre o painel `baseline`, grounding+abstenção, saída no
   envelope. Rodar no webapp/CLI como 3º cenário.
   *Entrega:* modo `rag` selecionável, fail-closed.
3. **F2 — Prova de confiabilidade:** golden set sintético validado + RAGAS
   reference-free + tabela comparativa clínica com ICs (Fase 2 §6). *Entrega:*
   evidência de ganho (ou não) vs baseline/volumétrico.
4. **F3 — Evolução:** combinar `rag + volumetric` se a latência permitir, Qdrant se
   o corpus crescer, citação por-claim no JSON, spot-check clínico (Fase 1 §9, Fase
   2 §8).

**MVP mínimo confiável = F0+F1** com `regulatory_mode: RESEARCH`, fail-closed, e
recuperação já medida. A **prova** de ganho é F2.

## 8. Decisões consolidadas e insumos pendentes

**Consolidado:**

1. Corpus = Opção B sem EASL.
2. Labels binários positivo/negativo são aceitáveis para sensibilidade/especificidade
   porque a equipe controla as entradas.
3. Sem validação clínica formal por enquanto; modo permanece `RESEARCH`.
4. MedCPT é alvo de qualidade; fallback leve deve ser medido.
5. Topologia = retrieve-then-verify.
6. Citações/proveniência = envelope externo.
7. Thresholds = provisórios e calibrados após o primeiro run.

**Ainda falta como insumo, não como decisão arquitetural:**

1. Lista concreta das fontes do corpus (títulos/URLs/licenças).
2. Conjunto de RMs saudáveis/doentes para comparar `baseline`, `volumetric` e `rag`.

Índice geral: [`README.md`](README.md).
