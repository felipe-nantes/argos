# 03 · Design de integração (Fase 3)

> Planejamento. **Nenhuma implementação.** Descreve *como* o modo `rag` se encaixa
> nas convenções da Fase 0 (regras em config, motor determinístico, abortar em
> falha), com pseudocódigo curto e ilustrativo apenas.

## 1. Princípio de encaixe

Do reconhecimento (Fase 0): **não há registro de modos**; o "modo" é um **arquivo
de config**, e a UI mapeia `scenario → config` via um dict. O RAG segue **os dois
mecanismos existentes**, sem inventar um terceiro:

- **É um eixo ORTOGONAL**, não uma nova `panel.strategy`. RAG aumenta o **prompt**;
  a estratégia de painel (`uniform_9`/`volumetric_blocks`) segue independente.
  Assim `rag` pode, no futuro, compor com baseline **ou** volumétrico.
- **Ativa-se por um bloco novo `rag:` no YAML** + um arquivo de config que o liga.

**Decisão operacional v1:** o primeiro modo `rag` deve herdar o baseline
(`uniform_9`), não o volumétrico, para evitar custo `2 × N painéis` enquanto ainda
estamos validando recuperação, grounding e abstenção.

## 2. Registrar/despachar `rag` no mesmo mecanismo dos outros

### 2.1 Webapp — adicionar uma entrada no dict de cenários

Hoje ([`webapp/server.py:69-72`](../../webapp/server.py)):

```python
BENCHMARK_SCENARIOS = {
    "baseline":  MEDGEMMA_CONFIG,
    "volumetric": VOLUMETRIC_MEDGEMMA_CONFIG,
}
```

Proposta (config, não código de dispatch novo):

```python
# ilustrativo
RAG_MEDGEMMA_CONFIG = os.environ.get(
    "WEBAPP_RAG_MEDGEMMA_CONFIG", "configs/medgemma_local_4b_rag.yaml"
)
BENCHMARK_SCENARIOS = {
    "baseline":  MEDGEMMA_CONFIG,
    "volumetric": VOLUMETRIC_MEDGEMMA_CONFIG,
    "rag":       RAG_MEDGEMMA_CONFIG,   # << nova entrada
}
```

Tudo o mais **já funciona sem mudança**: `_parse_benchmark_manifest` valida
`scenario in BENCHMARK_SCENARIOS` ([`server.py:727`](../../webapp/server.py));
`_benchmark_config` resolve o caminho com a trava "dentro de `configs/`"
([`server.py:436-444`](../../webapp/server.py)); `process_benchmark` passa o
config ao executor ([`server.py:623-639`](../../webapp/server.py)). **Justificativa:**
menor superfície de mudança possível; reusa a trava de segurança existente.
**Trade-off:** o executor (`dtwin.medgemma_screening`) precisa saber ativar o RAG a
partir do config (§3).

### 2.2 CLI — nada a registrar

[`dtwin/medgemma_benchmark.py`](../../dtwin/medgemma_benchmark.py) já aceita
`--medgemma-config <arquivo>`. Apontar para `configs/medgemma_local_4b_rag.yaml`
**já seleciona o modo RAG**. Zero mudança de CLI.

### 2.3 UI (frontend)

O toggle de cenário recém-criado ([`webapp/static/benchmark.html`](../../webapp/static/benchmark.html))
é uma lista de botões `data-scenario`. Adicionar um terceiro botão
`data-scenario="rag"` e uma entrada em `SCENARIO_LABEL`. **Trade-off:** hoje o
layout é `grid-template-columns: 1fr 1fr` (2 colunas) em
[`webapp/static/argos.css`](../../webapp/static/argos.css) — com 3 opções vira
`repeat(3, 1fr)` ou empilha no mobile. Ajuste puramente cosmético.

## 3. Onde o RAG roda dentro do screening

Ponto de inserção: `run_screening`
([`dtwin/medgemma_screening.py:168-342`](../../dtwin/medgemma_screening.py)),
**entre** montar o prompt (`build_medgemma_prompt`,
[`:282`](../../dtwin/medgemma_screening.py)) e chamar o cliente
(`medgemma_client.generate`, [`:289`](../../dtwin/medgemma_screening.py)).

Fluxo proposto (topologia B mínima da Fase 1 — retrieve-then-verify):

```
# ilustrativo — não é implementação
base_prompt = build_medgemma_prompt(config)          # já existe, valida salvaguardas
if rag_enabled(config):
    draft   = client.generate(panel, base_prompt)    # 1ª passagem (rascunho)
    query   = build_query(config, draft)             # sinais + consulta-base fixa
    ctx     = retriever.search(query)                # BM25+denso → RRF → rerank → top-n
    if not ctx.acceptable(config):                   # score < min_score ⇒ abster
        raise AbstainInsufficientEvidence(...)        # vira INCONCLUSIVA/abort (§5)
    prompt  = inject_context(base_prompt, ctx)       # bloco [S1]..[Sn] + instrução de citar
    report  = client.generate(panel, prompt)         # 2ª passagem (final, grounded)
    envelope_extra = {"rag_context": ctx.provenance, "rag_citations": ...}
else:
    report  = client.generate(panel, base_prompt)    # caminho atual, intacto
```

**Justificativa:** o retriever é uma **camada de orientação** (monta o prompt),
não de enforcement — o gate clínico (`validate_medgemma_report`) permanece o mesmo
e continua sendo a autoridade final. **Trade-off:** o `run_screening` ganha um
ramo condicional; manter o RAG isolado em um módulo próprio (`dtwin/rag/…`, Fase 4)
evita inchar o orquestrador.

**Contrato do gateway:** **inalterado**. O RAG só muda o **texto** do `prompt`
enviado ([`medgemma_client.py:508`](../../dtwin/medgemma_client.py)); a imagem e o
schema seguem iguais. Nenhuma mudança em `tools/medgemma_server.py`.

## 4. Config via YAML (regras em config, nunca em código)

Novo bloco `rag:` dentro de `medgemma_screening`, e um arquivo que herda o baseline
e o liga. Esboço (**ilustrativo**):

```yaml
# configs/medgemma_local_4b_rag.yaml
extends: medgemma_local_4b.yaml
medgemma_screening:
  rag:
    enabled: true
    topology: retrieve_then_verify      # ou: single_pass_fixed_query
    knowledge_base:
      index_dir: "rag/index/liver_en_v1"   # fora de casos/; versionado à parte
      corpus_version: "liver_en_v1"
    embedding:
      provider: medcpt                  # alvo de qualidade biomédico
      query_model: "ncbi/MedCPT-Query-Encoder"
      document_model: "ncbi/MedCPT-Article-Encoder"
      max_chunk_tokens: 480             # evita truncamento no limite de 512 tokens
      fallback_provider: ollama          # baseline técnico/fallback medido
      fallback_model: "nomic-embed-text"
    retrieval:
      top_k_dense: 20
      top_k_sparse: 20
      rrf_k: 60
      candidates: 40
      rerank_model: "ncbi/MedCPT-Cross-Encoder"
      fallback_rerank_model: "BAAI/bge-reranker-v2-m3"
      rerank_top_n: 5
      min_score: 0.30                   # abaixo disto ⇒ abstenção
    grounding:
      cite_sources: true
      require_citation_exists: true     # citar [S#] inexistente ⇒ rejeita (fail-closed)
    evaluation:
      thresholds:
        faithfulness: 0.85
        answer_relevancy: 0.80
        context_precision: 0.70
    abstain:
      on_empty_retrieval: true
      map_to_state: INCONCLUSIVA        # não inventa estado novo
```

Isto exige **estender `_validate_config`**
([`medgemma_client.py:153`](../../dtwin/medgemma_client.py)) com uma validação do
bloco `rag` **quando presente** (fail-closed: RAG ligado com `index_dir` ausente,
`rerank_top_n<=0`, `min_score` fora de [0,1], provider desconhecido → `PipelineError`).
**Justificativa:** mantém a promessa "regras em config" e a validação estrita que já
caracteriza o projeto. **Trade-off:** mais superfície de validação — mas concentrada
num único bloco opcional.

**Herança e prompt:** o `prompt.template` do baseline **já contém** as salvaguardas
exigidas ([`medgemma_client.py:266-272`](../../dtwin/medgemma_client.py)); o RAG
**acrescenta** o bloco de contexto em runtime, sem remover salvaguarda alguma. Se o
RAG precisar de instrução extra de citação, ela entra no `template` do config de RAG
(ainda no YAML), preservando os fragmentos obrigatórios.

## 5. Aplicação do "abortar em falha" ao RAG

Cada falha do RAG mapeia para um comportamento **fail-closed** já existente — nunca
um laudo grounded em evidência inexistente:

| Falha do RAG | Comportamento | Onde se encaixa |
|---|---|---|
| Índice ausente/corrompido | `PipelineError` no load | como `_validate_config`/`_ensure_ready` (aborta o caso) |
| Embedding/reranker indisponível | `PipelineError` | idem — sem fallback silencioso |
| Recuperação vazia ou `score < min_score` | **Abstenção** → INCONCLUSIVA | reusa estado válido do schema (§01) |
| Citação `[S#]` a fonte inexistente | Rejeita a resposta | novo validador, no espírito de `validate_medgemma_report` |
| Timeout da 1ª/2ª passagem | `TIMEOUT` | já tratado no benchmark ([`server.py:586-589`](../../webapp/server.py)) |
| Métricas de confiabilidade < piso | Modo marcado **não-confiável** | gate de RAG (Fase 2 §7) |

No **benchmark**, essas falhas já têm classificação: `classify_screening_failure`
([`runner.py:157-167`](../../dtwin/benchmark/runner.py)) e os estados
`FAILURE/TIMEOUT/INVALID_RESPONSE` ([`models.py:21-27`](../../dtwin/benchmark/models.py))
— o RAG **não precisa de estados novos**, só emitir mensagens que caiam nas classes
certas. **Justificativa:** o RAG **reforça** o princípio anti-fabricação em vez de
abrir exceção a ele. **Trade-off:** mais INCONCLUSIVAs/abortos que os modos atuais —
esperado e desejável quando a evidência não existe.

## 6. Auditabilidade (manter o padrão do repo)

O repo hasheia config, painéis e entradas (`effective_config_sha256`,
`sha256_of`, `run_manifest`). O RAG deve estender isso:

- Hash do **índice/corpus** (`corpus_version` + digest) no envelope e no
  `run_manifest`, para que um resultado de `rag` seja reproduzível e rastreável ao
  conhecimento exato usado.
- Proveniência da recuperação (`rag_context`: lista de `(doc_id, section, url,
  score)`) no **envelope externo** (§01 §7, Opção 1) — não no `report` clínico.

**Justificativa:** um laudo grounded só é auditável se soubermos **em qual
conhecimento** ele se apoiou; sem o hash do corpus, "grounded" é inverificável.
**Trade-off:** exige tratar o índice como artefato versionado de primeira classe
(Fase 4), não um cache descartável.

## 7. Resumo das mudanças (superfície mínima)

| Camada | Mudança | Novo vs alterado |
|---|---|---|
| `configs/medgemma_local_4b_rag.yaml` | Config do modo (bloco `rag:`) | **Novo** |
| `dtwin/rag/…` | Retriever/index/embedding/rerank/citação | **Novo** (módulo isolado) |
| `dtwin/medgemma_screening.py` | Ramo condicional RAG em `run_screening` | Alterado (mínimo) |
| `dtwin/medgemma_client.py` | Validação do bloco `rag` em `_validate_config` | Alterado (aditivo) |
| `webapp/server.py` | +1 entrada em `BENCHMARK_SCENARIOS` | Alterado (1 linha lógica) |
| `webapp/static/benchmark.html` + `argos.css` | 3º botão de cenário | Alterado (cosmético) |
| Avaliação RAG (RAGAS/DeepEval) | Script separado do gate clínico | **Novo** (não toca `metrics.py`) |

Detalhe de arquivos, deps e requisitos: [`04-entregaveis-e-requisitos.md`](04-entregaveis-e-requisitos.md).
