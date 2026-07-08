# Modo RAG para o ARGOS — Plano técnico (SOMENTE PLANEJAMENTO)

> Este diretório contém **documentos de planejamento**. Nenhuma linha de código
> funcional foi criada; nenhum `.py` foi alterado; nenhuma dependência foi
> instalada. O objetivo é permitir ao Felipe **decidir** antes de autorizar a
> implementação de um terceiro modo de operação — `rag` — ao lado de `baseline` e
> `volumétrico`.

## Os documentos

| # | Documento | O que responde |
|---|---|---|
| 00 | [`00-reconhecimento.md`](00-reconhecimento.md) | Como o repo define/despacha "modo" hoje (com evidência arquivo+linha), como o benchmark seleciona modo, como o MedGemma é chamado, schema dos configs e onde "abortar em falha" é aplicado. |
| 01 | [`01-arquitetura.md`](01-arquitetura.md) | Arquitetura RAG local/offline em inglês/Apple Silicon: fontes, chunking, embeddings, vector store, recuperação híbrida + reranking, grounding/abstenção, saída estruturada. |
| 02 | [`02-metricas-e-avaliacao.md`](02-metricas-e-avaliacao.md) | Como **provar** o ganho em cold-start: reference-free primeiro, golden set sintético, RAGAS/DeepEval local, comparação com baseline/volumétrico, thresholds — e o que fica fora sem validação clínica. |
| 03 | [`03-integracao.md`](03-integracao.md) | Como registrar/despachar `rag` nos mecanismos existentes, config via YAML (bloco `rag:`), e aplicação do "abortar em falha". |
| 04 | [`04-entregaveis-e-requisitos.md`](04-entregaveis-e-requisitos.md) | Construção do corpus e do golden set do zero, arquivos a criar/modificar, dependências (checagem Apple Silicon), hardware, riscos, rollout. |
| 05 | [`05-perguntas-abertas.md`](05-perguntas-abertas.md) | As 7 decisões destrinchadas: o que se decide, por que importa, opções, recomendação e **a forma concreta de responder**. |
| corpus | [`corpus_manifest_v1.yaml`](corpus_manifest_v1.yaml) | Manifesto auditável do corpus candidato v1: 41 fontes, categorias, prioridade, cobertura e status de licença/ingestão. |

## Estado atual das decisões

O achado que rege tudo: **o ARGOS não tem registro de modos** — o modo **é um
arquivo de config** (`baseline` e `volumétrico` diferem por um único campo,
`panel.strategy`), e a UI mapeia `scenario → config` por um dict. Portanto o RAG
entra **sem inventar mecanismo novo**: um **eixo ortogonal** ativado por um bloco
`rag:` no YAML e um arquivo `configs/medgemma_local_4b_rag.yaml`, mais uma entrada
em `BENCHMARK_SCENARIOS`. O gate clínico (`validate_medgemma_report`, `metrics.py`)
permanece **intocado** como autoridade final.

As sete respostas do Felipe já foram incorporadas como direção de trabalho:

| Decisão | Estado consolidado |
|---|---|
| Corpus | **Opção B sem EASL**: StatPearls, anatomia de Couinaud, PMC Open Access e resumos autorais/seguros de LI-RADS/AASLD quando possível. |
| Validação dos labels | Sem especialista no momento, mas a equipe controla os exames e sabe se são saudáveis/doentes; portanto o label binário positivo/negativo é utilizável para sensibilidade/especificidade. |
| Validação clínica | Não há validação formal agora. O modo permanece `RESEARCH` e não deve prometer confiabilidade clínica. |
| Embedding/reranker | **MedCPT é o alvo de qualidade** por casar melhor o jargão biomédico; `nomic`/`bge` ficam como alternativa simples de comparação. |
| Topologia | **retrieve-then-verify** com duas passagens e abstenção baseada em evidência. |
| Saída do RAG | Citações/proveniência no **envelope externo**, sem alterar o schema clínico validado. |
| Thresholds | Pisos provisórios (`faithfulness ≥ 0,85`, `answer_relevancy ≥ 0,80`, `context_precision ≥ 0,70`) e calibração após o primeiro run. |

**Arquitetura MVP atualizada:** retrieve-then-verify (2 passagens), corpus pequeno e
curado, vector store **Chroma**, BM25 (`bm25s`), fusão **RRF**, reranker/embedding
biomédicos **MedCPT** como alvo de confiança, chunking adaptativo por seção com teto
compatível com MedCPT (~480 tokens úteis), e **abstenção → INCONCLUSIVA** (reusando
o estado já validado). Tudo deve rodar em **MPS/CPU** — nenhuma dependência
CUDA-only. O `nomic-embed-text` via Ollama e `bge-reranker-v2-m3` permanecem como
baseline técnico simples para comparação, não como decisão clínica.

**Prova de valor (a parte difícil do cold-start):** a **acertividade** é provada
pelas métricas clínicas **já existentes** (sensibilidade/especificidade/F1 vs
labels, mesmo dataset nos três modos, com IC de Wilson); a **confiabilidade/
grounding** é provada por métricas **reference-free** (faithfulness, answer/context
relevancy) via RAGAS com juiz **local**. Sem validação por especialista, isso prova
**consistência e grounding — não confiabilidade clínica** (lacuna declarada).

Ponto metodológico importante: o label binário positivo/negativo permite medir se o
RAG ajuda a classificar o exame, mas **não valida** localização segmentar, sinais
radiológicos finos ou se o modelo acertou "pelo motivo certo". Esses itens continuam
como lacuna até revisão especializada.

## Insumos que ainda faltam antes de implementar

As decisões de arquitetura estão encaminhadas, mas ainda faltam dois insumos
concretos:

1. Preparar a ingestão das 41 fontes aprovadas no
   [`corpus_manifest_v1.yaml`](corpus_manifest_v1.yaml): baixar/normalizar texto,
   gerar chunks, preservar metadados e construir o índice.
2. O conjunto rotulado de RMs saudáveis/doentes para rodar `baseline`,
   `volumetric` e `rag` lado a lado.

## Definition of Done deste plano

✔ Entender exatamente quais **arquivos e requisitos** são necessários (doc 03/04).
✔ Saber **quais métricas provam** o ganho e **o que fica fora** sem validação
clínica (doc 02). ✔ Registrar as **decisões consolidadas** acima. ✔ **Nenhuma linha
de código funcional** — só planejamento. A implementação começa **após** autorização
explícita.
