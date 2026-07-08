# 01 · Arquitetura RAG (Fase 1)

> Planejamento. Pseudocódigo é ilustrativo e curto — não é implementação. Cada
> escolha traz **justificativa** (impacto em acertividade/confiabilidade) e
> **trade-off**. Premissas: corpus em **inglês/biomédico**; deploy em **Apple
> Silicon (MPS/CPU, sem CUDA)**; dev box com 8 GB de VRAM (mas o gargalo de
> memória é só o LLM gerador, já contabilizado).

## 0. Qual é o "shape" do RAG aqui (a decisão que rege tudo)

A tarefa do ARGOS é **classificação a partir de imagem** (POSITIVA/NEGATIVA/
INCONCLUSIVA sobre um painel PNG). RAG clássico é *pergunta textual → recuperação
→ resposta*. O descasamento é real e precisa de uma decisão explícita sobre **o
que é a consulta** (PA-1 do reconhecimento).

Duas topologias possíveis:

| | **A) Passagem única, consulta fixa** | **B) Duas passagens, consulta = rascunho** |
|---|---|---|
| Consulta | Derivada da tarefa+órgão (fixa): *"focal liver lesion signs on abdominal MRI, Couinaud segments"* | 1ª passagem MedGemma gera rascunho (sinais/resumo); esse texto vira a consulta |
| Chamadas ao LLM | 1 | 2 |
| Adapta ao caso? | Não (mesmo contexto p/ todo exame) | Sim (contexto segue o achado) |
| Custo/latência | Baixo | ~2× o gerador |
| Ganho de confiabilidade | Injeta vocabulário/critérios consistentes (padroniza a saída) | Além disso: permite **verificação/abstenção** do rascunho contra evidência |
| Risco | Contexto pode ser irrelevante para o caso | 1ª passagem pode "poluir" a consulta com alucinação |

**Decisão consolidada:** começar por **B com salvaguarda**, na forma mínima —
*retrieve-then-verify*: (1) MedGemma vê o painel e gera um rascunho estruturado;
(2) recuperamos evidência usando os `sinais_visuais_observados` + consulta-base
fixa; (3) MedGemma re-avalia o painel **com o contexto recuperado** e emite o JSON
final, podendo **abster-se** ("informação insuficiente") se a evidência não
sustentar o achado. **Justificativa:** é a topologia que efetivamente move
"confiabilidade" (grounding + abstenção), não só "consistência". **Trade-off:**
dobra o custo do gerador e adiciona um ponto de falha (a 1ª passagem). Por isso a
**consulta-base fixa** sempre entra junto, ancorando a recuperação mesmo quando o
rascunho é fraco.

> **Nota operacional:** no v1, compor o RAG preferencialmente com o painel
> `baseline` (`uniform_9`) para evitar custo `2 × N painéis` quando o cenário
> volumétrico gerar muitos painéis. A combinação `rag + volumetric` pode continuar
> como evolução posterior.

---

## 1. Fontes de conhecimento (inglês)

**Objetivo:** um corpus pequeno, curado e **rastreável** — não um dump massivo. Em
domínio clínico, *precisão da fonte > cobertura*.

Candidatos (todos em inglês, com licença utilizável para pesquisa):

- **Couinaud / anatomia segmentar hepática** — base para `localizacao_aproximada`.
  Fontes abertas: Radiopaedia (checar licença por artigo — CC BY-NC-SA em muitos),
  StatPearls (NCBI Bookshelf, domínio público-friendly), capítulos de anatomia
  abertos.
- **Sinais de lesão focal hepática em RM** — hemangioma, HNF, adenoma, HCC,
  metástase, cisto: aparência por sequência (T1/T2/DWI/fases de contraste). Fontes:
  StatPearls, artigos *open-access* (PMC OA subset), diretrizes.
- **Diretrizes / protocolos** — LI-RADS (ACR) e AASLD quando disponíveis como
  resumos autorais/seguros. **EASL fica fora do v1** por falta de acesso/segurança
  de licenciamento.
  **Atenção a licenciamento**: muitas diretrizes têm copyright restritivo; usar
  **sumários/parafrase próprios** ou trechos sob *fair use* de pesquisa, com
  citação — nunca republicar o texto integral no repo.

**Requisitos de curadoria/licenciamento (bloqueantes de ingestão):**

- Cada documento entra com metadados obrigatórios: `source`, `license`, `url`,
  `retrieved_at`, `section`. Sem esses campos → não indexa (fail-closed).
- Preferir **PMC Open Access subset** e **StatPearls** para texto integral
  redistribuível; para material restrito, indexar **apenas resumos autorais**.
- Formato de ingestão: **Markdown/JSON limpos** (um arquivo por documento/seção),
  versionados fora de `casos/` (nunca dado de paciente). Ver Fase 4.

> **Decisão consolidada:** Opção B sem EASL. Ainda falta como insumo a lista exata
> de documentos/fontes (títulos, URLs e licenças), mas o tipo de corpus está
> definido.

**Justificativa:** corpus pequeno e curado maximiza *context precision* (menos
ruído recuperado) e mantém rastreabilidade — condição para o princípio "não
fabricar". **Trade-off:** menor cobertura; achados fora do corpus levarão à
abstenção (aceitável e desejável aqui).

---

## 2. Chunking

| Estratégia | Prós | Contras | Adequação clínica |
|---|---|---|---|
| Fixed-length (ex. 512 tok, overlap 64) | Simples, previsível | Corta conceito no meio | Baixa |
| **Semantic** (quebra por similaridade de sentenças) | Preserva unidade conceitual | Custo de emb*​edding no pré-processo | **Boa** |
| Proposition (1 fato atômico/chunk, via LLM) | Máxima precisão de citação | Caro; exige LLM na ingestão; fragmenta contexto | Média |
| Adaptativo (por estrutura do doc: seção/heading) | Respeita a organização médica (por lesão, por sinal) | Depende de docs bem estruturados | **Boa** |

**Decisão consolidada:** **adaptativo por estrutura** (heading/seção) **+ teto de
tamanho compatível com MedCPT** — cada chunk é uma seção clínica coerente (ex.
"Hemangioma — MRI appearance"), quebrada só se exceder ~480 tokens úteis.
**Justificativa adicional:** MedCPT tem limite prático de 512 tokens; o teto menor
evita truncamento silencioso após metadados, prefixos de consulta e marcadores de
citação.
**Justificativa:** integridade conceitual (o modelo recebe o sinal inteiro, não
meia-frase) e rastreabilidade (chunk = seção citável). **Trade-off:** exige que a
curadoria produza docs com headings limpos (empurra trabalho para a ingestão, onde
ele é barato e auditável). **Evolução:** proposition chunking para as seções de
maior valor, se a citação por *claim* provar ser necessária.

**Metadados por chunk (obrigatórios):** `doc_id`, `section`, `source`, `license`,
`url`, `chunk_id`, `token_count`. São o que permite citação e filtragem.

---

## 3. Embeddings (inglês/biomédico, servível local/MPS)

| Modelo | Tipo | Dim | Nota em Apple Silicon | Observação |
|---|---|---|---|---|
| **BGE-large-en-v1.5** | Geral EN, forte em retrieval | 1024 | Roda em MPS/CPU (sentence-transformers) | Baseline robusto; exige prefixo de *query* |
| **nomic-embed-text-v1.5** | Geral EN, long-context | 768 | MPS/CPU; **também via Ollama** (`nomic-embed-text`) | Ótimo custo/latência; Matryoshka (dim ajustável) |
| **mxbai-embed-large** | Geral EN | 1024 | MPS/CPU; via Ollama | Competitivo com BGE-large |
| **MedCPT Query/Article Encoder** | **Biomédico** (treinado em PubMed) | 768 | MPS/CPU (transformers) | Melhor casamento de *jargão* clínico; dois encoders (query≠doc) |
| **PubMedBERT-based (S-PubMedBert)** | Biomédico | 768 | MPS/CPU | Alternativa biomédica |

**Decisão consolidada:**

- **Alvo de qualidade:** **MedCPT** (query/article encoder), por casar melhor o
  vocabulário biomédico e radiológico.
- **Fallback/baseline técnico:** `nomic-embed-text-v1.5` via Ollama, por ser simples,
  rápido, local e barato no dev box.
- **Critério de confiança:** o primeiro build do índice deve medir MedCPT contra a
  trilha leve em recall@k/MRR. Se MedCPT perder neste corpus, `nomic` permanece
  como escolha operacional. Confiança vem do dado medido, não do rótulo
  "biomédico".

Consequências técnicas do MedCPT:

- usar dois encoders quando aplicável (`query` ≠ `article/document`);
- limitar chunks a ~480 tokens úteis;
- carregar embedding/reranker em CPU/MPS para não disputar VRAM com o LLM no PC de
  8 GB.

**Regra de config:** modelo, dimensão e prefixos de *query/passage* vivem no bloco
`rag.embedding` do YAML — nunca no código.

---

## 4. Vector store local

| Store | Apple Silicon | Metadados/filtragem | Híbrido (esparso+denso) | Nota |
|---|---|---|---|---|
| **Chroma** | Sim (CPU, pip puro) | Sim | Denso nativo; esparso externo | Mais simples de embutir; persiste em disco |
| **Qdrant** | Sim (binário ARM / Docker) | Sim, rico | **Sim, nativo (denso+esparso)** | Mais capaz; serviço à parte |
| **FAISS** | Sim (CPU/`faiss-cpu`) | Não nativo (só vetores) | Só denso | Rápido, mas você gerencia metadados/BM25 à parte |

**Recomendação MVP:** **Chroma**, persistente em disco, embutido no processo.
**Justificativa:** corpus pequeno (centenas–poucos milhares de chunks) não precisa
de um serviço dedicado; Chroma dá metadados+filtragem e roda em CPU/ARM sem
fricção, coerente com "estação única, sem nuvem". **Trade-off:** o esparso (BM25)
fica **fora** do store (índice BM25 separado, §5). **Evolução:** migrar para
**Qdrant** se/quando o corpus crescer e o híbrido nativo compensar o serviço
extra.

> BM25 puro-Python: **`bm25s`** (rápido, numpy) ou `rank-bm25`. Ambos CPU, sem
> CUDA, triviais em Apple Silicon.

---

## 5. Recuperação HÍBRIDA + reranking (a maior alavanca de precisão)

Pipeline de recuperação recomendado (todos os passos CPU/MPS):

```
consulta ──┬─► BM25 (esparso)         top-k_sparse ─┐
           └─► denso (embeddings)     top-k_dense  ─┤─► fusão RRF ─► candidatos (N)
                                                     ┘
candidatos (N) ─► cross-encoder reranker ─► top-n (n ≪ N) ─► contexto do prompt
```

**Fusão RRF** (Reciprocal Rank Fusion): combina os rankings sem precisar calibrar
escalas de score. **Justificativa:** BM25 pega termos raros exatos (nomes de
sinais, siglas — "HNF", "T2 hyperintense"); denso pega paráfrase semântica; RRF
capta os dois. **Trade-off:** dois índices para manter.

**Reranking por cross-encoder** — a etapa que mais melhora precisão@n:

| Reranker | Apple Silicon | Domínio | Nota |
|---|---|---|---|
| **bge-reranker-v2-m3** | MPS/CPU | Geral (multilíngue) | Forte e amplamente usado |
| **MedCPT Cross-Encoder** | MPS/CPU | **Biomédico** | Casa o par (query, artigo) clínico |
| **ms-marco MiniLM** | MPS/CPU (leve) | Geral | Menor/mais rápido; menor teto |

**Decisão consolidada:** testar **MedCPT cross-encoder** como alvo de qualidade e
manter **bge-reranker-v2-m3** como baseline técnico/fallback. **Justificativa:** o
reranker corta os falsos-positivos da fusão antes de eles entrarem no prompt — é o
passo que mais protege contra grounding em evidência irrelevante. **Trade-off:** +1
modelo e +latência por candidato (mitigado por `n` pequeno).

**Parâmetros iniciais (em `rag.retrieval` do YAML, ajustáveis por benchmark):**

- `top_k_dense: 20`, `top_k_sparse: 20`, `rrf_k: 60` (constante padrão do RRF).
- `candidates` (pós-fusão): 30–40.
- `rerank_top_n: 4–6` (o que efetivamente vai ao prompt).

**Justificativa dos números:** `n` pequeno mantém o prompt dentro de
`max_prompt_chars` (12000) e reduz a chance de o modelo se perder; `top_k`
generoso antes do rerank aumenta recall bruto (o rerank filtra depois).

---

## 6. Grounding e abstenção (coerência com "não fabricar")

- **Injeção de contexto:** o contexto recuperado entra no prompt como um bloco
  delimitado e **numerado**, cada trecho com um rótulo `[S1]`, `[S2]`… mapeado a
  `(doc_id, section, url)`. O prompt instrui: *"Baseie-se APENAS nas imagens e,
  quando citar conhecimento, referencie [S#]. Se a evidência recuperada não
  sustentar um achado, classifique INCONCLUSIVA."*
- **Citação por claim:** cada item de `sinais_visuais_observados` **pode** carregar
  um `[S#]`. Citação é **rastreável**, não decorativa: o `[S#]` precisa existir no
  contexto injetado — um validador (Fase 3) rejeita citação a fonte inexistente
  (fail-closed, no espírito do repo).
- **Política de abstenção:** mapear "informação insuficiente" ao estado
  **INCONCLUSIVA** já existente (não inventar um estado novo — o schema é fechado,
  §00). Gatilhos de abstenção: recuperação vazia/fraca (score do melhor candidato
  abaixo de `rag.retrieval.min_score`), ou contradição evidência×rascunho.

**Justificativa:** reaproveita o vocabulário e os gates já validados
(`INCONCLUSIVA`, `necessidade_de_revisao_humana`), então o RAG **fortalece** o
princípio anti-fabricação em vez de abrir uma brecha. **Trade-off:** mais
INCONCLUSIVAs (queda de cobertura) — mas isso é *desejável* quando a evidência não
existe, e as métricas (Fase 2) contam inconclusivo como não-decisão, não como
acerto forçado.

---

## 7. Saída estruturada e interação com o parsing atual

O parsing atual ([`medgemma_client.py:365-405`](../../dtwin/medgemma_client.py))
valida um **schema fechado de 7 campos** e descarta chaves extras
([`medgemma_client.py:374`](../../dtwin/medgemma_client.py)). Duas opções para os
metadados de RAG (citações, fontes, sinais de recuperação):

- **Opção 1 (recomendada) — envelope externo:** o `report` continua **idêntico**
  (7 campos); as citações e a proveniência do RAG entram no **envelope** do
  screening (`build_report_envelope`, [`medgemma_screening.py:30-76`](../../dtwin/medgemma_screening.py)),
  em chaves novas tipo `rag_context`, `rag_citations`, `retrieval_metrics`.
  **Justificativa:** não mexe no gate clínico já validado (risco mínimo); as
  citações ficam auditáveis ao lado do relatório. **Trade-off:** o modelo não é
  *forçado* a citar dentro do JSON — a citação por-claim vira pós-associação
  (rascunho `[S#]` → mapa no envelope).
- **Opção 2 — estender o schema:** adicionar `citacoes` a
  `REQUIRED_REPORT_FIELDS` e à validação. **Justificativa:** força o modelo a citar
  no próprio JSON. **Trade-off:** altera código validado e de segurança; todo modo
  (baseline/volumétrico) passaria a exigir o campo, ou a validação vira
  condicional por modo (complexidade). **Evitar no MVP.**

> **Decisão consolidada:** Opção 1. Citações/proveniência vivem no **envelope
> externo**; o `report` clínico validado permanece idêntico.

---

## 8. Recomendação MVP enxuta (o "mínimo confiável")

- Topologia **B mínima** (retrieve-then-verify), inicialmente sobre o painel
  `baseline` para controlar latência.
- Embedding/reranker **MedCPT** como alvo de qualidade; `nomic`/`bge` como baseline
  técnico e fallback. Vector store **Chroma**; BM25 **`bm25s`**; fusão **RRF**.
- Chunking **adaptativo por seção** (teto ~480 tok úteis).
- Grounding com `[S#]` + **abstenção → INCONCLUSIVA**.
- Saída de RAG no **envelope externo** (schema clínico intocado).
- Tudo parametrizado no bloco `rag:` do YAML (Fase 3).

## 9. Trilha de evolução

1. Integrar `rag + volumetric` se a latência do `rag + baseline` for aceitável e o
   ganho justificar.
2. **Qdrant** (híbrido nativo) se o corpus crescer.
3. **Proposition chunking** nas seções de maior valor para citação por-claim
   dentro do JSON (Opção 2 do §7).
4. Reranking com *listwise*/LLM-reranker local, se precisão@n estagnar.

Próximo: [`02-metricas-e-avaliacao.md`](02-metricas-e-avaliacao.md).
