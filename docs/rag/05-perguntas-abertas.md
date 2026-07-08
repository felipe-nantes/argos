# 05 · Decisões respondidas sobre o RAG

> Planejamento. Este documento registrava as 7 perguntas abertas do modo RAG. As
> respostas do Felipe já foram absorvidas abaixo como **decisões consolidadas**.
> O documento mantém as opções originais por auditabilidade, mas a linha "Resposta
> atual" é a que orienta qualquer implementação futura.

Resumo rápido:

| # | Resposta atual | Implicação |
|---|---|---|
| Q1 | Opção B sem EASL | Corpus público/redistribuível + resumos autorais seguros. |
| Q2 | Sem especialista; labels binários controlados | Sens/esp são mensuráveis; raciocínio fino não é validado. |
| Q3 | Sem validação clínica formal | Modo fica `RESEARCH`. |
| Q4 | Priorizar a opção mais confiável: MedCPT | MedCPT vira alvo de qualidade, com fallback medido. |
| Q5 | Retrieve-then-verify | Duas passagens e abstenção por evidência. |
| Q6 | Envelope externo | Schema clínico validado permanece intocado. |
| Q7 | Thresholds provisórios + calibração | Pisos iniciais em config, ajustados após o primeiro run. |

---

## Q1 — Fontes do corpus de conhecimento (inglês) 🔴

**Resposta atual:** **Opção B sem EASL.** O corpus v1 deve usar StatPearls,
anatomia de Couinaud, PMC Open Access e resumos autorais/seguros de LI-RADS/AASLD
quando possível. EASL fica fora por falta de acesso/segurança de licenciamento.

**O que se decide:** exatamente **quais documentos em inglês** entram no corpus v1
que o retriever vai indexar.

**Por que importa (impacto máximo):** o corpus é a **maior alavanca de
acertividade** de todo o RAG. Corpus preciso e enxuto → o modelo recebe evidência
pertinente e a saída melhora. Corpus ruidoso ou irrelevante → o RAG **injeta ruído
e pode piorar** o baseline. Também define o **risco de licenciamento** (o que pode
ir para o repo).

**Opções:**

| Opção | Conteúdo | Prós | Contras |
|---|---|---|---|
| **A) Núcleo curado (recomendada)** | StatPearls (fígado) + anatomia de Couinaud + ~poucas dezenas de revisões PMC Open Access sobre lesão focal hepática em RM | Alta precisão, licença redistribuível, rastreável | Cobertura menor → mais abstenções |
| B) Núcleo + diretrizes | A + resumos **autorais** de LI-RADS/EASL/AASLD | Mais cobertura de critérios | Diretrizes têm copyright restritivo — só paráfrase própria, nunca texto integral |
| C) PMC OA em massa (por query) | Centenas de artigos puxados automaticamente | Cobertura ampla | Ruído alto → precisão@n cai → **risco de piorar o modelo** |

**Recomendação atualizada:** executar **B com controle de risco**, não B amplo. A
base deve priorizar material redistribuível/publicado; resumos de diretriz entram
como complemento pequeno, autoral e citado, idealmente com sanity-check do médico de
outra especialidade.

**Insumo ainda pendente:** lista concreta de fontes aprovadas (15–30
títulos/URLs/licenças). A decisão do tipo de corpus está fechada; falta o conteúdo.

**Impacto da resposta:** define o volume de curadoria (Fase 4 §1), o tamanho do
índice e a **taxa de abstenção esperada** (corpus pequeno → mais INCONCLUSIVA, o
que é seguro, não um bug).

---

## Q2 — Dono e critério de validação do golden set 🔴

**Resposta atual:** não há especialista para validar o golden set textual agora. A
equipe controla as entradas dos exames e sabe se cada RM é saudável ou doente; logo
o **label binário** é confiável para medir sensibilidade/especificidade.

**Limite da resposta:** isso valida o acerto positivo/negativo, mas não valida se o
modelo acertou a localização, os sinais radiológicos ou o raciocínio clínico.

**O que se decide:** **quem** revisa os ~100 QAs sintéticos e o que significa
"aprovado".

**Por que importa:** sem revisão humana, o golden set mede **o gerador contra si
mesmo** (circular) — e as métricas com-referência (context recall, factual
correctness) ficam sem valor. É o que separa "número bonito" de "número confiável".

**Opções de dono:**

| Opção | Quem revisa | Prós | Contras |
|---|---|---|---|
| a) Clínico revisa tudo | Radiologista/cirurgião nos 100 | Máxima confiança | Caro em tempo do clínico |
| **b) ML cura + clínico amostra (recomendada)** | ML limpa os 100; clínico valida ~30 estratificados | Barato e confiável onde importa | Exige um pouco de tempo do clínico |
| c) Só ML/Felipe | Sem clínico | Rápido | Fraco em correção clínica |

**Critério de "aprovado" (proposto):** um QA passa se (1) a pergunta é
**respondível a partir do chunk citado**, (2) a resposta é **clinicamente
correta**, (3) **não há vazamento** (a resposta não está copiada trivialmente da
pergunta). Registrar revisor + data (auditabilidade, como o resto do repo).

**O que passa a valer:** o golden set de QAs pode ser usado para desenvolvimento e
grounding, mas deve ser apresentado como não-clinicamente-validado até revisão
especializada. A prova principal de acurácia vem do benchmark binário dos exames.

**Impacto da resposta:** destrava as métricas com-referência da Fase 2. Sem dono, o
golden set fica preso em "rascunho" e só as métricas reference-free valem.

---

## Q3 — Existe/haverá validação clínica? 🔴

**Resposta atual:** ainda não há validação clínica formal. Opção A por enquanto.
O modo `rag` permanece `RESEARCH`.

**O que se decide:** se há um radiologista/cirurgião que **valide as saídas do
modelo** (não os QAs — as predições em exames reais), e quando.

**Por que importa (define o teto das promessas):** sem isso, o máximo honesto que
podemos afirmar é *"mais consistente e mais grounded que os modos atuais"* — **não**
*"clinicamente confiável"*. Também mantém a postura regulatória (`RESEARCH`).

**Opções:**

| Opção | Cenário | O que permite afirmar |
|---|---|---|
| A) Sem validador agora | Só métricas automáticas | "Consistente + grounded em dados de pesquisa" |
| **B) Spot-check único (recomendado)** | Clínico revisa ~20 casos estratificados (10 onde RAG mudou vs baseline, 5 abstenções, 5 concordâncias) | Acima + "amostra clínica humana concorda em X%" |
| C) Validação contínua | Clínico no loop | Afirmações mais fortes, mas compromisso grande |

**Recomendação atualizada:** manter o spot-check como evolução futura. Até lá, a
lacuna é declarada e o modo permanece `RESEARCH`.

**O que passa a valer:** sem validação clínica, não prometer "clinicamente
confiável"; prometer apenas consistência, grounding e acurácia experimental em
labels conhecidos.

**Impacto da resposta:** muda **o texto das conclusões** do relatório final e o que
podemos prometer ao HU. Não muda o código.

---

## Q4 — Embedding e reranker finais 🟡

**Resposta atual:** escolher "a melhor, a que dá mais confiança". Isso aponta para
**MedCPT** como alvo de qualidade, por ser biomédico.

**O que se decide:** o par embedding + reranker default, e se padronizamos o
biomédico **MedCPT** no Mac.

**Por que importa:** qualidade de recuperação vs simplicidade operacional. Um
embedding biomédico pode casar melhor o jargão clínico (recall maior), ao custo de
mais modelos para gerenciar (MedCPT usa **dois** encoders, query≠doc).

**Opções:**

| Opção | Embedding | Reranker | Perfil |
|---|---|---|---|
| A) Simples | nomic-embed-text (via Ollama) | bge-reranker-v2-m3 | 1 caminho de serving, roda igual no Win/Mac |
| B) Biomédico | MedCPT article/query encoder | MedCPT cross-encoder | Melhor jargão, mais complexo |
| **C) Deixe o dado decidir (recomendada)** | nomic default | bge-reranker | + testar MedCPT via recall@k/MRR (Fase 2 §1) antes de promover |

**Recomendação atualizada:** implementar a trilha MedCPT como alvo de qualidade,
mas medir contra a alternativa simples (`nomic`/`bge`) no F0. Se MedCPT perder no
corpus real, a alternativa simples vira fallback operacional.

**Consequências técnicas:** MedCPT usa encoders diferentes para query/documento e
tem limite prático de 512 tokens. O chunking deve usar teto de ~480 tokens úteis.

**Impacto da resposta:** quase nulo no design (é config: `rag.embedding.model` /
`rag.retrieval.rerank_model`); só muda qual número olhamos para escolher.

---

## Q5 — Topologia: passagem única vs retrieve-then-verify 🟡

**Resposta atual:** **B) retrieve-then-verify**.

**O que se decide:** **1 chamada** ao LLM (injeta contexto de uma consulta fixa) vs
**2 chamadas** (rascunho → recupera → verifica).

**Por que importa:** é o trade-off central **custo × confiabilidade**. Duas
passagens **dobram** a latência por exame — relevante no Mac com o 27B via Ollama —
mas são o que permite **adaptar a recuperação ao caso** e **abster-se** quando a
evidência contradiz o rascunho (justamente o "aumentar confiabilidade" que a tarefa
pediu).

**Opções:**

| Opção | Chamadas | Adapta ao caso? | Abstenção por evidência? | Custo/latência |
|---|---|---|---|---|
| A) Passagem única, consulta fixa | 1 | Não | Não | ~1× (baixo) |
| **B) Retrieve-then-verify (recomendada p/ o objetivo)** | 2 | Sim | Sim | ~2× |

Aritmética de latência: se um exame leva ~T s no 27B, **B ≈ 2T**. Num benchmark de
N exames, isso é 2× o tempo total de inferência.

**Recomendação atualizada:** usar B no v1, mas inicialmente sobre o painel
`baseline` para evitar custo `2 × N painéis`. `rag + volumetric` fica como evolução.

**O que passa a valer:** duas passagens, consulta derivada do rascunho + consulta
fixa, e abstenção quando a recuperação for fraca ou contraditória.

**Impacto da resposta:** muda **só o orquestrador** (Fase 3 §3); todos os
componentes (índice, retriever, rerank) servem às duas topologias. É trocável por
config (`rag.topology`).

---

## Q6 — Onde vive a saída do RAG (citações/proveniência) 🟢

**Resposta atual:** **Opção 1 — envelope externo**.

**O que se decide:** citações e proveniência no **envelope externo** (Opção 1) vs
**estender o schema clínico validado** (Opção 2).

**Por que importa:** o `report` clínico tem um schema **fechado e validado por
segurança** (`REQUIRED_REPORT_FIELDS` + `validate_medgemma_report`,
[`dtwin/medgemma_client.py:365`](../../dtwin/medgemma_client.py)). Mexer nele é
mexer em código de segurança e afeta **todos** os modos.

**Opções:**

| Opção | Onde entram citações | Prós | Contras |
|---|---|---|---|
| **1) Envelope externo (recomendada)** | `rag_context`/`rag_citations` no envelope do screening ([`medgemma_screening.py:30`](../../dtwin/medgemma_screening.py)) | Não toca o gate clínico; risco mínimo; auditável ao lado do laudo | Modelo não é *forçado* a citar dentro do JSON |
| 2) Estender o schema | Campo `citacoes` obrigatório no `report` | Força citação no JSON | Altera código validado; afeta baseline/volumétrico ou vira validação condicional (complexidade) |

**Recomendação:** **Opção 1** para o MVP. Baixíssimo risco; a citação por-*claim*
dentro do JSON vira trilha de evolução (Fase 1 §9) se provar necessária.

**O que passa a valer:** `report` clínico continua com o schema validado. Campos
como `rag_context`, `rag_citations`, `retrieval_metrics` ficam no envelope.

**Impacto da resposta:** define se tocamos ou não o gate clínico (Opção 1 = não
tocamos, muito mais seguro).

---

## Q7 — Thresholds de confiabilidade do RAG 🟢

**Resposta atual:** usar thresholds provisórios e calibrar.

**O que se decide:** os pisos das métricas reference-free abaixo dos quais o modo
RAG é declarado **não-confiável** (e a resposta abstém/aborta).

**Por que importa:** é o gate que impede um RAG mal-grounded de virar laudo
silencioso. **Alto demais** → tudo abstém (cobertura zero). **Baixo demais** → não
protege.

**Valores de partida propostos (não são finais):**

| Métrica | Piso inicial | Papel |
|---|---|---|
| faithfulness | ≥ 0,85 | Resposta ancorada no contexto recuperado |
| answer_relevancy | ≥ 0,80 | Resposta on-topic para a triagem |
| context_precision | ≥ 0,70 | Recuperado pertinente |

**Recomendação:** adotar como **provisórios** e **calibrar** após o primeiro run do
golden set — só congelar quando virmos a **distribuição real** das métricas. Ficam
em config (`rag.evaluation.thresholds`), nunca hard-coded.

**O que passa a valer:** pisos iniciais em config, sem hard-code:
`faithfulness ≥ 0,85`, `answer_relevancy ≥ 0,80`, `context_precision ≥ 0,70`.

**Impacto da resposta:** governa a **taxa de abstenção** e o rótulo
"confiável/não-confiável" do run. Trocável por config a qualquer momento.

---

## Resumo final — decisões fechadas e pendências reais

| Item | Estado |
|---|---|
| Arquitetura | Fechada para planejamento: RAG por config, retrieve-then-verify, envelope externo. |
| Modelo de recuperação | MedCPT como alvo de qualidade, fallback leve medido. |
| Métrica principal | Sens/esp/F1 em labels binários conhecidos. |
| Métrica de grounding | RAGAS/reference-free + recall@k/MRR. |
| Limitação declarada | Sem especialista, não há validação clínica fina. |
| Insumo pendente 1 | Lista concreta do corpus. |
| Insumo pendente 2 | Dataset de RMs saudáveis/doentes para comparar cenários. |

Próximo passo quando houver autorização para implementar: **F0 — curar corpus,
construir índice e medir recuperação isolada** antes de integrar ao pipeline.
Índice geral: [`README.md`](README.md).
