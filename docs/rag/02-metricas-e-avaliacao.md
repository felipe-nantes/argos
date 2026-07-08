# 02 · Métricas e avaliação em cenário cold-start (Fase 2)

> Planejamento. Núcleo da tarefa: **provar** que o modo `rag` aumenta acertividade
> e confiabilidade — partindo do zero (sem golden set, sem validação clínica
> confirmada). Este documento separa com rigor **o que cada métrica prova** do
> **que ela não prova**.

## 0. O descasamento que molda toda a avaliação (ler primeiro)

Métricas RAG padrão (RAGAS/DeepEval) assumem *pergunta textual → contexto → resposta
textual*. O ARGOS classifica a partir de **imagem**; boa parte dos "claims" do
relatório é **visual** (lido do painel), e **não** estará no corpus textual. Logo:

- **Faithfulness / grounding** medem se o **raciocínio textual** do modelo está
  ancorado no conhecimento recuperado — **não** se o modelo **leu a imagem
  corretamente**.
- A prova de **acertividade** (ler a imagem certo) continua vindo das **métricas
  clínicas end-to-end já existentes** (sensibilidade/especificidade/F1 contra os
  labels), que o harness atual **já calcula**
  ([`dtwin/benchmark/metrics.py`](../../dtwin/benchmark/metrics.py)).

**Consequência prática (a espinha dorsal do plano de prova):**

| Pergunta | Métrica que responde | De onde vem |
|---|---|---|
| O RAG deixa o modelo **mais certeiro**? | Δ sensibilidade/especificidade/F1 `rag` vs `baseline`/`volumétrico` | Harness **existente** (mesmos labels) |
| O RAG deixa o modelo **mais confiável/grounded**? | faithfulness, answer/context relevancy (reference-free) | Framework RAG **novo** (Fase 4) |
| A **recuperação** é boa? | recall@k, precision@k, MRR, NDCG@k, hit rate | Golden set sintético **novo** |

Nenhuma dessas prova **confiabilidade clínica** — ver §6 (lacuna explícita).

**Decisão consolidada sobre labels:** como a equipe controla as entradas e sabe se
cada RM é de fígado saudável ou doente, o label binário positivo/negativo é
considerado utilizável para a comparação principal (`rag` vs `baseline` vs
`volumétrico`). Isso destrava sensibilidade/especificidade, mas **não** valida se a
localização segmentar, os sinais descritos ou o raciocínio causal estão
clinicamente corretos.

---

## 1. Métricas de recuperação (exigem golden set de pares query↔doc)

- **recall@k** — fração dos docs relevantes recuperados nos top-k. *Prova:* o
  retriever **encontra** a evidência. *Não prova:* que o modelo a usou.
- **precision@k** — fração dos top-k que é relevante. *Prova:* pouco ruído entra no
  prompt.
- **MRR** — posição do 1º relevante. *Prova:* o relevante vem **cedo** (importa com
  `n` pequeno pós-rerank).
- **NDCG@k** — qualidade do ranking ponderada por posição/grau.
- **hit rate** — houve ao menos 1 relevante nos top-k (piso).

**Uso no projeto:** medir o pipeline de recuperação (§5 da arquitetura)
**isoladamente** — antes de envolver o LLM — para tunar `top_k`, `rrf_k`,
`rerank_top_n` e escolher embedding/reranker (A/B da Fase 1). **Barato e
determinístico** (não usa LLM-juiz).

---

## 2. Métricas de geração (padrão RAGAS/DeepEval)

- **faithfulness** — todo claim da resposta é sustentado pelo contexto recuperado?
  *Prova:* ausência de alucinação **em relação ao texto recuperado**. *Não prova:*
  correção visual.
- **answer relevancy** — a resposta responde à "pergunta" (aqui: a tarefa de
  triagem)?
- **context precision** — os trechos recuperados **usados** eram os relevantes (o
  relevante veio no topo)?
- **context recall** — o contexto cobre o que a resposta-referência exigia?
  (**exige referência** → só após golden set com respostas).
- **factual correctness** — alinhamento factual com uma referência (**exige
  referência**).

---

## 3. Prioridade: **reference-free primeiro**

Sem respostas rotuladas, começamos pelas métricas que **não** exigem ground truth:

| Métrica | Precisa de referência? | O que prova | O que NÃO prova |
|---|---|---|---|
| **faithfulness** | Não | Resposta ancorada no contexto recuperado | Que a leitura da imagem está correta |
| **answer relevancy** | Não | Resposta on-topic para a triagem | Correção clínica |
| **context relevance** | Não | O recuperado é pertinente à consulta | Completude da evidência |

**Justificativa:** dão sinal **imediato** de grounding/consistência sem esperar a
curadoria de um golden set com respostas. **Trade-off:** silenciosas quanto a
acertividade — por isso andam **sempre** ao lado das métricas clínicas do harness
(§0). Métricas com-referência (context recall, factual correctness) entram na 2ª
onda, quando o golden set (§4) existir.

---

## 4. Golden set do zero (synthetic-first)

Sem dados rotulados de RAG, geramos um golden set **sintético a partir do próprio
corpus**, com **LLM-juiz LOCAL** (privacidade):

- **Ferramenta:** RAGAS `TestsetGenerator` (ou DeepEval `Synthesizer`) apontado
  para um **LLM local** (via Ollama — o mesmo daemon do deploy) e para o
  **embedding local**. Gera trios *(pergunta, contexto-fonte, resposta-referência)*
  a partir dos chunks.
- **Tamanho-alvo:** **~100 QAs** no v1 (cobrindo os principais sinais/lesões e a
  anatomia de Couinaud), com distribuição de dificuldade (simples/multi-hop/
  raciocínio).
- **Validação (estado atual):** sem especialista no momento. O golden set de QAs
  fica como instrumento de desenvolvimento/grounding, apoiado em corpus curado,
  métricas reference-free e sanity-check do médico de outra especialidade quando
  possível. Sem revisão especializada, ele **não** deve ser apresentado como prova
  de correção clínica fina.
- **Versionamento:** dataset em arquivo versionado (fora de `casos/`), com
  `dataset_version`, hash do corpus que o originou e do modelo gerador. Regenerar =
  nova versão, nunca sobrescrever (coerência com a auditabilidade do repo:
  `run_manifest`, hashes).

> **Decisão consolidada:** o benchmark clínico principal usa labels binários
> controlados pela equipe. O golden set textual do RAG permanece útil para medir
> recuperação/grounding, mas deve ser rotulado como não-clinicamente-validado até
> haver revisão especializada.

**Justificativa:** synthetic-first destrava as métricas com-referência sem esperar
anotação manual completa. **Trade-off:** QAs sintéticos herdam o viés do LLM
gerador — mitigado pela revisão humana amostral e por manter o gerador **local e
versionado**.

---

## 5. Framework de avaliação (local, Apple Silicon)

- **Escolha:** **RAGAS** como núcleo (métricas reference-free maduras) — ou
  **DeepEval**, se a integração de asserts em teste for preferível. Ambos aceitam
  **LLM-juiz e embeddings customizados**, então apontamos para **Ollama local**
  (juiz) + embedding local — **nada vai para a nuvem**.
- **Execução:** roda em CPU/MPS (o juiz é o próprio Ollama/Metal no Mac; no dev box,
  CPU). Sem CUDA.
- **Integração no repo:** um script/módulo de avaliação **separado** do gate
  clínico (`metrics.py` permanece puro e sem dependência de LLM — ver
  [`metrics.py:1`](../../dtwin/benchmark/metrics.py) "sem dependência do webapp ou
  do modelo"). As métricas de RAG são **anexadas ao relatório**, não misturadas ao
  cálculo clínico.

> **Decisão de planejamento:** usar **RAGAS** no MVP (reference-free consolidado);
> DeepEval permanece como alternativa futura se a equipe quiser asserts integrados
> ao `pytest`.

---

## 6. Comparabilidade `rag` × `baseline` × `volumétrico`

O harness já roda datasets rotulados por modo e produz o mesmo bloco de métricas
clínicas ([`metrics.py:56`](../../dtwin/benchmark/metrics.py)). O plano de
comparação:

1. **Mesmas entradas, três modos:** rodar o **mesmo dataset rotulado** em
   `baseline`, `volumétrico` e `rag` (o webapp já seleciona modo por `scenario`;
   Fase 3 adiciona `rag`).
2. **Métricas clínicas lado a lado:** montar uma **tabela de ganho/regressão** com
   Δ de sensibilidade, especificidade, F1, cobertura e taxa de inconclusivo, cada
   um com o **IC de Wilson** que o harness já calcula
   ([`metrics.py:139-146`](../../dtwin/benchmark/metrics.py)) — a sobreposição dos
   ICs diz se o ganho é real ou ruído no tamanho de amostra atual.
3. **Métricas de RAG (só no `rag`):** faithfulness/relevancy/recall@k anexadas,
   para explicar *por que* o `rag` mudou (grounding melhor? recuperação ruim?).

Esboço da tabela de saída (ilustrativo):

| Métrica | baseline | volumétrico | rag | Δ(rag−melhor) | ICs sobrepostos? |
|---|---|---|---|---|---|
| Sensibilidade | … | … | … | … | … |
| Especificidade | … | … | … | … | … |
| F1 | … | … | … | … | … |
| Inconclusivo % | … | … | … | … | — |
| faithfulness | — | — | … | — | — |

**Justificativa:** reaproveita o gate e os ICs já auditáveis; a prova de
acertividade é *apples-to-apples* (mesmos labels, mesmos exames). **Trade-off:**
requer um dataset rotulado com positivos **e** negativos suficientes para os ICs
não engolirem o efeito (ver §8).

---

## 7. Limiar de aceite (thresholds)

Dois níveis, ambos **em config** (`rag.evaluation.thresholds` — regras no YAML):

- **Gate clínico (já existe):** sens ≥ 0.75 **e** esp ≥ 0.75
  ([`metrics.py:168-177`](../../dtwin/benchmark/metrics.py)). O `rag` **não pode
  regредir** abaixo disso — e a meta é **superar** o melhor modo atual sem os ICs
  se sobreporem.
- **Gate de confiabilidade do RAG (novo, reference-free):** pisos mínimos por
  métrica, ex. `faithfulness ≥ 0.85`, `answer_relevancy ≥ 0.80`,
  `context_precision ≥ 0.70`. **Abaixo do piso → o modo RAG é declarado
  não-confiável**: o resultado é marcado como tal e, no runtime, a resposta
  **abstém** (INCONCLUSIVA) ou **aborta** — coerente com o projeto (§00/§03).

> **Decisão consolidada:** usar os valores acima como pisos provisórios e calibrar
> após o primeiro run real. Eles ficam em config, nunca hard-coded.

**Justificativa:** um número ruim de grounding não pode virar laudo silencioso —
o threshold transforma "RAG fraco" em abstenção auditável. **Trade-off:** pisos
altos demais derrubam cobertura; baixos demais não protegem. Daí calibrar.

---

## 8. Lacuna explícita e validação clínica (o que fica FORA do alcance)

**Sem validação por especialista clínico, estas métricas provam CONSISTÊNCIA e
GROUNDING — não "confiabilidade clínica".** Especificamente:

- faithfulness alto = coerente com o corpus; **não** = correto para o paciente.
- sensibilidade/especificidade altas num dataset **público/sintético** não
  transferem automaticamente para a população do HU.
- O corpus pode estar correto e **incompleto** — abstenção reduz risco, não o zera.

**Plano mínimo de spot-check humano (futuro, não disponível agora):**

- Um radiologista/cirurgião revisa uma **amostra estratificada** (ex. 20 casos:
  10 onde `rag` mudou a predição vs baseline, 5 abstenções, 5 concordâncias),
  julgando (a) a predição e (b) se as **citações** sustentam o texto.
- Métrica humana simples: *taxa de citação sustentada* e *concordância
  clínico×modelo* — reportadas **separadas**, jamais fundidas às métricas
  automáticas.
- Enquanto não houver especialista, o modo `rag` **não** deve sair de
  `regulatory_mode: RESEARCH`.

> **Decisão consolidada:** não há validação clínica formal agora. O teto honesto do
> que entregamos é "mais consistente e mais grounded que os modos atuais", com prova
> estatística de acertividade **em dados de pesquisa** — não "clinicamente
> confiável".

## 9. Dependência de dados (bloqueante de mensuração)

- Métricas de **recuperação** e **com-referência** dependem do **golden set** (§4).
- Métricas **clínicas comparativas** dependem de um **dataset rotulado com
  positivos e negativos** em volume suficiente para ICs úteis. A equipe pode montar
  esse conjunto porque controla as entradas e conhece o label binário; o repo ainda
  não deve versionar esses exames.

Próximo: [`03-integracao.md`](03-integracao.md).
