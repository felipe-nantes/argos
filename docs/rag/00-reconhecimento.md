# 00 · Reconhecimento do repositório (Fase 0)

> Documento de **planejamento**. Nenhuma linha de código funcional foi criada ou
> alterada. Toda afirmação abaixo está ancorada em arquivo + trecho real do repo,
> no commit atual. O que não pôde ser determinado pelo código foi atualizado com as
> decisões externas já respondidas pelo Felipe.

## Resumo executivo

O ARGOS **não tem um "registro de modos"** (enum/factory/dispatch central). O que
existe é um princípio arquitetural — *núcleo determinístico + regras em config
versionada* ([`contexto/04_ARQUITETURA.md`](../../contexto/04_ARQUITETURA.md)) —
aplicado de forma que **o "modo de operação" É um arquivo de configuração YAML**.
`baseline` e `volumétrico` diferem por **um único campo**: `panel.strategy`
(`uniform_9` vs `volumetric_blocks`). Adicionar um terceiro modo, no espírito do
projeto, é adicionar **um arquivo de config** (e, para o RAG, um novo eixo de
configuração ortogonal), **sem tocar no motor de dispatch — porque não há um**.

Isto é decisivo para o desenho do RAG (Fase 3): o RAG não é uma nova *estratégia
de painel*; é uma **camada de aumento do prompt** que pode compor com qualquer
estratégia de painel existente.

---

## 1. Como um "modo de operação" é definido, registrado e despachado hoje

### 1.1 Não há registro/enum/factory de modo

Busca no código: não existe `class Mode`, `MODE_REGISTRY`, `register_mode` nem
equivalente. O que existe:

- **Eixo de cobertura do painel** — `panel.strategy`, validado em
  [`dtwin/medgemma_client.py:176-189`](../../dtwin/medgemma_client.py):

  ```python
  panel = config.get("panel", {})
  strategy = str(panel.get("strategy", "uniform_9"))
  if strategy not in {"uniform_9", "volumetric_blocks"}:
      raise PipelineError("panel.strategy deve ser 'uniform_9' ou 'volumetric_blocks'.")
  ```

- **Eixo de renderização do tile** — `panel.mode`, também validado no mesmo
  `_validate_config` ([`medgemma_client.py:190-209`](../../dtwin/medgemma_client.py)):
  `single_grayscale` | `multiphase_fusion` | `texture_fusion`. Despachado em
  [`dtwin/medgemma_screening.py:187-255`](../../dtwin/medgemma_screening.py).

> **Achado central:** há **dois eixos ortogonais** já hoje (`strategy` de
> cobertura × `mode` de renderização), ambos resolvidos por `if/elif` sobre
> valores lidos do YAML — não por um registrador. O "modo baseline/volumétrico"
> da UI corresponde **apenas** ao eixo `strategy`.

### 1.2 O despacho de `strategy` (baseline × volumétrico)

Acontece dentro do gerador de painel, [`dtwin/medgemma_panel.py:222-255`](../../dtwin/medgemma_panel.py):

```python
strategy = panel_strategy(panel_cfg)           # lê panel.strategy do YAML
if strategy == "volumetric_blocks":
    panel_set = render_volumetric_panel_set(...)  # N painéis, 100% do fígado
# senão: caminho uniform_9 (1 painel, 9 cortes axiais)
...
"panel_strategy": strategy,                     # gravado no manifesto do painel
```

- `baseline` (`uniform_9`): 1 painel, 9 cortes axiais uniformes + coronal +
  sagital.
- `volumétrico` (`volumetric_blocks`): N painéis cobrindo 100% dos voxels do
  fígado, com **gate de cobertura voxel-a-voxel** em
  [`dtwin/medgemma_volumetric.py:214-231`](../../dtwin/medgemma_volumetric.py)
  (`covered_liver_voxels == total_liver_voxels`, senão `PipelineError`).

### 1.3 O arquivo de config **é** o modo

O modo volumétrico inteiro é este arquivo
([`configs/medgemma_local_4b_volumetric.yaml`](../../configs/medgemma_local_4b_volumetric.yaml)):

```yaml
extends: medgemma_local_4b.yaml
medgemma_screening:
  panel:
    strategy: volumetric_blocks
    axial_tiles_per_panel: 9
```

Herança via `extends` é resolvida em
[`medgemma_client.py:97-106`](../../dtwin/medgemma_client.py) (`_deep_merge`, com a
trava de que `extends` deve apontar para arquivo no **mesmo diretório**).

---

## 2. Como a geração de benchmarks seleciona e executa um modo

Há **dois pontos de entrada** que convergem para o mesmo executor
(`dtwin.medgemma_screening`), mas selecionam o modo de formas diferentes.

### 2.1 Webapp (a UI que expõe "Baseline/Volumétrico")

Ponto de entrada: `POST /api/benchmarks`
([`webapp/server.py:853-893`](../../webapp/server.py)).

1. O manifesto do form traz um campo `scenario`
   ([`server.py:721`](../../webapp/server.py)), validado contra um **dict fixo**
   ([`server.py:69-72`](../../webapp/server.py)):

   ```python
   BENCHMARK_SCENARIOS = {
       "baseline":  MEDGEMMA_CONFIG,             # configs/medgemma_local_4b.yaml
       "volumetric": VOLUMETRIC_MEDGEMMA_CONFIG, # configs/medgemma_local_4b_volumetric.yaml
   }
   ```

2. `_parse_benchmark_manifest` rejeita `scenario` fora do dict
   ([`server.py:727-728`](../../webapp/server.py)).
3. `process_benchmark` resolve `scenario → caminho de config` via
   `_benchmark_config` ([`server.py:436-444`](../../webapp/server.py)), que impõe
   uma trava de segurança: o arquivo resolvido **precisa estar dentro de
   `configs/`**.
4. Cada exame roda em `_run_benchmark_case(..., medgemma_config)`
   ([`server.py:466-598`](../../webapp/server.py)): segmentação + subprocesso
   `python -m dtwin.medgemma_screening --medgemma-config <config> --case-dir ...`.

> A UI **não** escolhe um "modo" no motor; ela escolhe uma **string** que é
> traduzida para **um arquivo de config**. É exatamente o ponto de extensão do
> RAG (ver Fase 3).

### 2.2 CLI auditável

Ponto de entrada: [`dtwin/medgemma_benchmark.py`](../../dtwin/medgemma_benchmark.py).
Aqui **não há abstração de "scenario"**: passa-se `--medgemma-config <arquivo>`
diretamente (`build_parser`, [`medgemma_benchmark.py:15-29`](../../dtwin/medgemma_benchmark.py)).
O arquivo de config É o modo. Executa via `run_benchmark`
([`dtwin/benchmark/runner.py:280-324`](../../dtwin/benchmark/runner.py)).

### 2.3 Formato de saída dos resultados

- Webapp: `benchmark_report.json` + saídas do núcleo compartilhado
  (`write_run_outputs`), com o campo `scenario` no relatório
  ([`server.py:646-658`](../../webapp/server.py)).
- CLI/núcleo: `run_manifest.json` (inclui `panel_strategy`,
  [`runner.py:118-134`](../../dtwin/benchmark/runner.py)), `cases.jsonl`, métricas.
- Métricas: [`dtwin/benchmark/metrics.py`](../../dtwin/benchmark/metrics.py) —
  accuracy, sensibilidade, especificidade, precisão, F1, IC de Wilson, cobertura,
  taxas de inconclusivo/falha, e um `gate` (sens/esp mínimas 0.75). **Nenhuma
  métrica de RAG hoje.**

---

## 3. Como o MedGemma/Ollama é invocado hoje

### 3.1 Cliente do pipeline → gateway HTTP

O pipeline conhece **apenas** um adaptador,
`HTTPJSONMedGemmaClient` ([`medgemma_client.py:408-557`](../../dtwin/medgemma_client.py)),
que faz `POST` a um gateway local com o contrato `dtwin-medgemma-v1`
([`medgemma_client.py:504-514`](../../dtwin/medgemma_client.py)):

```python
payload = {
    "contract": "dtwin-medgemma-v1",
    "model_id": ..., "model_version": ...,
    "prompt": prompt,                                  # << TEXTO (único canal textual)
    "image": {"mime_type": "image/png", "base64": ...},# << UMA imagem PNG
    "generation": {"max_output_tokens": ...},
}
```

> **Canal de injeção do RAG:** o contrato aceita **prompt textual arbitrário**
> (limitado por `max_prompt_chars`, default 12000 —
> [`medgemma_client.py:273-277`](../../dtwin/medgemma_client.py)) + **uma imagem**.
> O RAG injeta contexto recuperado **dentro do texto do prompt**. Nenhuma mudança
> no contrato do gateway é necessária para o MVP.

### 3.2 Montagem do prompt

`build_medgemma_prompt` ([`medgemma_client.py:264-278`](../../dtwin/medgemma_client.py))
lê `config.prompt.template` e **exige fragmentos de salvaguarda** (`POSITIVA`,
`NEGATIVA`, `INCONCLUSIVA`, "modo de pesquisa", "revisão humana obrigatória", "não
é diagnóstico", "não é laudo médico"). Um prompt sem eles → `PipelineError`.

O prompt vive **inteiramente no YAML** (`configs/*.yaml`, bloco `prompt.template`
— ex. [`medgemma_local_4b.yaml:66-98`](../../configs/medgemma_local_4b.yaml)).

### 3.3 Backend / runtimes

Gateway: [`tools/medgemma_server.py`](../../tools/medgemma_server.py). Dois
runtimes, escolhidos por `medgemma.runtime`
([`medgemma_server.py:293-300`](../../tools/medgemma_server.py)):

- `MedGemmaRuntime` (Transformers): `device: cuda` (NF4 via bitsandbytes) **ou**
  `device: mps` (bf16, **sem** quantização — bitsandbytes exige CUDA,
  [`medgemma_server.py:97-108`](../../tools/medgemma_server.py)).
- `OllamaRuntime`: delega ao daemon Ollama (GGUF/Metal),
  [`medgemma_server.py:198-290`](../../tools/medgemma_server.py). **É o runtime de
  deploy no Mac** ([`RUNBOOK_MAC.md`](../../RUNBOOK_MAC.md),
  [`configs/medgemma_ollama_27b.yaml`](../../configs/medgemma_ollama_27b.yaml)).

O gateway `/generate` monta **uma única mensagem `user`** com `[image, text]`
([`medgemma_server.py:172-180`](../../tools/medgemma_server.py)); não há papel
`system` (o template Gemma não aceita). As salvaguardas ficam no próprio texto.

### 3.4 Parsing da resposta

`_parse_json_report` ([`medgemma_client.py:281-319`](../../dtwin/medgemma_client.py))
extrai **um único** objeto JSON (tolera *reasoning* antes e cercas ```json```;
rejeita se houver 0 ou >1 objetos). `validate_medgemma_report`
([`medgemma_client.py:365-405`](../../dtwin/medgemma_client.py)) impõe: 7 campos
obrigatórios (`REQUIRED_REPORT_FIELDS`, [`medgemma_client.py:25-33`](../../dtwin/medgemma_client.py)),
estados/confiança permitidos, `necessidade_de_revisao_humana == true`, e uma
**lista de padrões proibidos** (diagnóstico definitivo / conduta clínica →
`PipelineError`).

> **Consequência para o RAG:** o schema JSON de saída é **fechado e validado**. Se
> o RAG precisar de novos campos (ex.: `citacoes`, `abstencao`), isso exige
> estender `REQUIRED_REPORT_FIELDS`/validação **ou** carregar essa informação no
> **envelope** externo (fora do `report`), não no `report`. Decisão de design na
> Fase 3.

---

## 4. Schema dos perfis e como um perfil é carregado

**Cuidado — há dois arquivos de config distintos, com papéis diferentes:**

| Arquivo | Papel | Carregado por |
|---|---|---|
| `profiles/figado.yaml` | Config do **órgão** (segmentação, malha, modalidade) | `load_profile` (`dtwin/core.py`) |
| `configs/medgemma_*.yaml` | Config do **screening/modelo** (painel, prompt, gates, backend) | `load_screening_config` ([`medgemma_client.py:87-142`](../../dtwin/medgemma_client.py)) |

### 4.1 Perfil do órgão (`profiles/figado.yaml`)

Estrutura ([`profiles/figado.yaml`](../../profiles/figado.yaml)): `id`,
`nome_exibicao`, `estado_regulatorio`, `modalidade`, `normalizacao`,
`segmentacao_orgao` (`motor`/`motor_task`/`rotulo_alvo`), `segmentacao_lesao`,
`refino`, `mesh`, `exportacao`, `flywheel`. Princípio explícito no cabeçalho:
*"Toda a configuração específica do órgão mora AQUI, não no código."*

### 4.2 Config de screening (`configs/medgemma_*.yaml`)

Blocos ([`medgemma_local_4b.yaml`](../../configs/medgemma_local_4b.yaml)):
`enabled`, `regulatory_mode`, `panel`, `validation`, `privacy`, `medgemma`
(provider/model/endpoint/timeout/device/quantization), `output`, `report`,
`prompt`. Validado inteiro por `_validate_config`
([`medgemma_client.py:153-242`](../../dtwin/medgemma_client.py)) — que **falha
fechado** em qualquer inconsistência. Suporta `extends` (herança) e overrides por
variável de ambiente (`MEDGEMMA_*`, [`medgemma_client.py:111-140`](../../dtwin/medgemma_client.py)).

> **Onde o RAG se configura:** um **novo bloco `rag:`** dentro de
> `medgemma_screening` (regras em config, nunca hard-coded), mais um novo arquivo
> `configs/medgemma_local_4b_rag.yaml` que `extends` o baseline e liga o RAG.
> Detalhe na Fase 3.

---

## 5. Onde e como o princípio "abortar em falha" é aplicado

O princípio ("SEMPRE abortar em falha, nunca fabricar saída") é **transversal** e
implementado por `PipelineError` levantado e **nunca** convertido em achado falso:

- **Config inválida** → `_validate_config` ([`medgemma_client.py:153`](../../dtwin/medgemma_client.py)).
- **Backend não pronto / identidade divergente** → `_ensure_ready` +
  `check_ready` ([`medgemma_client.py:420-488`](../../dtwin/medgemma_client.py));
  confere host loopback, health `status==ready`, e model_id/version idênticos.
- **Resposta fora do schema ou com conduta clínica** → `validate_medgemma_report`
  ([`medgemma_client.py:365`](../../dtwin/medgemma_client.py)).
- **Cobertura volumétrica incompleta** → gate voxel-a-voxel
  ([`medgemma_volumetric.py:230-231`](../../dtwin/medgemma_volumetric.py)).
- **Benchmark**: falhas viram `BenchmarkStatus.FAILURE/TIMEOUT/INVALID_RESPONSE`
  (`classify_screening_failure`, [`runner.py:157-167`](../../dtwin/benchmark/runner.py)),
  **nunca** uma predição inventada; métricas contam inconclusivo/falha como erro
  ([`metrics.py:116-129`](../../dtwin/benchmark/metrics.py)).
- **Webapp**: cartão gracioso `_graceful` ([`server.py:123-131`](../../webapp/server.py)),
  jamais um achado fabricado; e a trava recém-adicionada que bloqueia iniciar o
  benchmark com backend fora do ar (`refreshBackend`/`backendReady` em
  [`webapp/static/benchmark.html`](../../webapp/static/benchmark.html)).

> **Consequência para o RAG:** o modo `rag` deve herdar esse contrato — índice
> ausente, recuperação vazia, embedding indisponível ou abstenção do modelo
> **abortam** (ou marcam `INVALID_RESPONSE`/`inconclusive`), nunca produzem um
> laudo grounded em contexto inexistente. Detalhe na Fase 3.

---

## 6. Fatos de plataforma (Apple Silicon / deploy) relevantes ao RAG

- **Deploy real = Mac**, Ollama + MedGemma 27B via Metal
  ([`RUNBOOK_MAC.md`](../../RUNBOOK_MAC.md)). Sem CUDA.
- O gateway já detecta/serve MPS (`device: mps`, bf16) e Ollama (Metal).
- A detecção de ambiente do benchmark já reconhece MPS
  ([`runner.py:80-98`](../../dtwin/benchmark/runner.py)).
- Extra de dependências `[medgemma]` inclui `transformers`, `accelerate`,
  `sentencepiece`; `bitsandbytes` (CUDA-only) só é usado no caminho `device: cuda`
  ([`pyproject.toml:38-45`](../../pyproject.toml)). **Nenhuma dependência de RAG
  existe ainda.**

---

## 7. Decisões externas ao código

1. **PA-1 — Consulta de recuperação:** a tarefa do ARGOS é classificação a partir
   de **imagem** (o único texto de entrada é o prompt fixo). Um retriever precisa
   de uma **consulta textual**. Não há no código nada que defina o que seria a
   consulta do RAG. → **Decisão consolidada:** duas passagens
   (*retrieve-then-verify*), usando rascunho do modelo + consulta-base fixa.
2. **PA-2 — Base de conhecimento:** não existe corpus, índice nem fonte definida
   no repo. → **Decisão consolidada:** Opção B sem EASL; ainda falta a lista
   concreta de documentos/fontes.
3. **PA-3 — Golden set / validação clínica:** não há `benchmarks/*` com QAs de
   RAG nem validador clínico. Os exemplos existentes
   ([`benchmarks/*.example.yaml`](../../benchmarks/)) são de **labels de exame**
   (positive/negative), não de recuperação. → **Decisão consolidada:** usar labels
   binários controlados para sensibilidade/especificidade; golden set textual segue
   como grounding/desenvolvimento até revisão especializada.
4. **PA-4 — Campos de saída do RAG:** se citações/abstenção entram no `report`
   (exige estender o schema validado) ou no envelope externo. → **Decisão
   consolidada:** envelope externo.

Próximo: [`01-arquitetura.md`](01-arquitetura.md).
