# ARGOS — Digital Twin Cirúrgico

![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![Modo](https://img.shields.io/badge/modo-Pesquisa-orange)
![IA](https://img.shields.io/badge/IA-MedGemma%2027B%20local-6f42c1)
![Licença](https://img.shields.io/badge/licença-Proprietária-lightgrey)

> **Do exame ao modelo 3D — sem a nuvem, sem PHI, sem achismo.**
> O ARGOS transforma uma ressonância magnética (série **DICOM**) no **modelo 3D do
> órgão do paciente com a lesão destacada**, e ainda executa uma **triagem visual
> por IA médica rodando 100% local**. Ponta a ponta, na sua máquina.

Projeto de **planejamento cirúrgico** que une, num só repositório, a **estratégia
completa** do produto (pasta `contexto/`, 12 módulos) e a **implementação real do
pipeline** (`dtwin/` + `profiles/` + `webapp/` + `viewer/`). Desenvolvido no
contexto acadêmico **UEM · GETS · HU**.

> ⚠️ **Modo Pesquisa.** Esta é a fundação anatômica (Nível 1) de um digital twin —
> **não é dispositivo médico** e **não se destina a diagnóstico ou decisão
> clínica**. Toda saída automática é rotulada `pending_review` e exige revisão
> humana. A transição para uso clínico é um gate formal (ver
> `contexto/03_REGULATORIO_LGPD.md`). Essa honestidade é uma decisão de projeto,
> não uma limitação escondida.

---

## O fluxo em um olhar

Arraste a pasta DICOM de um exame no webapp local e o ARGOS faz o resto — **em
subprocessos isolados, à prova de falhas, sem você ver a complexidade**:

```
 pasta DICOM (RM)
      │
      ▼
 1. des-identificação  →  2. segmentação do fígado  →  3. montagem 2D
      │                         (TotalSegmentator)          (painel de revisão)
      ▼
 4. triagem MedGemma 27B (local)  →  relatório estruturado (pending_review)
      │
      ▼
 5. malha + STL do fígado  →  6. visualizador 3D  →  ✔ aprovação humana registrada
```

Se **qualquer** etapa falhar, o app mostra um cartão honesto — *"análise não
concluída"* — e **nunca** um achado clínico fabricado. Um crash nativo (OOM de
GPU, segfault de biblioteca) derruba apenas o subprocesso: **o servidor web
permanece de pé**.

---

## Por que o ARGOS impressiona

- 🧠 **IA médica que roda no seu hardware.** MedGemma 27B servido localmente via
  **Ollama** (acelerado por Metal em Apple Silicon), sob um contrato próprio
  `dtwin-medgemma-v1`. Nenhum byte do paciente sai da máquina. Sem chave de API,
  sem nuvem, sem dependência externa em tempo de inferência.

- 🚫 **Fail-closed por princípio — não fabrica dado, nunca.** Se a segmentação não
  carrega ou o modelo não está pronto, o pipeline **aborta** (`PipelineError`) em
  vez de inventar. O `/health` do backend expõe a falha real; não há resposta
  clínica simulada nem *fallback* silencioso para outro modelo.

- 🛡️ **Trava de segurança na saída da IA.** A resposta do modelo é validada contra
  um esquema rígido (`POSITIVA` / `NEGATIVA` / `INCONCLUSIVA` + confiança) e
  **rejeitada** se contiver diagnóstico definitivo ou recomendação de conduta.
  Todo relatório carrega o disclaimer e `requires_human_review: true`.

- 📊 **Benchmark científico embutido.** Aponte um dataset rotulado e o ARGOS roda
  **cada exame pelo pipeline real**, produzindo **matriz de confusão,
  sensibilidade, especificidade, precisão, F1, acurácia e cobertura**, com
  exportação **JSON e CSV**. Inconclusivos e falhas ficam **visíveis e separados**
  para não maquiar a métrica por exclusão. Nenhuma métrica é fabricada.

- 🧩 **Órgão-agnóstico de verdade.** A regra clínica mora em **config YAML
  versionada, não no código**. Trocar de fígado para baço = copiar um perfil e
  mudar um rótulo. **O motor não muda uma linha.**

- 🔒 **Privacidade por construção (LGPD).** A des-identificação descarta os
  cabeçalhos DICOM ao converter para NIfTI; dados de paciente (`casos/`) **nunca**
  são versionados. O risco residual (PHI queimada em pixel) é declarado
  explicitamente, não varrido para debaixo do tapete.

- 🎯 **Determinístico e auditável.** Mesmo exame + mesma config + mesma marcação =
  mesma saída. Cada caso deixa uma trilha `run.log` e um manifesto com hashes
  SHA-256 das entradas.

- 🖥️ **Visualizador 3D offline.** Three.js **vendorizado** (sem CDN) renderiza o
  STL do órgão/lesão em coordenadas LPS e registra a decisão humana
  (`Aprovar` / `Solicitar revisão`) em `outputs/approval.json`.

- ⚙️ **Engenharia levada a sério.** Mais de **100 testes automatizados** (geometria,
  gates de segurança, parser do MedGemma, webapp), **CI no GitHub Actions**,
  subcomando `doctor` de preflight, gerador de caso sintético (testa o pipeline
  sem GPU/DICOM) e **`run_mac.sh` que sobe todo o ambiente com um comando** e
  verificações de saúde.

---

## Arquitetura em sete estágios

O motor (`dtwin/`) é determinístico e órgão-agnóstico; cada estágio **valida a
própria entrada e aborta se algo estiver errado** — nunca "segue mesmo assim".

| # | Estágio | O que faz |
|---|---|---|
| 1 | **Ingestão + des-identificação** | Lê o DICOM com geometria correta e anonimiza (NIfTI, sem PHI). |
| 2 | **Normalização** | z-score para RM (referência de inspeção). |
| 3 | **Segmentação do órgão** | Automática — TotalSegmentator MRI (`total_mr`). |
| 4 | **Lesão** | `4a` prepara o handoff para o 3D Slicer; `4b` importa a marcação humana. |
| 5 | **Refino** | Morfologia + remoção de fragmentos (gentil com a lesão). |
| 6 | **Malha** | Marching cubes + suavização de superfície. |
| 7 | **Exportação** | STL em LPS + manifesto para o visualizador web. |

Fluxo de duas fases com **revisão humana no meio**: `prepare` (1–4a) → marcação da
lesão no 3D Slicer → `finalize` (4b–7).

---

## Rodar

### Modo rápido (MAC — um comando)

```bash
bash run_mac.sh
```

Sobe **Ollama → gateway MedGemma → webapp** na ordem certa, com verificação de
saúde entre cada etapa, e abre `http://127.0.0.1:8080`. Passo a passo completo em
[`RUNBOOK_MAC.md`](RUNBOOK_MAC.md).

### Pipeline por linha de comando

```bash
pip install -e ".[seg]"          # traz TotalSegmentator + torch (GPU recomendada)

# Fase 1 — ingestão + des-identificação + segmentação automática do órgão
python digital_twin.py prepare /caminho/serie_dicom \
       --case-dir casos/paciente001 --profile profiles/figado.yaml

# >>> marque a LESÃO no 3D Slicer (instruções impressas ao fim do prepare) <<<

# Fase 2 — importa a lesão + refino + malha + STL + manifesto do visualizador
python digital_twin.py finalize casos/paciente001 --profile profiles/figado.yaml
```

Saídas em `casos/paciente001/outputs/`: `figado_orgao.stl`, `figado_lesao.stl`
(LPS) e `viewer_manifest.json`. Sem lesão? `finalize ... --no-lesion` (escolha
explícita — o pipeline não fabrica nada).

### Triagem e benchmark pelo navegador

- **Exame individual:** `http://127.0.0.1:8080` — arraste a pasta DICOM, receba o
  relatório MedGemma e o botão **"Visualizar fígado em 3D e revisar"**.
- **Benchmark:** `http://127.0.0.1:8080/benchmark.html` — arraste um dataset
  rotulado (uma subpasta por exame) e obtenha a matriz de confusão e as métricas.

---

## Adicionar um novo órgão (ex.: baço)

1. Copie `profiles/figado.yaml` para `profiles/baco.yaml`.
2. Ajuste `id`, `nome_exibicao` e `segmentacao_orgao.rotulo_alvo: spleen`.
3. Rode com `--profile profiles/baco.yaml`. **O motor não muda.**

---

## Estrutura do repositório

```
argos/
├── contexto/          ESTRATÉGIA — 12 módulos (visão, domínio, LGPD, roadmap...)
├── dtwin/             O MOTOR — órgão-agnóstico e determinístico
│   ├── core.py            gates, geometria, Case, loader de perfil
│   ├── stages.py          os 7 estágios, cada um com seu gate
│   ├── engine.py          orquestrador (prepare / finalize)
│   ├── medgemma_client.py config, prompt, adaptador HTTP e travas de resposta
│   ├── medgemma_panel*.py montagem 2D (single / multifásica / textura)
│   └── medgemma_screening.py  fluxo pós-prepare
├── profiles/figado.yaml   TODA a regra específica do órgão (config, não código)
├── configs/               backends MedGemma (Ollama 27B, MPS, 4-bit...)
├── webapp/                orquestração FastAPI: exame individual + benchmark
├── viewer/                visualizador 3D Three.js (offline, vendorizado)
├── digital_twin.py        CLI — a camada fina que obedece o motor
├── run_mac.sh             sobe todo o ambiente com um comando
└── RUNBOOK_MAC.md         passo a passo operacional
```

### Mapa da estratégia (`contexto/`)

| Arquivo | O que responde |
|---|---|
| `00_CONTEXTO.md` | Índice, fase atual, as regras de ouro |
| `01_VISAO.md` | Produto, usuário, dor, o que o MVP **não** é |
| `02_DOMINIO_CLINICO.md` | Fígado, RM, fidelidade, lesão, expansão de órgãos |
| `03_REGULATORIO_LGPD.md` | Pesquisa→Clínico, anonimização, CEP, responsabilidade |
| `04_ARQUITETURA.md` | Pipeline órgão-agnóstico, perfis plugáveis, stack |
| `05_PIPELINE.md` | Os estágios ponta a ponta e seus gates |
| `06_SEGMENTACAO.md` | Órgão automático + lesão manual, dados, validação |
| `07_INFRA_CUSTOS.md` | Hardware, deploy local, custo, origem dos exames |
| `08_ROADMAP.md` | Fase 0→3, gates entre fases |
| `09_NEGOCIO.md` | Modelos de receita e sustentação |
| `10_MATURIDADE_DIGITAL_TWIN.md` | Escada de 4 níveis: anatômico → twin preditivo |
| `11_MEDGEMMA_SCREENING.md` | Triagem visual hepática MedGemma |

---

## Rigor de engenharia: bugs reais do script original corrigidos

O ponto de partida foi um script que, entre outros problemas, **gerava uma máscara
aleatória quando a IA não carregava**. O ARGOS reescreveu isso do zero:

- Slope/intercept de HU aplicado em dobro → o leitor já aplica; não reaplicar.
- Ordem de eixos do `spacing` (x↔z) → geometria física completa via SimpleITK
  (origin + direction + spacing); trata até aquisições oblíquas.
- `selem=` → `footprint=` (API atual do scikit-image).
- `faces.reshape(-1, 3)` (embaralhava faces) → contador `[3, i, j, k, ...]` correto.
- Import de nnU-Net inexistente → TotalSegmentator (API verificada).
- **Fallback de máscara aleatória → abortar.**

---

## Fora do escopo (por enquanto)

Interface clínica integrada, integração PACS, pseudonimização ativa, FEA /
tetraedralização (Nível 2), segmentos de Couinaud e vasculatura hepática — todos
mapeados e datados em `contexto/08_ROADMAP.md` e
`contexto/10_MATURIDADE_DIGITAL_TWIN.md`. O que está aqui **funciona hoje**; o
resto está no mapa, não no marketing.

---

<sub>ARGOS · Digital Twin Cirúrgico · UEM · GETS · HU · Modo Pesquisa — não
destinado a diagnóstico ou decisão clínica.</sub>
