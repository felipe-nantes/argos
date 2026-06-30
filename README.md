# Digital Twin Cirúrgico — UEM · GETS · HU

Projeto de **planejamento cirúrgico** que transforma uma série **DICOM de RM** em
um **modelo 3D do órgão do paciente com a lesão destacada**, exportável em **STL** e
publicado para um visualizador web.

Este repositório reúne, num só lugar, **a estratégia completa** (pasta `contexto/`)
e **a implementação do pipeline** (raiz + `dtwin/` + `profiles/`).

> **Modo Pesquisa.** O que está aqui é a fundação anatômica (Nível 1) de um digital
> twin — **não é dispositivo médico** e **não se destina a decisão clínica**. A
> transição para uso clínico (ANVISA, pseudonimização, etc.) é um gate formal
> descrito em `contexto/03_REGULATORIO_LGPD.md` e `contexto/10_MATURIDADE_DIGITAL_TWIN.md`.

---

## Por onde começar

- **Para entender o produto e a estratégia:** abra `contexto/00_CONTEXTO.md` (é a
  camada de orientação que linka todos os outros módulos).
- **Para rodar o pipeline:** siga a seção "Como rodar" abaixo.

## Estrutura do repositório

```
digital_twin_cirurgico/
├── README.md                       # este arquivo (porta de entrada)
├── contexto/                       # ESTRATÉGIA — 11 módulos
│   ├── 00_CONTEXTO.md              #   índice / camada de orientação
│   ├── 01_VISAO.md
│   ├── 02_DOMINIO_CLINICO.md
│   ├── 03_REGULATORIO_LGPD.md
│   ├── 04_ARQUITETURA.md
│   ├── 05_PIPELINE.md
│   ├── 06_SEGMENTACAO.md
│   ├── 07_INFRA_CUSTOS.md
│   ├── 08_ROADMAP.md
│   ├── 09_NEGOCIO.md
│   └── 10_MATURIDADE_DIGITAL_TWIN.md
│
├── digital_twin.py                 # CÓDIGO — entrada (CLI), obedece o motor
├── requirements.txt
├── profiles/
│   └── figado.yaml                 # perfil do órgão (TODA a config específica)
└── dtwin/                          # o motor (órgão-agnóstico, determinístico)
    ├── core.py                     #   gates, geometria, Case, loader de perfil
    ├── stages.py                   #   os 7 estágios, cada um com seu gate
    └── engine.py                   #   orquestrador (prepare / finalize)
```

Princípio central que liga as duas metades: **regra de domínio mora em config
versionada, não no código**. Trocar de órgão = adicionar um perfil em `profiles/`,
nunca mexer no motor.

---

## Mapa da estratégia (`contexto/`)

| Arquivo | O que responde |
|---|---|
| `00_CONTEXTO.md` | Índice, fase atual, as 5 regras de ouro |
| `01_VISAO.md` | Produto, usuário, dor, o que o MVP **não** é |
| `02_DOMINIO_CLINICO.md` | Fígado, RM, fidelidade, lesão, expansão de órgãos |
| `03_REGULATORIO_LGPD.md` | Pesquisa→Clínico, anonimização, CEP, responsabilidade |
| `04_ARQUITETURA.md` | Pipeline órgão-agnóstico, perfis plugáveis, stack |
| `05_PIPELINE.md` | Os estágios ponta a ponta e seus gates de segurança |
| `06_SEGMENTACAO.md` | Órgão automático + lesão manual, dados, validação |
| `07_INFRA_CUSTOS.md` | Hardware, deploy local, custo, origem dos exames |
| `08_ROADMAP.md` | Fase 0→3, gates entre fases |
| `09_NEGOCIO.md` | Modelos de receita e sustentação (em aberto) |
| `10_MATURIDADE_DIGITAL_TWIN.md` | Escada de 4 níveis: modelo anatômico → twin preditivo |

---

## O pipeline (código)

### Instalação

```bash
pip install -r requirements.txt
# 3D Slicer à parte: https://www.slicer.org  (marcação manual da lesão)
```

GPU é fortemente recomendada para a segmentação automática (ver
`contexto/07_INFRA_CUSTOS.md`). Em CPU, use `--device cpu --fast`.

### Como rodar (fluxo de duas fases)

A lesão é marcada por um humano no 3D Slicer, então o pipeline tem dois comandos
com uma etapa manual no meio.

**Fase 1 — `prepare`** (estágios 1–4a): ingestão + des-identificação +
normalização + segmentação automática do órgão.

```bash
python digital_twin.py prepare /caminho/serie_dicom \
       --case-dir casos/paciente001 --profile profiles/figado.yaml
```

**Etapa manual — 3D Slicer** (estágio 4): abra o volume e a máscara do órgão,
revise/corrija o órgão e **marque a lesão**, salvando em
`casos/paciente001/mask_lesion.nii.gz`. (As instruções exatas são impressas ao
fim do `prepare`.)

**Fase 2 — `finalize`** (estágios 4b–7): importa a lesão + refino + malha + STL +
manifesto do visualizador.

```bash
python digital_twin.py finalize casos/paciente001 --profile profiles/figado.yaml
```

Saídas em `casos/paciente001/outputs/`:
`figado_orgao.stl`, `figado_lesao.stl` (LPS) e `viewer_manifest.json`.

Caso realmente não haja lesão: `finalize ... --no-lesion` (escolha explícita; o
pipeline não fabrica nada).

### Os sete estágios

1. **Ingestão + des-identificação** — lê DICOM com geometria correta; anonimiza
   (converte para NIfTI, descartando os cabeçalhos com PHI).
2. **Normalização** — z-score para RM (referência/inspeção; *não* é o que vai ao
   segmentador).
3. **Segmentação do órgão** — automática (TotalSegmentator MRI, task `total_mr`).
4. **Lesão** — `4a` prepara o handoff para o Slicer; `4b` importa a marcação
   humana e a arquiva para o *flywheel*.
5. **Refino** — morfologia + remoção de fragmentos (gentil na lesão).
6. **Malha** — marching cubes + suavização (superfície).
7. **Exportação** — STL em LPS + manifesto para o visualizador web.

### Regras de ouro (no código)

- **Nunca fabricar dado.** Se a segmentação não carrega ou algo falha, o pipeline
  **aborta** (`PipelineError`). O script original gerava máscara aleatória — aqui
  isso é proibido.
- **Regra de domínio em config**, não no código (perfis YAML).
- **Saída automática nunca é confiada às cegas** — sempre há revisão humana.
- **Dado de paciente entra anonimizado**; a pseudonimização é um ponto de
  extensão reservado para o uso clínico futuro.

### Bugs do script original corrigidos

- Slope/intercept de HU aplicado em dobro → o leitor já aplica; não reaplicar.
- Ordem de eixos do `spacing` (x↔z) → geometria via SimpleITK + transformação
  física completa (origin + direction + spacing); trata até aquisições oblíquas.
- `selem=` → `footprint=` (API atual do scikit-image).
- `pv.save_mesh_as` (inexistente) → `mesh.save(...)`.
- `faces.reshape(-1, 3)` (embaralhava faces) → contador `[3, i, j, k, ...]` correto.
- Exportar `.feb` via Trimesh (não funciona) → FEA adiado para a fase 2.
- Fallback de máscara aleatória → **abortar**.
- Import de nnU-Net inexistente → TotalSegmentator (API verificada).
- Visualizador PyVista bloqueante no fim → removido; visualização é o app web.

### Como adicionar um novo órgão (ex.: baço)

1. Copie `profiles/figado.yaml` para `profiles/baco.yaml`.
2. Ajuste `id`, `nome_exibicao` e `segmentacao_orgao.rotulo_alvo: spleen`.
3. Rode com `--profile profiles/baco.yaml`. **O motor não muda.**

(Para descobrir nomes de classe válidos: `totalseg_info --classes -ta total_mr`.)

### Ferramentas de produção

- **Testes:** `.venv/Scripts/python.exe -m pytest` (não requer GPU/torch).
- **Preflight:** `digital-twin doctor` — checa dependências e device.
- **Caso sintético:** `python tools/make_synthetic_case.py --out casos/sintetico`
  gera um caso fictício para rodar `finalize` sem GPU/Slicer/DICOM.
- **Visualizador web:** `viewer/index.html` (Three.js, sem build) — ver `viewer/README.md`.
- **Guia de uso passo a passo (tutorial prático):** `docs/GUIA_DE_USO.md`.
- **Roteiro de apresentação (demo 5 min, offline):** `docs/DEMO.md`.
- **Preparar a caixa GPU (1 comando):** `py -3.13 tools/setup_real_env.py` — cria
  venv, instala `[seg]` e verifica GPU + rótulo do órgão.
- **Smoke test na caixa GPU (exame real):** `tools/smoke_gpu.py`.
- **Guia de execução (referência curta):** `docs/RUNNING.md`.

### Fora do escopo do MVP

Frontend do visualizador web, integração PACS, pseudonimização ativa, FEA /
tetraedralização (Nível 2), relatório PDF comparativo e modelo próprio de lesão —
todos mapeados em `contexto/08_ROADMAP.md` e `contexto/10_MATURIDADE_DIGITAL_TWIN.md`.
