# ARGOS · Pré - Digital Twin Cirúrgico

![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![Modo](https://img.shields.io/badge/modo-Pesquisa-orange)
![Licença](https://img.shields.io/badge/licença-Proprietária-lightgrey)

O ARGOS transforma uma série DICOM de ressonância magnética no modelo 3D do órgão
do paciente, com a lesão destacada e exportável em STL. Ele também faz uma triagem
visual por IA médica que roda inteiramente na máquina local, sem mandar dados do
paciente para fora.

O repositório reúne duas coisas: a estratégia do produto (pasta `contexto/`, 12
módulos) e a implementação do pipeline (`dtwin/`, `profiles/`, `webapp/`,
`viewer/`). O trabalho é desenvolvido no contexto acadêmico UEM · GETS · HU.

> **Modo Pesquisa.** Isto é a fundação anatômica (Nível 1) de um digital twin. Não
> é dispositivo médico e não serve para diagnóstico ou decisão clínica. Toda saída
> automática sai marcada como `pending_review` e depende de revisão humana. A
> passagem para uso clínico é um gate formal, descrito em
> `contexto/03_REGULATORIO_LGPD.md`.

## O fluxo

Você arrasta a pasta DICOM de um exame no webapp local e o ARGOS faz o resto, em
subprocessos isolados:

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
 5. malha + STL do fígado  →  6. visualizador 3D  →  aprovação humana registrada
```

Se alguma etapa falhar, o app mostra um cartão de "análise não concluída" em vez
de um achado clínico inventado. E se uma biblioteca nativa quebrar (falta de
memória na GPU, segfault), quem cai é só o subprocesso; o servidor web continua no
ar.

## Características principais

- A IA roda no seu próprio hardware. O MedGemma 27B é servido localmente pelo
  Ollama (acelerado por Metal no Apple Silicon), atrás de um contrato HTTP
  chamado `dtwin-medgemma-v1`. Não há chave de API nem chamada externa na hora da
  inferência.

- O pipeline aborta em vez de inventar. Se a segmentação não carrega ou o modelo
  não está pronto, ele para com um erro explícito (`PipelineError`). O `/health`
  do backend mostra a falha real. Não existe resposta clínica simulada nem troca
  silenciosa por outro modelo.

- A saída do modelo passa por uma trava. A resposta é validada contra um esquema
  fixo (`POSITIVA`, `NEGATIVA` ou `INCONCLUSIVA`, com nível de confiança) e é
  rejeitada se contiver diagnóstico definitivo ou recomendação de conduta. Todo
  relatório sai com `requires_human_review: true`.

- Tem um benchmark embutido. Você aponta um dataset rotulado e o ARGOS roda cada
  exame pelo pipeline real, calculando matriz de confusão, sensibilidade,
  especificidade, precisão, F1, acurácia, cobertura e intervalos de confiança,
  com exportação em JSON e CSV. Casos inconclusivos e falhas contam como erro nas
  métricas principais e também ficam discriminados, evitando melhora por exclusão.

- Trocar de órgão é trocar um arquivo. A regra clínica fica num YAML versionado,
  não no código. Sair do fígado para o baço é copiar o perfil e mudar o rótulo do
  alvo. O motor não muda.

- Privacidade por construção. A des-identificação descarta os cabeçalhos DICOM na
  conversão para NIfTI, e a pasta `casos/` nunca vai para o Git. O risco que sobra
  (PHI queimada no pixel) está escrito no manifesto, não escondido.

- Saída determinística. Mesmo exame, mesma config e mesma marcação dão o mesmo
  resultado. Cada caso deixa um `run.log` e um manifesto com os hashes SHA-256 das
  entradas.

- Visualizador 3D offline. O Three.js vai vendorizado (sem CDN) e mostra o STL em
  coordenadas LPS. A decisão do revisor fica gravada em `outputs/approval.json`.

- Testes e ferramentas. São mais de 100 testes automatizados (geometria, gates de
  segurança, parser do MedGemma, webapp), CI no GitHub Actions, um subcomando
  `doctor` de preflight, um gerador de caso sintético para testar sem GPU nem
  DICOM, e o `run_mac.sh`, que sobe o ambiente inteiro com um comando.

## Os sete estágios

O motor (`dtwin/`) é determinístico e órgão-agnóstico. Cada estágio valida a
própria entrada e aborta se algo estiver errado, em vez de seguir mesmo assim.

| # | Estágio | O que faz |
|---|---|---|
| 1 | Ingestão + des-identificação | Lê o DICOM com a geometria correta e anonimiza (NIfTI, sem PHI). |
| 2 | Normalização | z-score para RM (referência de inspeção). |
| 3 | Segmentação do órgão | Automática, com TotalSegmentator MRI (`total_mr`). |
| 4 | Lesão | `4a` prepara o handoff para o 3D Slicer; `4b` importa a marcação humana. |
| 5 | Refino | Morfologia e remoção de fragmentos, sem apagar lesão pequena. |
| 6 | Malha | Marching cubes e suavização de superfície. |
| 7 | Exportação | STL em LPS e manifesto para o visualizador web. |

O fluxo tem duas fases com revisão humana no meio: `prepare` (estágios 1 a 4a), a
marcação da lesão no 3D Slicer, e depois `finalize` (4b a 7).

## Rodar

### Modo rápido (MAC, um comando)

```bash
bash run_mac.sh
```

Sobe Ollama, gateway MedGemma e webapp na ordem certa, com verificação de saúde
entre cada etapa, e abre `http://127.0.0.1:8080`. O passo a passo completo está em
[`RUNBOOK_MAC.md`](RUNBOOK_MAC.md).

### Modo rápido (Windows, MedGemma 4B)

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_win.ps1
```

O launcher valida a `.venv-win`, CUDA e as dependências, sobe o gateway com
`configs/medgemma_local_4b.yaml` e inicia o mesmo webapp em
`http://127.0.0.1:8080`. O backend muda, mas o contrato HTTP e o benchmark são os
mesmos usados pelo 27B no Mac. O `ExecutionPolicy Bypass` vale somente para esse
processo e não altera permanentemente a política do Windows.

Na primeira execução, aceite a licença do MedGemma no Hugging Face, autentique-se
com `.\.venv-win\Scripts\hf.exe auth login` e baixe os pesos com
`.\.venv-win\Scripts\python.exe tools\setup_medgemma.py --config configs\medgemma_local_4b.yaml`.

### Pipeline por linha de comando

```bash
pip install -e ".[seg]"          # traz TotalSegmentator e torch (GPU recomendada)

# Fase 1: ingestão + des-identificação + segmentação automática do órgão
python digital_twin.py prepare /caminho/serie_dicom \
       --case-dir casos/paciente001 --profile profiles/figado.yaml

# marque a LESÃO no 3D Slicer (as instruções são impressas ao fim do prepare)

# Fase 2: importa a lesão + refino + malha + STL + manifesto do visualizador
python digital_twin.py finalize casos/paciente001 --profile profiles/figado.yaml
```

As saídas ficam em `casos/paciente001/outputs/`: `figado_orgao.stl`,
`figado_lesao.stl` (em LPS) e `viewer_manifest.json`. Se o caso não tiver lesão,
use `finalize ... --no-lesion`. É uma escolha explícita; o pipeline não fabrica
nada.

### Triagem e benchmark pelo navegador

- Exame individual: `http://127.0.0.1:8080`. Arraste a pasta DICOM e receba o
  relatório do MedGemma e o botão "Visualizar fígado em 3D e revisar".
- Benchmark: `http://127.0.0.1:8080/benchmark.html`. Arraste um dataset rotulado
  (uma subpasta por exame) e obtenha a matriz de confusão e as métricas.

Para runs auditáveis por CLI, importação NIfTI/MIDS, dry-run, isolamento de ground
truth e reprodução por commit/config/hash, consulte
[`benchmarks/README.md`](benchmarks/README.md).

## Adicionar um novo órgão (por exemplo, baço)

1. Copie `profiles/figado.yaml` para `profiles/baco.yaml`.
2. Ajuste `id`, `nome_exibicao` e `segmentacao_orgao.rotulo_alvo: spleen`.
3. Rode com `--profile profiles/baco.yaml`. O motor não muda.

## Estrutura do repositório

```
argos/
├── contexto/          estratégia do produto, 12 módulos (visão, domínio, LGPD, roadmap)
├── dtwin/             o motor, órgão-agnóstico e determinístico
│   ├── core.py            gates, geometria, Case, loader de perfil
│   ├── stages.py          os 7 estágios, cada um com seu gate
│   ├── engine.py          orquestrador (prepare / finalize)
│   ├── medgemma_client.py config, prompt, adaptador HTTP e travas de resposta
│   ├── medgemma_panel*.py montagem 2D (single, multifásica, textura)
│   └── medgemma_screening.py  fluxo pós-prepare
├── profiles/figado.yaml   a regra específica do órgão (config, não código)
├── configs/               backends MedGemma (Ollama 27B, MPS, 4-bit)
├── webapp/                orquestração FastAPI: exame individual e benchmark
├── viewer/                visualizador 3D Three.js (offline, vendorizado)
├── digital_twin.py        CLI, a camada fina que obedece o motor
├── run_mac.sh             sobe o ambiente com um comando
└── RUNBOOK_MAC.md         passo a passo operacional
```

### Mapa da estratégia (`contexto/`)

| Arquivo | O que responde |
|---|---|
| `00_CONTEXTO.md` | Índice, fase atual, as regras de ouro |
| `01_VISAO.md` | Produto, usuário, dor, o que o MVP não é |
| `02_DOMINIO_CLINICO.md` | Fígado, RM, fidelidade, lesão, expansão de órgãos |
| `03_REGULATORIO_LGPD.md` | Pesquisa para clínico, anonimização, CEP, responsabilidade |
| `04_ARQUITETURA.md` | Pipeline órgão-agnóstico, perfis plugáveis, stack |
| `05_PIPELINE.md` | Os estágios ponta a ponta e seus gates |
| `06_SEGMENTACAO.md` | Órgão automático e lesão manual, dados, validação |
| `07_INFRA_CUSTOS.md` | Hardware, deploy local, custo, origem dos exames |
| `08_ROADMAP.md` | Fase 0 a 3, gates entre fases |
| `09_NEGOCIO.md` | Modelos de receita e sustentação |
| `10_MATURIDADE_DIGITAL_TWIN.md` | Escada de 4 níveis: anatômico a twin preditivo |
| `11_MEDGEMMA_SCREENING.md` | Triagem visual hepática MedGemma |

## Bugs do script original corrigidos

O ponto de partida foi um script que, entre outros problemas, gerava uma máscara
aleatória quando a IA não carregava. O ARGOS reescreveu essa parte:

- Slope/intercept de HU aplicado em dobro. O leitor já aplica, então não se
  reaplica.
- Ordem de eixos do `spacing` (x e z trocados). Agora a geometria física é
  reconstruída via SimpleITK (origin, direction e spacing), e trata aquisições
  oblíquas.
- `selem=` virou `footprint=` (API atual do scikit-image).
- `faces.reshape(-1, 3)`, que embaralhava as faces, virou o contador correto
  `[3, i, j, k, ...]`.
- Import de nnU-Net inexistente virou TotalSegmentator, com a API verificada.
- O fallback de máscara aleatória virou abortar.

## Fora do escopo por enquanto

Interface clínica integrada, integração com PACS, pseudonimização ativa, FEA e
tetraedralização (Nível 2), segmentos de Couinaud e vasculatura hepática. Estão
mapeados e datados em `contexto/08_ROADMAP.md` e
`contexto/10_MATURIDADE_DIGITAL_TWIN.md`. O que está listado acima funciona hoje;
o resto está no roadmap.

---

ARGOS · Digital Twin Cirúrgico · UEM · GETS · HU · Modo Pesquisa, não destinado a
diagnóstico ou decisão clínica.
