# 08 · Roadmap

Fases com **gates** entre elas: uma fase só começa quando o gate da anterior é
satisfeito. Isto evita pular etapas (ex.: ir a paciente real sem ética aprovada).

---

## Fase 0 — Fundação e dados *(o gate que resolve o "não sei")*

O maior risco prático é **disponibilidade de exames**. Esta fase existe para
destravá-lo **em paralelo**, sem segurar o desenvolvimento.

Trilha técnica (pode começar **já**):
- Montar o ambiente (GPU, TotalSegmentator MRI, 3D Slicer, backend).
- Desenvolver o pipeline ponta a ponta contra **dados públicos de RM de fígado**
  (ex.: CHAOS) — valida ingestão, segmentação de órgão, refino, mesh, STL e
  visualizador sem depender do dado clínico.

Trilha de dados (em paralelo, mais lenta):
- Definir, com o orientador e o CEP, o **protocolo de pesquisa** e a coleta
  **retrospectiva** de exames de RM de fígado do HU (idealmente alguns com lesão).
- Definir o **perfil de des-identificação** aceito pelo HU/CEP.

**Gate para a Fase 1:** pipeline roda fim-a-fim em dado público **E** existe um
caminho aprovado (ou em vias de) para obter casos reais des-identificados.

---

## Fase 1 — MVP (fígado · RM · modo Pesquisa)

Objetivo: o produto mínimo descrito em `01_VISAO.md`.

Escopo:
- Ingestão de RM com des-identificação (anonimização).
- Segmentação **automática** do fígado.
- Marcação **manual** da lesão no 3D Slicer (+ revisão do órgão).
- Refino, mesh, **STL para download** (órgão + lesão, em LPS).
- **Visualizador web** (girar, zoom, camadas órgão/lesão/transparência).
- Tudo em **modo Pesquisa** (aviso + anonimização).
- Arquivamento das marcações para o **flywheel** (`06_SEGMENTACAO.md`).

**Gate para a Fase 2:** um cirurgião valida que os modelos representam fielmente
casos reais; o fluxo é repetível para um novo caso sem reescrever código.

---

## Fase 2 — Robustez clínica e expansão

Objetivo: ampliar utilidade e começar a reduzir o trabalho manual.

Itens (cada um independente, priorizar por valor):
- **Modelo próprio de lesão** treinado com o acervo do flywheel.
- **Novos órgãos** como perfis (config): **baço**, depois **rim** — sem reescrever
  o motor.
- **Vasculatura do fígado** e noção de segmentos (rumo ao planejamento de
  ressecção).
- **Relatório PDF** inicial (volumetria, descrição de alterações) — versão simples,
  ainda sem atlas normativo.
- Avaliar **TC** e protocolos de RM dedicados para o caminho de precisão sub-mm.

**Gate para a Fase 3:** robustez e validação suficientes para iniciar o caminho de
dispositivo médico.

---

## Fase 3 — Produto clínico

Objetivo: transição formal Pesquisa → Clínico (`03_REGULATORIO_LGPD.md`).

Itens:
- Caminho **ANVISA** (dispositivo médico) e validações documentadas.
- **Pseudonimização** ativa (identificação inequívoca do paciente).
- **Integração PACS/DICOMweb**.
- Servidor interno **multiusuário** (vários cirurgiões).
- Rastreabilidade, versionamento de modelos, registro de revisão/aprovação.

---

## Marcos de longo prazo (fase distante)

- **Atlas normativo** de formas anatômicas → habilita o **relatório comparativo
  completo** (paciente × padrão saudável de mesmo perfil). É um projeto de pesquisa
  próprio, com centenas de exames.
- **Simulação FEA** (tensão/deformação) → exige malha **volumétrica**
  (tetraedralização) e validação biomecânica. Era a "aspiração de fase 2" original;
  na prática é mais pesada e fica para depois da robustez clínica.
- **Casos complexos** (tórax/pulmão, tumores cerebrais) — novos perfis, possivelmente
  novas modalidades e novos cirurgiões validadores.

---

## Princípio do roadmap

Cada salto de ambição (mais órgãos, IA própria, clínico, FEA) deve ser **adicionar
um perfil ou um módulo isolado**, nunca reescrever o que já existe. Se um item
exigir reescrever o motor, isso é sinal de que o desenho da Fase 1 precisa ser
revisto **antes** — não depois.
