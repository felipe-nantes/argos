# 10 · Maturidade do Digital Twin

Este módulo formaliza a evolução do projeto **do modelo anatômico (MVP) até um
digital twin de fato**. Ele existe para deixar explícito, para o time e para a
banca, onde estamos na escada, o que falta em cada degrau, e qual gate (dados +
validação + regulatório) controla a subida.

> **Posição honesta.** O termo "digital twin" está no nome do projeto, mas o MVP
> entrega um **modelo digital** (uma fotografia 3D fiel num instante), não ainda um
> gêmeo digital. Isto não é demérito — é o degrau de entrada, sobre o qual o twin é
> construído. O que não pode é ambiguidade sobre onde estamos. Ver "Naming" ao fim.

## O que define um digital twin (e por que o MVP ainda não é um)

Um digital twin de verdade tem três propriedades que um modelo 3D estático não tem:

1. **Comportamento** — ele não só *parece* o órgão; ele *se comporta* como o órgão
   (deforma, responde a forças, a fluxo, a um corte).
2. **Sincronização** — ele *se atualiza* com dados do paciente ao longo do tempo;
   deixa de ser um instante e passa a acompanhar a evolução.
3. **Predição** — ele permite rodar cenários ("e se?") e antecipa como o paciente
   real responderia.

O MVP tem forma fiel, mas é estático, de um instante e não preditivo. Por isso é a
**fundação** do twin, o **Nível 1** da escada abaixo.

---

## A escada de maturidade (4 níveis)

### Nível 1 — Modelo anatômico  *(o MVP)*

- **O que adiciona:** forma 3D fiel do órgão + lesão destacada. O twin *parece* o
  paciente.
- **Capacidades técnicas:** ingestão DICOM + des-identificação, segmentação do
  órgão (automática) + lesão (manual no 3D Slicer), refino, mesh, STL, visualizador
  web. *(Todas já no escopo do MVP — ver `05_PIPELINE.md`.)*
- **Dados necessários:** um exame de RM por paciente (snapshot). Dados públicos
  (CHAOS) para desenvolver + casos retrospectivos do HU. *(Ver `06_SEGMENTACAO.md`.)*
- **Validação:** cirurgião confirma **fidelidade visual** ao caso. Métrica (Dice) é
  instrumental, não a meta. *(Ver `01_VISAO.md`.)*
- **Carga regulatória:** baixa — ferramenta de visualização em modo Pesquisa, com
  anonimização e aviso. *(Ver `03_REGULATORIO_LGPD.md`.)*
- **Gate de entrada:** já estamos nele (Fases 0–1 do `08_ROADMAP.md`).

### Nível 2 — Twin biomecânico

- **O que adiciona:** **comportamento mecânico**. O twin responde a forças —
  deformação de tecido, interação com instrumento, ensaio de corte/ressecção. É o
  "ensaio cirúrgico" sem risco.
- **Capacidades técnicas (novas):**
  - **Tetraedralização** — malha **volumétrica** (tetraedros) em vez da casca oca do
    marching cubes (TetGen, gmsh ou pygalmesh). *Sem isto não há FEA real* — é a
    lacuna já apontada no `05_PIPELINE.md`.
  - **Propriedades de material por tecido** (módulo de Young, Poisson; idealmente
    heterogêneas), inicialmente da literatura.
  - **Solver FEA** (FEBio, SfePy ou equivalente); eventualmente **CFD** para fluxo
    em vasos.
- **Dados necessários:** além do exame, propriedades mecânicas dos tecidos; para a
  validação, dados experimentais/observados de deformação.
- **Validação:** **credibilidade do modelo computacional** — comparar a simulação
  com comportamento real/experimental. É uma disciplina própria (verificação,
  validação e quantificação de incerteza de modelos paciente-específicos).
- **Carga regulatória:** sobe. Um modelo que simula comportamento e informa decisão
  é dispositivo de risco maior que um visualizador.
- **Gate de entrada:** Nível 1 validado **+** capacidade de tetraedralização e FEA
  **+** propriedades de material confiáveis. → Fase 2 do roadmap (era a "aspiração
  fase 2" original).

### Nível 3 — Twin temporal

- **O que adiciona:** **sincronização no tempo**. O twin acompanha a evolução do
  paciente (exames de acompanhamento, labs, sinais). Deixa de ser snapshot.
- **Capacidades técnicas (novas):**
  - **Modelo de dados longitudinal** — o paciente passa a ter *uma série de estados
    no tempo*, não um só.
  - **Registro/alinhamento** entre exames sucessivos (image registration).
  - **Integração de dados não-imagem** (prontuário/EHR, exames laboratoriais) e
    **análise de séries temporais**.
- **Dado regulatório crítico:** o Nível 3 **exige identificação/pseudonimização** —
  você precisa saber que o exame de hoje é do **mesmo** paciente do exame de seis
  meses atrás. Isto ativa o ponto de extensão da chave reservado no
  `03_REGULATORIO_LGPD.md` e empurra o produto para o estado **Clínico**.
- **Dados necessários:** múltiplos exames do mesmo paciente no tempo + dados
  clínicos. *O acervo do flywheel e o relatório comparativo são as sementes disto.*
- **Validação:** o twin reflete corretamente a evolução observada do paciente.
- **Carga regulatória:** alta — dado clínico longitudinal e identificado, LGPD
  pesada, transição formal Pesquisa→Clínico.
- **Gate de entrada:** Nível 1 (idealmente também 2) **+** pseudonimização ativa
  **+** integração de dados clínicos **+** acesso longitudinal a pacientes. → Fase 3
  do roadmap.

### Nível 4 — Twin preditivo  *(fronteira de pesquisa)*

- **O que adiciona:** **predição**. Simula cenários futuros e contrafactuais: "e se
  eu ressecar aqui?", "como o tumor vai progredir?", "qual a resposta esperada a
  este tratamento?".
- **Capacidades técnicas (novas):**
  - **Modelos preditivos** (ex.: crescimento tumoral, resposta terapêutica) —
    mecanísticos e/ou de IA.
  - Possível **integração intraoperatória em tempo real** (navegação, imagem
    intraoperatória).
  - Possível **modelagem multiescala / multi-ômica** (integrar genômica, etc.).
- **Dados necessários:** tudo dos níveis anteriores **+ desfechos (outcomes)** para
  treinar e validar a predição; possivelmente dados moleculares.
- **Validação:** a predição precisa **bater com o desfecho real** — o nível mais
  difícil de validar, exige estudos **prospectivos**.
- **Carga regulatória:** a mais alta — um sistema que prevê e influencia a conduta
  cirúrgica é dispositivo de **alto risco**, com exigência de validação clínica
  prospectiva.
- **Gate de entrada:** níveis anteriores maduros **+** dados de desfecho **+**
  validação prospectiva **+** estrutura regulatória robusta. → Visão de longo prazo.

---

## Visão consolidada

| Nível | Propriedade-chave | Fase (roadmap) | Gate principal | Risco regulatório |
|---|---|---|---|---|
| **1 · Anatômico** | Parece | Fase 1 (MVP) | Cirurgião valida fidelidade | Baixo |
| **2 · Biomecânico** | Comporta-se | Fase 2 | Tetraedralização + FEA + propriedades | Médio |
| **3 · Temporal** | Sincroniza | Fase 3 | Pseudonimização + dados longitudinais | Alto |
| **4 · Preditivo** | Prevê | Longo prazo | Desfechos + validação prospectiva | Muito alto |

---

## Como cada nível pluga sem reescrever

O princípio do projeto (`04_ARQUITETURA.md`) se mantém em toda a escada: **subir de
nível é adicionar uma camada, nunca reescrever o motor.**

- **Nível 2** entra como **um estágio novo depois da malha** (tetraedralização →
  FEA), consumindo a mesma máscara/mesh que o MVP já produz.
- **Nível 3** entra como uma **dimensão a mais no modelo de dados** (tempo) e novos
  conectores de dados clínicos; o pipeline por exame não muda — passa a ser chamado
  por estado temporal.
- **Nível 4** entra como **modelos sobre** o twin biomecânico + temporal.

A fronteira de API (navegador↔pipeline) e os perfis em config absorvem cada camada.
Se um nível exigir reescrever o núcleo, é sinal de que o desenho da fase anterior
precisa ser revisto **antes** de subir.

---

## Verdades incômodas (disciplina de escopo)

1. **O valor do MVP não depende de virar twin.** A redução de tempo de cirurgia e o
   melhor pré-operatório já vêm do **Nível 1**. Subir a escada é um programa de
   pesquisa de anos — trate o twin completo como **norte**, não como extensão do
   MVP. Não deixe a palavra "twin" sabotar o MVP enxuto.
2. **Cada degrau multiplica três custos ao mesmo tempo:** computacional (FEA, CFD e
   predição são pesados), de validação (de "fidelidade visual" a "predição confere
   com desfecho prospectivo") e regulatório (de visualizador a dispositivo de alto
   risco).
3. **O campo ainda é majoritariamente pesquisa.** A literatura de 2025 é consistente
   sobre o potencial transformador, mas igualmente sobre a **desconexão entre a
   inovação e a aplicação clínica real** e sobre as demandas computacionais. Subam
   degrau por degrau, com gate, sem pular para a promessa preditiva.

---

## Naming e posicionamento por nível

Para manter a honestidade intelectual (e sobreviver a uma banca/revisor), alinhe o
nome ao degrau real:

- **Hoje (Nível 1):** posicione como *"base anatômica de um digital twin
  cirúrgico"* ou *"digital twin — nível anatômico (fase 1)"*. Manter "Digital Twin
  Cirúrgico" como **visão-norte** é legítimo, desde que fique explícito que o
  entregável atual é a fundação.
- **A cada subida**, o sistema *ganha o direito* a mais do termo: comporta-se
  (N2) → sincroniza (N3) → prevê (N4). Aí "digital twin" deixa de ser aspiração e
  passa a ser descrição.

---

## Princípio de evolução

Gerar o MVP funcional (Nível 1) e **evoluí-lo conforme a capacidade do time**,
subindo um degrau por vez, cada um liberado por seu gate de dados + validação +
regulatório. Nenhum salto de nível deve quebrar o que já funciona; cada nível é uma
camada nova sobre uma base já validada.

---

## Referências (para fundamentação acadêmica)

Material recente sobre digital twins em medicina/cirurgia, útil para a banca:

- *Current progress of digital twin construction using medical imaging* (Zhao,
  2025, **J. Appl. Clin. Med. Phys.**) — imagem como base do twin; FEA/CFD e séries
  temporais como técnicas-chave.
  https://aapm.onlinelibrary.wiley.com/doi/10.1002/acm2.70226
- *Digital twins: pioneering personalized precision in modern surgery* (2025, **PMC**)
  — twin cirúrgico = réplica com dados em tempo real + modelagem preditiva.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12578008/
- *Digital twins in healthcare: a comprehensive review and future directions*
  (2025, **Frontiers in Digital Health**) — ensaio de procedimentos em modelos
  paciente-específicos; experimentação sem risco.
  https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1633539/full
- *Digital twins in oncology: from predictive modelling to personalised treatment*
  (2026, **ScienceDirect**) — simulação dinâmica de tumor e resposta a tratamento
  (relevante para a ambição de tumores).
  https://www.sciencedirect.com/science/article/pii/S1040842826000582
- *Digital Twins in Personalized Medicine: Bridging Innovation and Clinical Reality*
  (2025, **MDPI/PMC**) — a desconexão entre inovação e aplicação clínica real.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12653454/

> As referências acima foram levantadas em pesquisa de junho/2026; confirme os
> dados bibliográficos completos (autores, volume, páginas) na fonte antes de citar
> formalmente em trabalho acadêmico.
