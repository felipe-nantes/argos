# 01 · Visão de Produto

## O problema

Cirurgias — sobretudo as complexas — são frequentemente planejadas a partir de
exames de imagem **2D** (cortes de TC/RM olhados fatia a fatia). O cirurgião
precisa reconstruir mentalmente, em 3D, a anatomia e a relação entre a lesão e as
estruturas vitais ao redor. Esse esforço de reconstrução mental é fonte de
incerteza, aumenta o tempo de procedimento e dificulta o pré-operatório.

**A dor que resolvemos:** falta de um planejamento pré-operatório tridimensional e
fiel ao paciente, que aumente a eficiência cirúrgica e reduza o tempo de cirurgia.

## O usuário

O usuário final é o **cirurgião** (no MVP, o cirurgião geral do time, depois os
cirurgiões de renome que farão a validação). No fluxo de produção do modelo
participam também:

- **Aluno de biotecnologia (JM)** — responsável provável pela marcação manual da
  lesão no 3D Slicer e pela checagem anatômica.
- **Engenharia de software (Nantes, Mateus)** — constroem e operam o pipeline.
- **Radiologista** (quando disponível) — referência para validar segmentações.

Observação: o cirurgião **não** deve precisar entender o pipeline. Para ele, a
experiência é "recebi um modelo 3D do fígado deste paciente, com o tumor
destacado, que eu giro, inspeciono e baixo".

## O que a pessoa recebe

**No MVP:**
- Um **modelo 3D do fígado** do paciente, fiel à forma e ao volume reais.
- A **patologia destacada** (tumor/nódulo) em cor distinta, sobreposta ao órgão.
- Visualização **no navegador** (girar, dar zoom, cortar, alternar
  órgão/lesão/transparência).
- **Download do STL** do órgão e da lesão, pronto para impressão 3D.

**No produto final (ver `08_ROADMAP.md`):**
- Relatório em **PDF** comparando o órgão do paciente com um padrão de referência
  para o perfil dele (idade, sexo, antropometria, histórico), descrevendo
  alterações e auxiliando o diagnóstico.
- Vasculatura e planos de ressecção.
- Precisão sub-milimétrica para guia cirúrgico.

## A cunha (por que fígado · RM)

Escolhemos atacar primeiro o caso mais **defensável** e mais **viável**:

- **Domínio do cirurgião geral** — fígado/baço/rim/vísceras estão dentro do que o
  campeão clínico do time opera e pode validar. (Ortopedia, neuro e cardíaco
  estariam fora.)
- **RM é a modalidade acessível ao time** e é apropriada para tecido mole.
- **Fígado é o caso clássico de planejamento 3D** no mundo — máxima credibilidade
  para apresentar aos cirurgiões validadores.
- **Existe modelo pré-treinado de RM** que segmenta o fígado sem precisarmos de
  dados rotulados (ver `06_SEGMENTACAO.md`).

Detalhe importante: o plano original mencionava "radiografia" e "ortopedia". Uma
radiografia 2D **não reconstrói 3D fiel** (a profundidade é perdida na captura);
osso exigiria TC, que o time não tem acesso fácil. Por isso o pivô para
fígado/RM, que **antecipa a fase abdominal** que já estava no plano.

## O que o MVP NÃO é

Deixar isto explícito evita escopo inflado:

- **Não** é um simulador biomecânico (FEA). → Fase 2+.
- **Não** integra com PACS/DICOMweb. → Upload manual de pasta DICOM no MVP.
- **Não** usa IA própria treinada. → Modelo pré-treinado + marcação manual.
- **Não** gera relatório comparativo nem usa atlas de "órgão saudável padrão".
- **Não** entrega precisão sub-milimétrica certificada para guia cirúrgico.
- **Não** é dispositivo médico ainda — opera em **modo Pesquisa**.
- **Não** segmenta automaticamente a patologia — a lesão é marcada por humano.

## Critérios de sucesso do MVP

O MVP é bem-sucedido se:

1. A partir de um exame de RM de fígado, o sistema produz, **de ponta a ponta e
   sem fabricar nada**, um modelo 3D do órgão com a lesão destacada.
2. Um **cirurgião olha o modelo e diz que ele representa fielmente** o que ele vê
   no exame e o que esperaria encontrar no paciente.
3. O cirurgião consegue **girar/inspecionar no navegador e baixar o STL**.
4. O processo é **repetível** para um novo caso sem reescrever código.
5. O fluxo todo respeita o **modo Pesquisa** (anonimização + aviso).

Sucesso aqui **não** é precisão numérica de referência — é utilidade clínica
percebida e robustez do fluxo. A métrica de precisão (Dice) é instrumental, não a
meta de produto nesta fase (ver `06_SEGMENTACAO.md`).
