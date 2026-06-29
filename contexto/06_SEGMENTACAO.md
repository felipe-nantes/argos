# 06 · Segmentação

## A divisão fundamental: anatomia × patologia

A decisão mais importante de segmentação do projeto:

| O quê | Como (MVP) | Por quê |
|---|---|---|
| **Órgão** (anatomia saudável) | **Automático** — modelo pré-treinado | Modelos prontos cobrem isso sem rotular |
| **Lesão** (patologia) | **Manual/semi-auto no 3D Slicer** | Nenhum modelo pronto faz patologia; sem dados para treinar |

## Órgão: TotalSegmentator MRI (motor padrão do MVP)

Por que este motor:
- É um **nnU-Net pré-treinado, aberto e gratuito** que segmenta dezenas de
  estruturas em **qualquer sequência de RM**, com casos de uso que incluem
  explicitamente planejamento cirúrgico.
- Cobre os órgãos do roteiro do time (fígado, baço, rim, etc.).
- **Não exige nenhum dado rotulado** do time para começar — resolve diretamente o
  problema de "não temos muitos dados rotulados".

Como funciona, em uma frase: instala-se o modelo, aponta-se para o exame de RM, e
ele devolve as máscaras das estruturas automaticamente. O perfil do órgão
(`04_ARQUITETURA.md`) só precisa dizer **qual rótulo** extrair (ex.: `liver`).

Alternativas equivalentes (caso a cobertura/qualidade peça): **MRSegmentator** e
**MRISegmentator-Abdomen** são modelos abertos no mesmo nível para RM abdominal. A
arquitetura de perfis permite trocar o motor sem mexer no pipeline (`motor:` no
perfil).

Limite importante: este modelo segmenta **anatomia saudável**, **não** patologia.
Por isso a lesão é tratada à parte.

## Lesão: marcação manual/semi-automática no 3D Slicer (MVP)

Como o time quer a patologia destacada desde o MVP, mas não há modelo pronto nem
dados para treinar, a lesão é marcada por um humano qualificado (biotecnologia /
radiologista) dentro do **3D Slicer**, que já está no loop para revisão
(`05_PIPELINE.md`, estágio 4). Ferramentas típicas:

- **Threshold** — separa por intensidade.
- **Region growing / Grow from seeds** — cresce a região a partir de sementes.
- **Paint/Erase** — ajuste fino com pincel.

A saída é a máscara da lesão, que volta ao pipeline para virar mesh e STL próprios,
sobrepostos ao órgão em cor distinta.

## O flywheel de dados (responde à pergunta dos "dados rotulados")

"Dado rotulado" é um exame em que um humano já marcou, voxel a voxel, o que é o
quê (ex.: "isto é tumor"). É o que se precisa para **treinar** uma IA de patologia.
Hoje o time **não tem** isso em quantidade — por isso a lesão é manual.

O ponto estratégico: **cada lesão marcada no MVP é, por definição, um dado
rotulado.** Guardando essas marcações de forma organizada (par exame↔máscara), o
time **acumula o conjunto de treino** que hoje não tem. Esse acervo é o combustível
para, na fase 2, treinar um **modelo próprio de lesão** (provavelmente também
nnU-Net) ou adotar um modelo mais complexo. O trabalho manual não é dívida — é
investimento que se paga em IA.

Requisito de desenho decorrente: a etapa de marcação deve **arquivar
sistematicamente** (volume des-identificado + máscara de lesão + metadados
mínimos), pensando já no treino futuro.

## Dados para desenvolver agora (destrava o "não sei" sobre exames)

Enquanto a coleta de casos reais do HU passa pelo CEP (ver `08_ROADMAP.md`, Fase
0), o pipeline pode ser **desenvolvido e validado contra dados públicos de RM de
fígado** — por exemplo o **CHAOS** (dataset público de RM abdominal usado como
referência na própria literatura desses modelos). Isso permite construir tudo sem
esperar o dado clínico.

Ressalva: datasets públicos de RM costumam ter **anatomia saudável**, não lesão
(o CHAOS, por exemplo, é de órgãos saudáveis). Ou seja, dá para validar a
segmentação de **órgão** com dado público, mas a **lesão** dependerá de casos
reais — o que reforça por que a marcação manual e o flywheel são centrais.

## Validação (Dice e revisão humana)

Como medir se a segmentação está boa:

- **Dice** (coeficiente de sobreposição) entre a máscara gerada e uma referência —
  útil quando há referência (dado público anotado ou marcação de radiologista).
  Como nota de calibragem de expectativa: o TotalSegmentator MRI reporta Dice na
  casa de ~0,84 em média no seu próprio teste interno, com órgãos grandes (fígado,
  baço, rim) tendendo a sair acima disso.
- **Revisão humana** — no MVP, o critério de produto **não** é o Dice, e sim o
  cirurgião confirmar que o modelo representa fielmente o caso (`01_VISAO.md`). O
  Dice é instrumental (monitorar regressões, comparar motores), não a meta.

## Roteiro de evolução da segmentação

1. **MVP:** órgão automático (TotalSegmentator MRI) + lesão manual (Slicer).
2. **Fase 2:** com o acervo do flywheel, treinar modelo próprio de **lesão**;
   adicionar perfis de baço/rim (só config); avaliar vasculatura do fígado.
3. **Fase 3+:** modelos mais complexos conforme a necessidade e o volume de dados.
