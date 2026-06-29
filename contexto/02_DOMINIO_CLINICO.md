# 02 · Domínio Clínico

## Órgão-alvo do MVP: fígado

O fígado é o carro-chefe por razões clínicas e técnicas (ver `01_VISAO.md`). O que
importa do ponto de vista de **planejamento cirúrgico**:

- O fígado tem uma **anatomia segmentar** (segmentos de Couinaud) definida pela
  distribuição dos vasos (veias hepáticas e ramos portais). A conduta cirúrgica
  (ressecção, segmentectomia, margem) depende de **onde a lesão está em relação a
  esses vasos**.
- Por isso, a versão completa do produto precisará, no futuro, **segmentar a
  vasculatura** — mas isso é difícil em RM e fica para fase posterior.
- **No MVP**, o valor já existe sem vasos: mostrar o **contorno fiel do fígado** e
  a **posição/volume da lesão** dentro dele já orienta o pré-operatório.

> Decisão de escopo: MVP = parênquima do fígado + lesão. Vasos e segmentos de
> Couinaud = fase 2 (ver `08_ROADMAP.md`).

## Modalidade de imagem

**MVP: somente RM (ressonância magnética).**

- RM é a modalidade de **tecido mole** — apropriada para fígado, baço, rim e suas
  lesões.
- É a modalidade **acessível ao time** neste momento.
- O motor de segmentação automática de órgão escolhido funciona em **qualquer
  sequência de RM** (ver `06_SEGMENTACAO.md`).

**Por que não TC no MVP:** TC seria ideal para osso e para vasos com contraste,
mas (a) o time não tem acesso fácil e (b) o caso de osso/ortopedia foi
descartado. TC retorna na fase do guia sub-milimétrico.

**Por que não as duas no MVP:** suportar TC+RM no dia 1 dobra a complexidade de
ingestão e normalização sem ganho para o caso do fígado. A arquitetura de perfis
(ver `04_ARQUITETURA.md`) permite adicionar TC depois sem retrabalho.

## Níveis de fidelidade

Distinção que governa expectativas e roadmap:

| Nível | Para quê | Modalidade adequada | Fase |
|---|---|---|---|
| **Visualização clara** | Inspecionar, comunicar, imprimir modelo de estudo | RM comum | **MVP** |
| **Sub-milimétrica / guia cirúrgico** | Guia físico usado dentro da cirurgia | TC ou protocolos de RM específicos | Fase 2+ |

**Verdade incômoda sobre RM e "fiel ao paciente":** RM dá excelente forma e volume
de órgão, mas tem **distorção geométrica**, **resolução pior entre fatias** (slices
grossos, às vezes com *gap*) e bordas menos nítidas em estruturas finas. Para o MVP
de visualização isso é perfeitamente aceitável. Para a meta futura de guia
sub-mm, **não se deve prometer precisão sub-milimétrica em cima de RM comum** — essa
meta puxa TC ou aquisições de RM dedicadas.

## A patologia (lesão)

Requisito do MVP: a lesão (tumor/nódulo) **destacada** sobre o órgão.

Restrição real: nenhum modelo pré-treinado segmenta patologia, e o time não tem
dados rotulados para treinar um. Logo, **no MVP a lesão é marcada manualmente /
semi-automaticamente no 3D Slicer** por um humano qualificado (biotecnologia /
radiologista), usando ferramentas como *threshold*, *region growing* e pincel.
Detalhes em `06_SEGMENTACAO.md`.

Consequência estratégica (o "flywheel"): cada lesão marcada manualmente vira um
**dado rotulado**. Acumulando essas marcações, o time constrói o conjunto de treino
que hoje não tem — combustível para o modelo próprio de lesão na fase 2.

## Expansão de órgãos (sem reescrever)

Ordem provável de crescimento, cada um entrando como **um novo perfil** (config),
não como código novo:

1. **Fígado** (MVP) — carro-chefe.
2. **Baço** — era o exemplo original do time; cirurgia geral; bem coberto por
   segmentação automática.
3. **Rim** — partial nephrectomy é caso clássico de planejamento 3D.
4. **Tórax/pulmão** (nódulos), depois casos complexos (ex.: tumores cerebrais).

O salto para **cérebro** envolve outra modalidade dominante, outros modelos e outro
cirurgião validador — é um perfil possível, mas de fase distante.

## O relatório comparativo (visão futura)

O exemplo do time (RM de baço com nódulo → comparar com baço saudável de um
paciente padrão de mesmo perfil) é uma meta de **fase distante**, porque exige um
**atlas estatístico/normativo** de formas anatômicas — um banco de referência que
não existe pronto e é, por si só, um projeto de pesquisa com centenas de exames.
Está registrado no `08_ROADMAP.md` como marco de longo prazo, não como item
próximo do MVP.
