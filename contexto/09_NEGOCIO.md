# 09 · Negócio e Sustentação

> Status: **em aberto.** O time ainda não definiu o modelo de receita. Este módulo
> mapeia opções e questões para decidir mais adiante — não fixa uma direção.

## Contexto que molda as opções

O projeto nasce **dentro da universidade** (UEM/GETS) e do **Hospital
Universitário**, com orientação acadêmica e clínica. Isso traz vantagens (acesso a
dados, validação por especialistas, credibilidade, possíveis fomentos) e
particularidades (propriedade intelectual em ambiente acadêmico, finalidade
pública do HU, exigências éticas). A definição de negócio precisa conversar com
essas condições, não ignorá-las.

## Opções de modelo (a avaliar)

- **Pesquisa/fomento + open source.** Sustentar via editais, bolsas e parcerias;
  liberar o núcleo aberto. Coerente com a origem acadêmica; monetização indireta
  (serviços, consultoria, publicações).
- **SaaS por assinatura.** Hospitais/serviços pagam recorrente pelo acesso. Implica
  sair do "estação única" para servidor/nuvem e maturidade clínica — fases
  avançadas.
- **Cobrança por exame/modelo.** Paga-se por caso processado. Alinha custo a uso;
  exige operação confiável e suporte.
- **Licença para hospitais/fabricantes.** Licenciar o sistema (ou a tecnologia)
  para instituições ou fabricantes de dispositivos/impressão 3D.
- **Open-core.** Núcleo aberto + recursos avançados (relatório, integrações,
  multiusuário, suporte) pagos.

Nenhuma é excludente; é comum começar como pesquisa/open e migrar conforme o
produto amadurece e o caminho regulatório avança.

## Questões a resolver antes de fixar o modelo

- **Propriedade intelectual:** de quem é a IP gerada (universidade, alunos,
  hospital)? Há política institucional de inovação/transferência de tecnologia?
- **Finalidade do HU:** o uso dentro de um hospital público tem implicações sobre
  cobrança e acesso.
- **Caminho regulatório:** só faz sentido cobrar por uso clínico **depois** da
  ANVISA (`03_REGULATORIO_LGPD.md`); antes disso, a sustentação é de pesquisa.
- **Quem mantém:** sustentação de longo prazo exige time/manutenção — isso pesa na
  escolha entre open source (comunidade/fomento) e produto comercial (receita).

## Validação e primeiros usuários

A validação inicial é com **cirurgiões de renome** (via os professores doutores do
time). Eles são, ao mesmo tempo, validadores clínicos e os primeiros "clientes" de
prova — o feedback deles define se há valor que justifique qualquer modelo de
receita. **Provar utilidade clínica vem antes de decidir como cobrar.**

## Recomendação de sequência

1. **Agora:** não decidir modelo de receita; focar em provar valor clínico no MVP
   sob financiamento/estrutura de pesquisa.
2. **Após validação do MVP:** mapear IP e política de inovação da universidade.
3. **Rumo ao clínico:** escolher o modelo (provavelmente um híbrido open-core +
   licença/serviço para instituições), condicionado ao regulatório.
