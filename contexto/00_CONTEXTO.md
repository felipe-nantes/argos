# Contexto do Projeto — Digital Twin Cirúrgico (UEM · GETS · HU)

> **Camada de orientação.** Este arquivo é o ponto de entrada. Ele não contém a
> estratégia inteira — ele resume o essencial e aponta para o módulo certo.
> Leia este arquivo primeiro; abra apenas o módulo que a sua tarefa exige.
> Os `.md` deste diretório são **orientação para humanos e agentes**, nunca uma
> camada de enforcement. A regra que precisa ser cumprida pelo sistema mora em
> **config versionada** (ver `04_ARQUITETURA.md`), não em prosa e não no código.

---

## O que é o produto (em três linhas)

Uma ferramenta de **planejamento cirúrgico** que transforma um exame de imagem
(RM no MVP) em um **modelo 3D fiel do órgão do paciente, com a patologia
destacada**, visualizável no navegador e exportável em STL. Desenvolvida na UEM
junto ao GETS, com acesso ao Hospital Universitário e validação por cirurgiões.

## Fase atual

**Fase 1 — MVP.** Órgão-alvo: **fígado**. Modalidade: **RM**. Execução: **estação
de trabalho única**, app web local. Status regulatório: **Pesquisa** (ver
`03_REGULATORIO_LGPD.md`). **Nível de maturidade: 1 — modelo anatômico** (a
fundação de um digital twin; ver `10_MATURIDADE_DIGITAL_TWIN.md`). Tudo o que é
"produto final" (uso clínico, ANVISA, PACS, FEA, relatório PDF) está adiado e
mapeado no `08_ROADMAP.md`.

## As cinco regras de ouro (inegociáveis)

1. **Nunca fabricar dado clínico.** Se um estágio falha (ex.: segmentação não
   carrega), o pipeline **aborta com erro explícito**. Jamais gerar máscara
   aleatória, jamais "seguir mesmo assim". Esta regra existe porque o script
   original fazia exatamente o contrário.
2. **Regra de domínio mora em config versionada**, não no código. Trocar de órgão
   = adicionar um arquivo de perfil, não reescrever o motor.
3. **O estado "Pesquisa" sempre carrega o aviso** de que a saída não se destina a
   decisão clínica. A transição para "Clínico" é um gate formal, não um booleano.
4. **Todo dado de paciente entra anonimizado.** A arquitetura já reserva o lugar
   da chave de pseudonimização para o uso clínico futuro, mas o MVP não a usa.
5. **A saída automática nunca é confiada às cegas.** Sempre há revisão humana
   (cirurgião / radiologista) antes de o modelo ser considerado válido.

## Mapa dos módulos

| Arquivo | O que responde | Quando abrir |
|---|---|---|
| `01_VISAO.md` | Produto, usuário, dor, o que o MVP **não** é | Decisões de escopo e prioridade |
| `02_DOMINIO_CLINICO.md` | Fígado, RM, fidelidade, lesão, expansão de órgãos | Decisões clínicas e anatômicas |
| `03_REGULATORIO_LGPD.md` | Pesquisa→Clínico, anonimização, CEP, responsabilidade | Qualquer coisa com paciente real |
| `04_ARQUITETURA.md` | Pipeline órgão-agnóstico, perfis plugáveis, stack | Desenho técnico, novos órgãos |
| `05_PIPELINE.md` | Os estágios ponta a ponta e seus gates de segurança | Implementação do fluxo |
| `06_SEGMENTACAO.md` | Órgão automático + lesão manual, dados, validação | Tudo que envolve segmentar |
| `07_INFRA_CUSTOS.md` | Hardware, deploy local, custo, origem dos exames | Infra e orçamento |
| `08_ROADMAP.md` | Fase 0→3, gates entre fases | Planejamento temporal |
| `09_NEGOCIO.md` | Modelos de receita e sustentação (em aberto) | Estratégia de longo prazo |
| `10_MATURIDADE_DIGITAL_TWIN.md` | Escada de 4 níveis: modelo anatômico → twin preditivo | Visão de evolução até virar um twin de fato |

## Time

- Engenharia de Software: Nantes, Mateus
- Biotecnologia: JM (provável responsável pela marcação manual de lesão)
- Orientação clínica: Prof. Dr. (Anestesista) e Prof. Dr. (Cirurgião Geral)
- Validação: cirurgiões de renome (ver `01_VISAO.md`)
