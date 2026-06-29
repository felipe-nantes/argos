# 03 · Regulatório, LGPD e Responsabilidade

> **Aviso.** Este documento mapeia o terreno do ponto de vista de **arquitetura de
> software**. Não é aconselhamento jurídico nem regulatório. O rumo clínico exige
> consultoria especializada e a passagem formal pelo CEP/Plataforma Brasil e por
> quem cuidar do regulatório (ANVISA) do projeto. Trate as afirmações abaixo como
> requisitos de desenho a serem confirmados com essas instâncias.

## O princípio que corrige um erro de raciocínio

Houve uma intuição inicial no projeto de que "não precisamos marcar como
não-clínico porque pretendemos escalar para clínico". **Isso é o inverso do
seguro.** O status de um sistema é **o que ele é hoje**, não o que se pretende que
ele seja. Hoje o sistema é **pesquisa**. O rótulo "uso em pesquisa, não destinado
a decisão clínica" é justamente o que **mantém o projeto legal e protegido** até a
aprovação para uso clínico existir.

Por isso o produto tem **dois estados explícitos**, com uma **transição formal**
entre eles — nunca um simples booleano no código.

## Os dois estados do produto

### Estado: PESQUISA (atual — MVP e toda a validação)

- Toda saída (tela, STL, futuros relatórios) carrega **aviso visível** de que se
  destina a pesquisa/educação e **não** a decisão clínica.
- Opera sob **aprovação do CEP** (Comitê de Ética em Pesquisa) via Plataforma
  Brasil, com o uso de dados de paciente sob o protocolo aprovado.
- Dados entram **anonimizados** (ver abaixo).
- Validação por médicos especialistas acontece **dentro** deste estado.

### Estado: CLÍNICO (produto final — futuro)

- Pré-requisito: caminho de **dispositivo médico** com a ANVISA cumprido, além das
  validações universitárias.
- Exige **identificação inequívoca** do paciente dono do modelo (o cirurgião
  precisa saber de quem é) — daí a pseudonimização, não anonimização definitiva.
- Exige rastreabilidade, controle de versão de cada modelo gerado, registro de
  quem revisou/aprovou.

### O gate de transição

Mudar de Pesquisa para Clínico **não é uma flag**. É um marco com pré-condições
verificáveis (aprovações, validação documentada, conformidade ANVISA/LGPD). O
software deve tornar essa transição **deliberada e auditável**, não acidental. Em
termos de arquitetura: o estado é um parâmetro de configuração de alto nível que
governa quais avisos aparecem e quais campos de identificação são exigidos — e
trocá-lo deixa rastro.

## Anonimização × Pseudonimização (decisão de pipeline)

Os cabeçalhos DICOM carregam **PHI** (dados pessoais de saúde): nome, ID, data de
nascimento, instituição, datas, às vezes até foto em campos privados. Isto **não
pode** circular no projeto sem tratamento. A LGPD trata dado de saúde como dado
**sensível**.

Decisão do time, traduzida em desenho:

- **MVP:** **anonimização** — o vínculo com o paciente é cortado. O sistema não
  precisa saber de quem é o exame para gerar e validar o modelo.
- **Produto clínico:** **pseudonimização** — o vínculo é substituído por uma chave;
  uma tabela protegida permite re-vincular paciente↔modelo quando autorizado.

Para que a migração **não seja trabalhosa no futuro** (requisito explícito do
time), o pipeline de privacidade do MVP já é construído com o **ponto de extensão
da chave reservado**: a etapa de "des-identificação" recebe a política como
configuração (`anonymize` vs `pseudonymize`), de modo que habilitar a chave no
futuro seja trocar a política, não reescrever a etapa. Detalhe em `05_PIPELINE.md`.

## O que precisa acontecer na ingestão (requisito firme)

1. **Des-identificar todo DICOM na entrada**, antes de qualquer processamento ou
   armazenamento — remover/limpar os campos de PHI segundo um perfil de
   des-identificação reconhecido (o padrão da área é o perfil de
   *Basic Application Level Confidentiality* do DICOM / DICOM PS3.15).
2. Nunca persistir o exame original com PHI fora de uma zona controlada.
3. Nunca colocar PHI em nome de arquivo, log, URL ou parâmetro.

## Responsabilidade clínica

- No estado **Pesquisa**, a responsabilidade pela interpretação é do médico
  pesquisador sob o protocolo do CEP; a saída é insumo de estudo, não laudo.
- No estado **Clínico**, a responsabilidade pela decisão permanece **do médico** —
  o sistema é ferramenta de apoio. Isso deve estar comunicado de forma explícita
  na interface e na documentação do produto.
- A validação prévia descrita pelo time (comitê de ética, especialistas,
  cirurgiões de renome) é a base dessa transição e deve ser **documentada** —
  vira parte do dossiê regulatório.

## Itens a confirmar com as instâncias certas

- Perfil exato de des-identificação aceito pelo CEP/HU.
- Enquadramento do software como dispositivo médico e a classe de risco na ANVISA.
- Política de retenção e de acesso aos dados (quem, por quanto tempo, onde).
- Termo de consentimento / dispensa para uso de exames retrospectivos.
