# 05 · Pipeline (Estágios e Gates)

Descreve cada estágio do pipeline, seu contrato (entra/sai) e seus **gates de
segurança**. A filosofia geral: **cada estágio valida sua própria entrada e
aborta com erro explícito se algo estiver errado.** Nunca fabricar, nunca
"seguir mesmo assim".

> Por que essa ênfase: o script original, quando a IA não carregava, gerava uma
> **máscara aleatória** e seguia construindo modelo, exportando e visualizando,
> com apenas um aviso. Em contexto cirúrgico isso produziria um "modelo do
> paciente" que não tem relação alguma com o paciente. **Esse comportamento está
> proibido** (regra de ouro nº 1).

---

## (1) Ingestão + des-identificação

**Entra:** pasta com série DICOM de RM.
**Sai:** volume normalizável + geometria correta + metadados des-identificados.

O que faz:
- Lê a série com **SimpleITK** (que já monta o volume na ordem e geometria
  corretas — não reaplicar manualmente slope/intercept; não reescrever spacing).
- **Des-identifica** os cabeçalhos segundo a política do perfil/projeto (ver
  `03_REGULATORIO_LGPD.md`): no MVP, `anonymize`; ponto de extensão para
  `pseudonymize` reservado.
- Preserva **spacing/origem/orientação** corretos (essencial para o modelo 3D não
  sair distorcido).

Gates:
- Pasta vazia / sem DICOM válido → **abortar**.
- Série inconsistente (fatias faltando, geometria incoerente) → **abortar com
  diagnóstico**, não interpolar silenciosamente.
- PHI não removida → **abortar** (nunca prosseguir com dado identificável).

Erros do script original corrigidos aqui:
- *Slope/intercept aplicado em dobro* → o reader já aplica; não multiplicar de
  novo. (No MVP é RM, então a janela de HU nem se aplica — ver estágio 2.)
- *Ordem de eixos do spacing trocada (x↔z)* → usar a geometria que o SimpleITK
  fornece, sem inverter à mão.

---

## (2) Normalização

**Entra:** volume.
**Sai:** volume normalizado conforme a modalidade do perfil.

O que faz:
- Para **RM** (caso do MVP): **z-score** (subtrai média, divide por desvio). RM
  não tem escala física absoluta como a HU da TC, então janela de HU não se
  aplica.
- A política de normalização vem do **perfil** (`normalizacao: zscore`).

Gates:
- Volume constante / desvio ~0 (exame corrompido) → **abortar**.

Nota: o motor de segmentação automática faz a **própria** normalização interna.
A normalização aqui serve para visualização/inspeção e para etapas que dependam de
intensidade; não duplicar a normalização do modelo dentro do que vai para ele.

---

## (3) Segmentação do ÓRGÃO (automática)

**Entra:** volume.
**Sai:** máscara do órgão (ex.: fígado).

O que faz:
- Roda **TotalSegmentator MRI** (ou alternativa, ver `06_SEGMENTACAO.md`).
- Extrai da saída **apenas o rótulo do perfil** (`rotulo_alvo: liver`).

Gates (críticos):
- Modelo não instalado / falha ao carregar → **ABORTAR**. **Jamais** gerar máscara
  aleatória ou placeholder. Esta é a correção mais importante do projeto.
- Órgão-alvo ausente na saída (modelo não encontrou fígado) → **abortar** e
  reportar, para revisão humana.

---

## (4) Segmentação da LESÃO (manual / semi-automática — 3D Slicer)

**Entra:** volume + máscara do órgão.
**Sai:** máscara da lesão + máscara do órgão revisada.

O que faz:
- Abre no **3D Slicer** para um operador qualificado (biotecnologia / radiologista):
  - **revisar e corrigir** a máscara automática do órgão;
  - **marcar a lesão** com *threshold*, *region growing*, *Grow from seeds* ou
    pincel.
- Devolve as máscaras finais ao pipeline.

Gates:
- Sem revisão humana confirmada → o modelo **não** é marcado como válido (regra de
  ouro nº 5: saída automática nunca é confiada às cegas).

Flywheel: cada lesão marcada aqui é **arquivada como dado rotulado** para treinar
o modelo de lesão da fase 2 (ver `06_SEGMENTACAO.md`).

---

## (5) Refino das máscaras

**Entra:** máscaras (órgão + lesão).
**Sai:** máscaras limpas.

O que faz:
- **Opening/closing** (morfologia) para remover ruído e fechar buracos.
- Remove componentes conexos menores que `min_volume_voxels` (do perfil).

Gates:
- Refino que apaga a estrutura inteira (parâmetro mal calibrado) → **abortar** em
  vez de devolver máscara vazia.

Erro do script original corrigido: usar **`footprint=`** (API atual do
scikit-image), não `selem=` (removido).

---

## (6) Geração de mesh

**Entra:** máscaras limpas + geometria (spacing correto).
**Sai:** meshes 3D (órgão e lesão).

O que faz:
- **Marching cubes** gera a superfície; **suavização** preservando bordas.
- Anexa metadados de exibição do perfil (cores).
- (Propriedades físicas de material só fazem sentido na fase FEA — ver
  `08_ROADMAP.md`. No MVP não são necessárias.)

Gates:
- Spacing incorreto → modelo distorcido. Garantir que a geometria do estágio (1)
  flui até aqui sem inversão de eixos.

Nota técnica para a fase FEA: marching cubes gera **superfície** (casca oca). FEA
de deformação precisa de malha **volumétrica** (tetraedros) — passo de
tetraedralização (TetGen/gmsh) que **não** existe no MVP e fica para fase 2.

---

## (7) Exportação STL + Publicação web

**Entra:** meshes.
**Sai:** arquivos STL para download + modelo no visualizador.

O que faz:
- Exporta **STL** do órgão e da lesão no sistema de coordenadas **LPS** (padrão de
  impressão/Slicer). Converter RAS→LPS quando aplicável (inverter X e Y) para o
  modelo impresso não sair **espelhado** — erro perigoso em contexto cirúrgico.
- Publica os meshes para o **visualizador web**.

Gates:
- Falha de escrita do STL → **abortar** com erro claro (não terminar em silêncio).

Erros do script original corrigidos:
- *`pv.save_mesh_as` não existe* → usar `mesh.save(...)`.
- *`mesh.faces.reshape(-1, 3)`* embaralhava as faces (o PyVista usa `[3,i,j,k,...]`)
  → tratar o contador de vértices corretamente.
- *Exportar `.feb` via Trimesh* não funciona (Trimesh não escreve FEBio) → FEA fica
  para fase 2, com ferramenta adequada.

---

## Observações transversais

- **Sem visualizador travando o processo:** o script original abria uma janela
  PyVista bloqueante no fim. No MVP, a visualização é o **app web**, desacoplado do
  processamento; o pipeline termina ao publicar os artefatos.
- **Contrato entre estágios:** cada estágio recebe e devolve artefatos bem
  definidos (volume, máscara, mesh), o que permite testar e substituir estágios
  isoladamente.
- **Determinismo:** mesmo input + mesma config + mesma marcação humana ⇒ mesmo
  output.

---

## Fluxo opcional pós-`prepare`: triagem visual MedGemma

Após o fígado ser segmentado/revisado, um fluxo paralelo pode gerar uma montagem
2D da RM com **somente o contorno hepático** e solicitar ao MedGemma uma hipótese
visual em modo Pesquisa. Ele não recebe máscara de lesão, não altera os sete
estágios acima e não produz diagnóstico. A saída aceita somente `POSITIVA`,
`NEGATIVA` ou `INCONCLUSIVA`, sempre pendente de revisão humana. Contrato, gates e
uso: `11_MEDGEMMA_SCREENING.md`.
