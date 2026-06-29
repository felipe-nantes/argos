# 04 · Arquitetura

Este é o módulo central. Ele descreve **como o sistema é organizado para crescer
sem reescrita** — o requisito que o time colocou de forma explícita.

## Princípio diretor: núcleo determinístico + perfis em config

A arquitetura separa duas coisas que normalmente se misturam e apodrecem juntas:

- **O motor (core):** o código que executa o pipeline. É **órgão-agnóstico** e
  **determinístico** — dado o mesmo input e a mesma config, produz o mesmo output.
  Ele não sabe o que é "fígado". Ele sabe executar estágios.
- **Os perfis (config versionada):** arquivos que descrevem **o que** processar e
  **com quais parâmetros**. "Fígado", "baço", "rim" são perfis. Trocar de órgão =
  adicionar um arquivo de perfil. **Nunca tocar no motor.**

Isto é o mesmo princípio já adotado pelo time em outros projetos (regras em config
versionada, nunca no código; motor determinístico). Aqui ele é o que impede que
cada órgão novo vire um fork.

## Anatomia de um perfil de órgão

Um perfil é um arquivo declarativo (YAML/JSON, versionado). Conceitualmente:

```yaml
# perfis/figado.yaml  (exemplo conceitual — não é a implementação final)
id: figado
nome_exibicao: "Fígado"
modalidade: [MRI]                 # MVP: só RM
segmentacao_orgao:
  motor: totalsegmentator_mri     # ver 06_SEGMENTACAO.md
  rotulo_alvo: liver              # estrutura a extrair da saída do modelo
  normalizacao: zscore            # RM usa z-score, não janela de HU
segmentacao_lesao:
  modo: manual_slicer             # MVP: marcação humana no 3D Slicer
  ferramentas_sugeridas: [threshold, region_growing, paint]
refino:
  min_volume_voxels: 300          # remove fragmentos
  suavizacao_iteracoes: 30
mesh:
  nivel_marching_cubes: 0.5
  cor_orgao: "tan"
  cor_lesao: "red"
exportacao:
  stl_orgao: true
  stl_lesao: true
  sistema_coordenadas: LPS        # padrão de impressão/Slicer
visualizacao:
  camadas: [orgao, lesao, transparencia]
```

Adicionar **baço** = criar `perfis/baco.yaml` com `rotulo_alvo: spleen`. O motor
não muda. Esse é o coração da promessa de "não reescrever os módulos".

## Estágios do pipeline (visão de blocos)

Detalhe completo, com gates de segurança, em `05_PIPELINE.md`. Em blocos:

```
[DICOM RM]
   │  (1) Ingestão + des-identificação
   ▼
[Volume + metadados]
   │  (2) Normalização (por perfil: z-score p/ RM)
   ▼
[Volume normalizado]
   │  (3) Segmentação do ÓRGÃO  ── automática (TotalSegmentator MRI)
   ▼
[Máscara do órgão]
   │  (4) Segmentação da LESÃO ── manual/semi-auto no 3D Slicer
   ▼
[Máscara órgão + lesão]
   │  (5) Refino (morfologia, remoção de fragmentos)
   ▼
[Máscaras limpas]
   │  (6) Geração de mesh (marching cubes + suavização)
   ▼
[Meshes 3D]
   │  (7) Exportação STL (LPS)  +  Publicação para o visualizador web
   ▼
[STL para download]  +  [Modelo no navegador]
```

O 3D Slicer entra no estágio (4) como **ambiente de revisão e edição humana**:
recebe o volume e a máscara automática do órgão, o operador corrige o que for
preciso e **marca a lesão**, devolvendo as máscaras finais ao pipeline.

## Stack do MVP

Escolhas guiadas por: estação única, RM, custo de operação baixo, ferramentas
maduras e abertas.

| Camada | Ferramenta | Papel |
|---|---|---|
| Leitura de imagem | **SimpleITK** + **pydicom** | Ler série DICOM, metadados, geometria |
| Des-identificação | **pydicom** (perfil DICOM PS3.15) | Remover PHI na entrada |
| Formato intermediário | **NIfTI** (`nibabel`) | Volume e máscaras entre estágios |
| Segmentação de órgão | **TotalSegmentator MRI** (nnU-Net) | Órgão automático, sem rotular |
| Edição / lesão | **3D Slicer** | Revisão humana + marcação de lesão |
| Processamento de máscara | **NumPy/SciPy/scikit-image** | Refino, morfologia, rotulagem |
| Geração de mesh | **scikit-image (marching cubes)** + **PyVista** | Voxels → superfície 3D |
| Exportação | **PyVista/Trimesh** (STL) | STL em LPS |
| Backend web | **FastAPI** (ou similar leve) | Orquestra o pipeline, serve a API |
| Visualizador web | **VTK.js / Three.js** (mesh) ou **Niivue** (volume) | 3D no navegador |

Notas de stack:

- **Niivue** é um visualizador médico de navegador (NIfTI/DICOM, WebGL) feito
  exatamente para este tipo de dado; bom candidato para mostrar volume+máscara. Para
  mostrar o **mesh** (STL) com órgão e lesão como objetos, VTK.js ou Three.js
  servem. A escolha final depende de se a tela principal é volume ou mesh — decidir
  na implementação do visualizador.
- A **GPU** é necessária para a inferência do nnU-Net em tempo razoável (ver
  `07_INFRA_CUSTOS.md`).

## Topologia de execução

**MVP: app web local em estação única.** Um backend roda na própria workstation; o
navegador (na mesma máquina ou na rede local do HU) acessa a interface. Sem nuvem,
sem dado saindo da máquina. Isto satisfaz a privacidade e zera custo de nuvem.

A fronteira para escalar (servidor interno multiusuário, e depois PACS) já fica
isolada: a comunicação navegador↔pipeline passa por uma **API**, então promover o
backend de "local" para "servidor interno" no futuro não reescreve o pipeline —
só muda onde ele roda e como autentica. Ver `08_ROADMAP.md`.

## O que NÃO entra no código (mora em config ou é externo)

- Parâmetros clínicos por órgão → **perfis** (config versionada).
- Política de privacidade (anonymize/pseudonymize) → **config**.
- Estado regulatório (Pesquisa/Clínico) e avisos → **config de alto nível**.
- Marcação de lesão e correção de máscara → **externo** (3D Slicer, humano).

## O que separa este desenho do script original

O script monolítico misturava motor, parâmetros de fígado, política e até um
fallback perigoso, tudo em um arquivo, e várias etapas não funcionavam de fato.
Aqui: motor isolado e determinístico, parâmetros em perfis, gates de segurança
explícitos (`05_PIPELINE.md`), e cada estágio com um contrato claro de entrada e
saída. A reescrita do script só deve ser feita **depois** de este contexto estar
aprovado, para não recodificar em cima de premissas erradas.
