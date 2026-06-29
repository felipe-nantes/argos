# Digital Twin Cirúrgico — Produção-Ready (Nível 1, modo Pesquisa)

**Data:** 2026-06-29
**Estado:** aprovado para implementação

## Objetivo

Levar o repositório de "código existe" para "tudo funcional e verificável", mantendo
o **modo Pesquisa** (não é dispositivo médico; uso clínico é um gate regulatório
ANVISA fora deste escopo — ver `contexto/03_REGULATORIO_LGPD.md`).

"Funcional" aqui = quatro frentes, todas aprovadas:

1. Pipeline rodando ponta a ponta (na máquina com GPU do usuário).
2. Robustez de código: testes automáticos, packaging, git, preflight.
3. Visualizador web (hoje inexistente).
4. Fixture sintético + smoke test (prova o pipeline sem GPU/Slicer/DICOM).

## Restrições e decisões

- **Python:** venv em **3.13** (3.14 cedo demais para wheels de torch/SimpleITK).
  `requires-python = ">=3.10,<3.14"`.
- **GPU:** a segmentação pesada (TotalSegmentator + torch) roda em outra máquina do
  usuário. Nesta máquina instala-se só o núcleo + dev (sem torch); os testes cobrem
  tudo, exceto o estágio 3, que é stubado.
- **Viewer:** HTML estático + Three.js (CDN), zero build.
- **Hospedagem:** git **local** por enquanto. CI/deploy adiados.
- **Princípio mantido:** regra de domínio mora em config (perfis YAML), nunca no
  motor. Nenhuma das mudanças toca esse contrato.

## Componentes

### A. Repo + packaging

- `git init` (feito), `.gitignore` cobrindo: `casos/`, `flywheel/`, `*.nii.gz`,
  `*.stl`, `*.vtp`, `__pycache__/`, `.venv/`, `*.egg-info/`, `.pytest_cache/`.
- `pyproject.toml` (PEP 621):
  - metadata: name `digital-twin-cirurgico`, version `0.1.0` (espelha
    `dtwin.__version__`), `requires-python = ">=3.10,<3.14"`.
  - `dependencies` = núcleo: SimpleITK>=2.3, nibabel>=5.0, numpy>=1.24, scipy>=1.10,
    scikit-image>=0.22, pyvista>=0.43, pydicom>=2.4, PyYAML>=6.0.
  - `optional-dependencies.seg` = `TotalSegmentator>=2.4` (heavy/torch — só GPU box).
  - `optional-dependencies.dev` = `pytest>=8`.
  - `project.scripts`: `digital-twin = digital_twin:main`.
- `requirements.txt` mantido como atalho; aponta para os mesmos pins (não duplica
  regra — é conveniência). Comentário liga ao `pyproject.toml` como fonte da verdade.
- `.venv` em 3.13 com `pip install -e .[dev]` nesta máquina.

### B. Testes (`tests/`, pytest) — sem GPU/Slicer/DICOM real

Unidades determinísticas e gates:

- `core.world_vertices_from_index`: direction identidade e oblíqua → LPS correto
  (verifica origin + direction + spacing aplicados; ordem de eixos zyx→xyz).
- `core.load_profile`: perfil válido carrega; faltando chave obrigatória ou
  `rotulo_alvo` → `PipelineError`; arquivo inexistente → `PipelineError`.
- `stages._refine_mask`: opening remove pontas; closing fecha buracos;
  `min_voxels` descarta fragmentos; máscara cheia não é zerada por engano.
- `stages._mesh_from_mask`: máscara cubo sintético → malha não-vazia; máscara
  vazia → `None`; vértices em coordenadas físicas (não em índice).
- `stages.stage2_normalize`: volume constante (std~0) → `PipelineError`;
  zscore produz média~0/std~1; método inválido → `PipelineError`.
- `engine.finalize` (estágios 4b–7) em **caso sintético** → gera
  `outputs/figado_orgao.stl`, `outputs/figado_lesao.stl` e
  `outputs/viewer_manifest.json` válidos (JSON parseável, meshes apontando para
  nomes relativos de STL existentes).
- `engine.prepare` com `stages.stage3_segment_organ` monkeypatchado (grava uma
  `mask_organ` sintética) → estágios 1, 2, 4a rodam sem torch; valida o gate de
  modalidade (DICOM MR aceito; outra modalidade → `PipelineError`).

### C. Fixture sintético + smoke (`tools/make_synthetic_case.py`)

- Gera, via SimpleITK, um caso completo em disco:
  - `volume.nii.gz` (blob com geometria não-trivial: spacing anisotrópico,
    origin != 0, direction identidade — suficiente para exercitar a geometria).
  - `mask_organ.nii.gz` (elipsoide/blob).
  - `mask_lesion.nii.gz` (esfera menor dentro do órgão).
  - `manifest.json` mínimo compatível com `Case.read_manifest()`
    (`case_id`, `policy`, `modality`, `regulatory_state`...).
- Gera também uma pasta DICOM sintética (pydicom, `Modality="MR"`, série de N
  fatias com geometria coerente) para exercitar o caminho `prepare` (estágios 1–2).
- O smoke test usa esse fixture: roda `finalize` e afirma que as saídas existem e
  parseiam — prova que os estágios 4b–7 funcionam de ponta a ponta localmente.

### D. Execução em dados reais (máquina GPU do usuário)

- `docs/RUNNING.md`: instalação (`pip install .[seg]`), fluxo `prepare` → 3D Slicer
  (marcar lesão) → `finalize`, flags de device (`--device cpu --fast`, `gpu:N`),
  e troubleshooting (classe inválida → `totalseg_info --classes -ta total_mr`).
- `dtwin doctor` (subcomando novo, pequeno, no CLI): checa imports do núcleo, reporta
  se `TotalSegmentator` é importável e qual device, e sai com mensagem clara — falha
  rápido antes de uma execução longa. Não baixa nada, não roda segmentação.

### E. Visualizador web (`viewer/`) — HTML estático + Three.js

- `viewer/index.html` + `viewer/app.js`:
  - Three.js + `STLLoader` + `OrbitControls` via CDN (versão pinada).
  - Carrega um caso por **drag-drop** da pasta `outputs/` (lê `viewer_manifest.json`
    + STLs referenciados), ou via `?case=<caminho>` quando servido por
    `python -m http.server`.
  - Cor por mesh vinda do manifesto (`color`); toggles de visibilidade órgão/lesão;
    slider de opacidade; **banner de disclaimer de modo Pesquisa**; nota de sistema
    de coordenadas (LPS).
- `viewer/README.md`: como servir e abrir.

### F. Passada leve de robustez no código existente

- Confirmar que `stage7_export_publish` mantém caminhos de STL **relativos** no
  manifesto (já mantém — viewer depende disso).
- Pinar versões de CDN do viewer e confirmar nas versões instaladas que
  `skimage.morphology` usa `footprint=` e `pyvista` usa `mesh.save(...)`
  (já corrigidos no código; o teste passa a ser a garantia de regressão).

## Fora de escopo

Uso clínico / ANVISA, pseudonimização ativa, FEA / tetraedralização (Nível 2),
integração PACS, frontend além do viewer estático, GitHub Actions / CI / deploy em
nuvem. Todos mapeados em `contexto/08_ROADMAP.md` e
`contexto/10_MATURIDADE_DIGITAL_TWIN.md`.

## Critérios de sucesso

1. `pip install -e .[dev]` instala limpo no venv 3.13 desta máquina.
2. `pytest` verde — todos os testes acima passando, sem torch instalado.
3. `python tools/make_synthetic_case.py` + `finalize` geram STLs + manifesto.
4. Abrir o viewer e carregar a pasta `outputs/` sintética renderiza órgão + lesão.
5. `digital-twin doctor` roda e reporta o estado das dependências.
6. `docs/RUNNING.md` descreve o caminho real ponta a ponta na máquina GPU.
