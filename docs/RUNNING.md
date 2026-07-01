# Executando o pipeline

Dois ambientes: (a) **desenvolvimento/teste** nesta máquina, sem GPU; (b)
**execução real** na máquina com GPU, onde roda a segmentação automática.

## (a) Dev/teste — sem GPU/Slicer/DICOM

```bash
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -e .[dev]
.venv/Scripts/python.exe -m pytest          # suíte verde
.venv/Scripts/python.exe digital_twin.py doctor

# caso sintético ponta a ponta (estágios 4b–7)
.venv/Scripts/python.exe tools/make_synthetic_case.py --out casos/sintetico
.venv/Scripts/python.exe digital_twin.py finalize casos/sintetico --profile profiles/figado.yaml
```

Abra o resultado no visualizador: ver `viewer/README.md`.

## (b) Execução real (máquina com GPU)

```bash
pip install -e .[seg]        # traz TotalSegmentator + torch (grande)
digital-twin doctor          # confirme "torch device: cuda"
```

**Fase 1 — prepare** (estágios 1–4a):

```bash
digital-twin prepare /caminho/serie_dicom \
    --case-dir casos/paciente001 --profile profiles/figado.yaml
# CPU: adicione --device cpu --fast (lento)
```

**Etapa manual — 3D Slicer:** abra `casos/paciente001/volume.nii.gz` e
`mask_organ.nii.gz`, revise o órgão, marque a lesão e salve EXATAMENTE em
`casos/paciente001/mask_lesion.nii.gz` (instruções exatas são impressas ao fim do
`prepare`).

**Fase 2 — finalize** (estágios 4b–7):

```bash
digital-twin finalize casos/paciente001 --profile profiles/figado.yaml
# Sem lesão (escolha explícita): adicione --no-lesion
```

Saídas em `casos/paciente001/outputs/`: `figado_orgao.stl`, `figado_lesao.stl`,
`viewer_manifest.json`.

## Troubleshooting

- `TotalSegmentator não está instalado` → `pip install -e .[seg]`.
- `Saída de segmentação esperada não encontrada` / classe inválida → confira nomes:
  `totalseg_info --classes -ta total_mr`.
- `Modalidade do exame (...) não bate` → use o perfil correto; `figado.yaml` espera MRI.
- Wheels falhando na instalação → confirme Python **3.13** (`py -3.13`); 3.14 ainda
  não tem wheels de torch/SimpleITK.

## MedGemma local (opcional, modo Pesquisa)

O fluxo MedGemma exige licença aceita e login no Hugging Face. Depois:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[medgemma]"
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python.exe tools\setup_medgemma.py
.\.venv\Scripts\python.exe tools\setup_medgemma.py --local-only
powershell -ExecutionPolicy Bypass -File tools\start_medgemma.ps1
```

Gere primeiro o painel e faça a revisão visual de PHI:

```powershell
.\.venv\Scripts\python.exe -m dtwin.medgemma_screening `
  --case-dir casos\caso_real_001 `
  --medgemma-config configs\medgemma_local_4b.yaml `
  --panel-only
```

Depois de revisar o PNG, execute a inferência:

```powershell
.\.venv\Scripts\python.exe -m dtwin.medgemma_screening `
  --case-dir casos\caso_real_001 `
  --medgemma-config configs\medgemma_local_4b.yaml `
  --confirm-no-visible-phi
```

O relatório fica em `outputs/medgemma/medgemma_report.json`, sempre
`pending_review`. Nunca coloque o token em arquivos do projeto. Uso e gates
completos: `contexto/11_MEDGEMMA_SCREENING.md`.
