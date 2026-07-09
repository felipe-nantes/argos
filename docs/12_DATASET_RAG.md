# Dataset registry para RAG hepático

Esta etapa cria uma camada segura entre datasets públicos/locais e as próximas fases do RAG/GraphRAG.

O registry não lê diagnóstico do modelo, não executa MedGemma e não envia labels para inferência. Ele apenas transforma uma pasta local em um manifesto JSONL padronizado, com metadados úteis para curadoria, auditoria, benchmark estratificado e futura ingestão no grafo.

## Objetivo

Separar, de forma rastreável:

- controles anatômicos negativos;
- negativos com limitações conhecidas;
- positivos amplos em NIfTI;
- positivos HCC em DICOM/RM;
- anotações disponíveis, sempre protegidas para uso pós-inferência.

## Configs versionadas

As configs iniciais ficam em `configs/datasets/`:

- `chaos_mri.yaml`
- `lld_mmri.yaml`
- `liverhccseg.yaml`
- `tcga_lihc_mr.yaml`

Regras importantes:

- CHAOS MRI é controle anatômico negativo, mas nunca deve ser chamado de “normal absoluto”.
- LLD-MMRI é NIfTI original; nunca deve sair como `dicom_original=true`.
- TCGA-LIHC só aceita séries com `Modality == MR`.
- LiverHccSeg/TCGA podem ter anotações, mas elas são registradas apenas como caminho protegido para avaliação posterior.

## CLI

Exemplo:

```bash
python -m dtwin.datasets.ingest \
  --config configs/datasets/chaos_mri.yaml \
  --root data/raw/CHAOS_MRI \
  --out data/registry/chaos_mri.jsonl
```

Repetir para os demais datasets:

```bash
python -m dtwin.datasets.ingest --config configs/datasets/lld_mmri.yaml --root data/raw/LLD-MMRI --out data/registry/lld_mmri.jsonl
python -m dtwin.datasets.ingest --config configs/datasets/liverhccseg.yaml --root data/raw/LiverHccSeg --out data/registry/liverhccseg.jsonl
python -m dtwin.datasets.ingest --config configs/datasets/tcga_lihc_mr.yaml --root data/raw/TCGA-LIHC --out data/registry/tcga_lihc_mr.jsonl
```

`data/` permanece fora do Git.

## Saída

Cada linha do JSONL segue `argos-dataset-registry-v1` e contém:

- `case_id` anonimizado/determinístico;
- `series_id` como hash, nunca UID DICOM bruto;
- `dataset_id`, `dataset_name`, `rag_class` e `label`;
- subtipos positivos/negativos quando definidos pela config;
- `source_format`, `dicom_original`, `nifti_original`;
- `raw_path` relativo à raiz local;
- `annotation_path`, quando uma máscara/anotação for encontrada;
- `research_only=true` e `clinical_use_allowed=false`.

## Segurança metodológica

- O registry não deve ser colocado diretamente na inferência do MedGemma.
- Labels, subtipos e anotações são usados apenas depois da inferência.
- UIDs DICOM brutos não são persistidos.
- O GraphRAG futuro deve ser alimentado a partir desses JSONL, não diretamente dos DICOMs/NIfTIs.
- Nenhum dado de paciente deve ser versionado no Git.
