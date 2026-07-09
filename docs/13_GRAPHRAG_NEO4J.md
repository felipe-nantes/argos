# GraphRAG Neo4j — metadados hepáticos

Esta etapa adiciona o primeiro GraphRAG do ARGOS: um grafo de metadados para separar anatomia normal, variantes benignas, pseudolesões/artefatos e patologia alvo.

Ele ainda não altera o prompt do MedGemma. Nesta fase, o grafo serve para consulta, auditoria e preparação da futura integração com o modo `pathology-target`.

## Fonte de dados

O Neo4j deve ser alimentado pelos JSONL gerados pelo dataset registry:

```bash
python -m dtwin.datasets.ingest \
  --config configs/datasets/chaos_mri.yaml \
  --root data/raw/CHAOS_MRI \
  --out data/registry/chaos_mri.jsonl
```

Depois:

```bash
python -m dtwin.graphrag.ingest_registry \
  --config configs/graphrag_neo4j.yaml \
  --manifests \
    data/registry/chaos_mri.jsonl \
    data/registry/lld_mmri.jsonl \
    data/registry/liverhccseg.jsonl \
    data/registry/tcga_lihc_mr.jsonl
```

## Configuração

Arquivo:

```text
configs/graphrag_neo4j.yaml
```

Senha via ambiente:

```bash
export NEO4J_PASSWORD='sua-senha'
```

No Windows PowerShell:

```powershell
$env:NEO4J_PASSWORD = "sua-senha"
```

## Consulta

Exemplo:

```bash
python -m dtwin.graphrag.query \
  --config configs/graphrag_neo4j.yaml \
  --negative-subtype benign_anatomic_variant \
  --phenotype-tag prominent_hepatic_vein \
  --target focal_liver_lesion
```

Saída esperada:

```json
{
  "query_mode": "metadata_graphrag",
  "target_condition": "focal_liver_lesion_suspicion",
  "retrieved_cases": [],
  "mimic_context": [],
  "limitations": [],
  "research_only": true,
  "clinical_use_allowed": false
}
```

## Relações iniciais

O grafo cria relações de mimetismo, por exemplo:

- `prominent_hepatic_vein CAN_MIMIC focal_liver_lesion`
- `vascular_structure CAN_MIMIC focal_liver_lesion`
- `perfusion_alteration CAN_MIMIC arterial_hyperenhancement`
- `motion_artifact CAN_MIMIC focal_lesion`
- `partial_volume_effect CAN_MIMIC focal_lesion`
- `focal_fat CAN_MIMIC focal_liver_lesion`
- `simple_cyst CAN_MIMIC hypovascular_lesion`
- `edge_of_liver_pseudolesion CAN_MIMIC focal_liver_lesion`

Também registra que subtipos negativos são negativos para a patologia alvo.

## Segurança metodológica

- O GraphRAG não lê DICOM/NIfTI diretamente.
- O GraphRAG não emite diagnóstico.
- UIDs DICOM brutos não são persistidos.
- `raw_path` não é salvo como propriedade no nó do caso.
- `research_only=true` e `clinical_use_allowed=false` são obrigatórios.
- Ground truth continua fora da inferência.

## Próxima etapa

A próxima parte do plano é o prompt `pathology-target` e/ou schema v2 do relatório MedGemma, usando a taxonomia protegida já implementada.
