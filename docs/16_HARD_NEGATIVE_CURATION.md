# Curadoria operacional de negativos difíceis

Transforma falsos positivos atuais em conhecimento reutilizável, fechando o
ciclo:

```text
erro atual → hard negative documentado → rótulo protegido → melhor prompt
→ melhor métrica estratificada → GraphRAG mais forte
```

## Arquivo de trabalho

A revisão humana vive em:

```text
data/curation/negative_hard_cases_review.jsonl
```

`/data/` é ignorado pelo Git, então esse arquivo nunca é versionado. Um
template sem dados sensíveis está em
`configs/curation/negative_hard_cases_review.template.jsonl`.

Cada linha descreve a revisão de um caso:

```json
{
  "case_id": "anon-001",
  "current_label": "NEGATIVE",
  "recommended_label": "NEGATIVE",
  "recommended_negative_subtype": "benign_anatomic_variant",
  "phenotype_tags": ["prominent_hepatic_vein", "vascular_structure"],
  "reviewer": "human",
  "review_status": "reviewed",
  "notes": "Veia calibrosa sem massa focal."
}
```

Regras validadas:

- `current_label`/`recommended_label` ∈ {POSITIVE, NEGATIVE};
- subtipos negativos/positivos pertencem ao vocabulário fechado da taxonomia;
- subtipo negativo e positivo são mutuamente exclusivos;
- subtipo negativo exige `recommended_label=NEGATIVE`; positivo exige POSITIVE;
- `phenotype_tags` restritas ao vocabulário;
- `review_status` ∈ {pending_review, reviewed, needs_second_opinion}.

## Ordem de curadoria

1. Revisar todos os negativos que o modelo classificou como `POSITIVA`.
2. Separar em: lesão real não rotulada; variante anatômica benigna;
   pseudolesão/artefato; qualidade ruim; caso realmente normal.
3. Atualizar os rótulos protegidos.
4. Reexecutar o benchmark pathology-target.
5. Comparar contra o baseline antigo.

## CLI

Validar a revisão e ver o resumo:

```bash
python -m dtwin.datasets.curation \
  --review data/curation/negative_hard_cases_review.jsonl
```

Emitir os rótulos protegidos (apenas casos `reviewed`) para alimentar o
manifesto de labels do benchmark:

```bash
python -m dtwin.datasets.curation \
  --review data/curation/negative_hard_cases_review.jsonl \
  --out data/curation/protected_labels.jsonl
```

Cada linha de saída já vem no formato de rótulo protegido
(`target_condition`, `negative_subtype`/`positive_subtype`, `phenotype_tags`,
`label_basis=human_review`, `review_status`), anexado somente após a
inferência.
