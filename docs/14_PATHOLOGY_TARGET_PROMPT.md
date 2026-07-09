# Prompt pathology-target — separação entre variante benigna e patologia alvo

Esta etapa adiciona um novo cenário de benchmark/inferência para reduzir falsos positivos em fígados sem lesão focal, mas com variantes anatômicas benignas ou pseudolesões.

## Objetivo

Trocar o foco do MedGemma de:

```text
qualquer alteração visual no fígado
```

para:

```text
lesão focal hepática ou patologia hepática suspeita
```

Assim, uma veia calibrosa, estrutura tubular contínua, variante vascular, pseudolesão ou artefato provável pode ser registrada como achado observado sem transformar automaticamente o caso em `POSITIVA`.

## Configs novas

- `configs/medgemma_local_4b_volumetric_pathology_target.yaml`
- `configs/medgemma_ollama_27b_volumetric_pathology_target.yaml`

Ambas herdam as configs volumétricas e alteram somente o `prompt.template`.

## Webapp

O benchmark do webapp ganhou o cenário autorizado:

```text
pathology_target
```

Por padrão, ele aponta para:

```text
configs/medgemma_local_4b_volumetric_pathology_target.yaml
```

Pode ser sobrescrito por variável de ambiente:

```powershell
$env:WEBAPP_PATHOLOGY_TARGET_MEDGEMMA_CONFIG = "configs/medgemma_ollama_27b_volumetric_pathology_target.yaml"
```

## O que muda na decisão

Regra central:

- `POSITIVA`: exige suspeita de lesão focal hepática ou patologia alvo.
- `NEGATIVA`: aceita fígado sem lesão focal mesmo que exista variante anatômica benigna ou pseudolesão provável.
- `INCONCLUSIVA`: usada quando não é possível separar variante/artefato de patologia alvo com segurança.

## O que ainda não muda

Nesta etapa o schema JSON do relatório passa a aceitar campos v2 opcionais:

```json
{
  "resultado_hipotese": "POSITIVA | NEGATIVA | INCONCLUSIVA",
  "resumo_do_achado": "string",
  "localizacao_aproximada": "string",
  "sinais_visuais_observados": ["string"],
  "confianca": "baixa | moderada | alta",
  "limitacoes_da_analise": ["string"],
  "necessidade_de_revisao_humana": true,
  "alvo_da_triagem": "lesao_focal_hepatica_suspeita",
  "ha_lesao_focal_suspeita": true,
  "ha_variante_anatomica_benigna": false,
  "ha_pseudolesao_ou_artefato": false,
  "tipo_alteracao_nao_alvo": "none | vascular_variant | perfusion_pseudolesion | artifact | focal_fat | cystic_benign | other",
  "justificativa_da_separacao": "string"
}
```

Os campos antigos continuam obrigatórios. Campos v2 desconhecidos ou inconsistentes são rejeitados/retry pelo validador.

## Como comparar

Rodar no mesmo conjunto:

```text
volumetric
pathology_target
```

Comparar:

- especificidade geral;
- especificidade em variantes anatômicas benignas;
- taxa de positivos em hard negatives;
- taxa de inconclusivos;
- sensibilidade nos positivos patológicos.

O ganho esperado é principalmente na especificidade, não necessariamente na sensibilidade.
