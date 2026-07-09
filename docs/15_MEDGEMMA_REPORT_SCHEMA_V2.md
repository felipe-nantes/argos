# Schema v2 do relatório MedGemma

Esta etapa adiciona campos opcionais ao relatório MedGemma para separar explicitamente:

- lesão focal hepática/patologia alvo suspeita;
- variante anatômica benigna;
- pseudolesão ou artefato.

O schema antigo continua compatível. Os campos antigos seguem obrigatórios.

## Campos antigos obrigatórios

```json
{
  "resultado_hipotese": "POSITIVA | NEGATIVA | INCONCLUSIVA",
  "resumo_do_achado": "string",
  "localizacao_aproximada": "string",
  "sinais_visuais_observados": ["string"],
  "confianca": "baixa | moderada | alta",
  "limitacoes_da_analise": ["string"],
  "necessidade_de_revisao_humana": true
}
```

## Campos v2 opcionais

```json
{
  "alvo_da_triagem": "lesao_focal_hepatica_suspeita",
  "ha_lesao_focal_suspeita": true,
  "ha_variante_anatomica_benigna": false,
  "ha_pseudolesao_ou_artefato": false,
  "tipo_alteracao_nao_alvo": "none | vascular_variant | perfusion_pseudolesion | artifact | focal_fat | cystic_benign | other",
  "justificativa_da_separacao": "string"
}
```

## Regras de consistência

- `resultado_hipotese=POSITIVA` com campos v2 exige `ha_lesao_focal_suspeita=true`.
- `ha_variante_anatomica_benigna=true` não pode, sozinha, tornar o caso `POSITIVA`.
- `ha_pseudolesao_ou_artefato=true` não pode, sozinha, tornar o caso `POSITIVA`.
- `tipo_alteracao_nao_alvo` deve estar na lista permitida.
- Campos extras desconhecidos continuam descartados.
- Diagnóstico definitivo e recomendação de conduta continuam bloqueados.

## Retry

Se o modelo devolver uma contradição v2, o adaptador HTTP rejeita a resposta e aciona o retry de schema, quando configurado.

## Agregação volumétrica

Quando há múltiplos painéis, a agregação preserva os campos v2:

- `ha_lesao_focal_suspeita`: verdadeiro se qualquer painel declarar verdadeiro;
- `ha_variante_anatomica_benigna`: verdadeiro se qualquer painel declarar verdadeiro;
- `ha_pseudolesao_ou_artefato`: verdadeiro se qualquer painel declarar verdadeiro;
- `tipo_alteracao_nao_alvo`: valor único se todos concordarem; `other` se houver mais de um tipo.

Essa agregação não inventa síntese clínica: as justificativas são preservadas por painel.
