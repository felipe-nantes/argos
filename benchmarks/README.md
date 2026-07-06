# Benchmark MedGemma — Etapa 0

Este diretório contém apenas exemplos e documentação. Datasets, manifests reais,
painéis, relatórios e runs ficam sob `casos/`, que não é versionado.

O benchmark mede a triagem visual atual sem alterar painel, seleção de fatias,
fusão, windowing ou prompt. Continua sendo modo Pesquisa, não diagnóstico.

## Política estatística

As métricas primárias usam todos os exames. `INCONCLUSIVA`, falha, timeout e
resposta inválida contam como erro do respectivo grupo. O gate só passa quando
sensibilidade **e** especificidade atingem 75%.

As métricas `decisions-only` consideram somente `POSITIVA` e `NEGATIVA`. Elas são
secundárias e podem superestimar desempenho porque excluem abstenções e falhas.

Cada run salva matriz categórica e uma matriz binária penalizada. Esta última é
uma convenção conservadora de pontuação; não significa que uma falha técnica em
caso negativo foi literalmente uma predição positiva.

## Manifestos

O manifest de datasets referencia um arquivo de labels separado. Veja
`datasets.example.yaml` e `labels.example.yaml`.

Dentro de cada caso, `inference` aceita somente dados que podem chegar ao modelo.
`ground_truth` contém label, máscara de lesão e anotações protegidas. O loader
rejeita qualquer máscara de lesão ou label colocado em `inference`.

Formatos:

- `DICOM`: informe `dicom_dir`; opcionalmente `organ_mask`. Sem máscara, ative
  `segment_if_missing` para executar o `prepare` existente.
- `NIFTI`: informe explicitamente `volume` e `organ_mask`.
- `MIDS`: usa o mesmo importador NIfTI explícito. O ARGOS não tenta adivinhar
  nomes OpenSwissHCC sem validar uma amostra/data dictionary real.

## Dry-run

```powershell
.\.venv-win\Scripts\python.exe -m dtwin.medgemma_benchmark `
  --datasets-manifest casos\benchmark_manifests\datasets.yaml `
  --medgemma-config configs\medgemma_local_4b.yaml `
  --experiment-config benchmarks\experiments\current_panel.example.yaml `
  --dry-run
```

O dry-run valida manifests, paths, labels, geometria, isolamento e hashes. Não
chama o MedGemma e não gera hipótese clínica.

## Execução completa

```powershell
.\.venv-win\Scripts\python.exe -m dtwin.medgemma_benchmark `
  --datasets-manifest casos\benchmark_manifests\datasets.yaml `
  --medgemma-config configs\medgemma_local_4b.yaml `
  --experiment-config benchmarks\experiments\current_panel.example.yaml `
  --out casos\webapp\benchmarks\runs
```

Filtros disponíveis: `--limit`, `--dataset`, `--case-id`, `--fail-fast` e
`--seed`. Para usar relatórios anteriores sem nova inferência, informe
`--skip-inference-use-existing-reports --existing-run <run>`. O reuso aborta se
modelo, config, estratégia, labels ou hashes divergirem.

## Saídas

Cada run contém:

- `run_manifest.json`: commit, dirty state/diff hash, modelo, config, ambiente e duração;
- `cases.jsonl`: um registro auditável por exame;
- `metrics_primary.json` e `metrics_decisions_only.json`;
- `confusion_matrices.json`;
- `summary.md`.

Execuções de desenvolvimento registram árvore dirty. Um experimento com
`final_evaluation: true` exige árvore Git limpa.

## Isolamento do ground truth

Cada caso é copiado para um `inference/` sanitizado contendo apenas volume,
máscara hepática e manifesto sem label. O subprocesso MedGemma recebe somente
esses caminhos. O avaliador anexa label e hashes protegidos depois que a resposta
foi persistida. Máscaras de lesão nunca são copiadas para o workspace de inferência.

## OpenSwissHCC

O contrato genérico `MIDS` está implementado e testado com fixtures sintéticas.
O adaptador específico OpenSwissHCC só deve ser declarado validado depois de um
dry-run com a estrutura real da versão baixada do dataset.
