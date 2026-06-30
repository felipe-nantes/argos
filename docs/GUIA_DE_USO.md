# Guia de Uso — Digital Twin Cirúrgico (passo a passo)

> **Modo Pesquisa.** Nada aqui se destina a decisão clínica. Coordenadas LPS.
> Este guia é prático e literal: copie e cole os comandos na ordem.
> Para a referência curta de operação, veja [`RUNNING.md`](RUNNING.md).

O sistema transforma uma série **DICOM de RM** em um **modelo 3D (órgão + lesão)**
em **STL**, visualizável no navegador. O fluxo tem **duas fases** com **uma etapa
manual no meio** (a lesão é marcada por um humano no 3D Slicer):

```
prepare  ──►  [marcação da lesão no 3D Slicer]  ──►  finalize  ──►  viewer
(1–4a)                  (etapa humana)               (4b–7)        (navegador)
```

Há **dois ambientes**:
- **(A) Esta máquina (sem GPU):** desenvolvimento, testes e o **caso sintético**.
- **(B) Máquina com GPU NVIDIA:** a execução real, onde roda a segmentação
  automática do órgão (TotalSegmentator).

---

## Parte 0 — Convenções

- Os comandos usam o Python do ambiente virtual do projeto:
  `\.venv\Scripts\python.exe`. Se preferir, ative o venv uma vez por terminal
  com `\.venv\Scripts\Activate.ps1` e use só `python`.
- Shell assumido: **PowerShell no Windows**. Em Linux/Mac troque
  `\.venv\Scripts\python.exe` por `.venv/bin/python`.
- Todos os comandos são rodados **na raiz do projeto**
  (`C:\Users\fnant\Projects\digital_twin_cirurgico`).

---

## Parte 1 — Instalação (uma vez por máquina)

### 1.1 Pré-requisitos

- **Python 3.13** instalado (`py -3.13 --version` deve responder).
- **Git** (para clonar/atualizar o repositório).
- Para o ambiente (B): **GPU NVIDIA** com driver CUDA e ~6 GB+ de VRAM.
- Para a etapa manual: **3D Slicer** (https://www.slicer.org) — gratuito.

### 1.2 Criar o ambiente virtual e instalar

**Ambiente (A) — sem GPU (testes + caso sintético):**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

**Ambiente (B) — com GPU (execução real): forma rápida (1 comando)**

Um bootstrap cria o venv, instala tudo e **verifica** (GPU + rótulo do órgão):
```powershell
py -3.13 tools\setup_real_env.py
```
Ele imprime um relatório e sai com **0 = ambiente pronto** ou **1 = algo falta**.
Para já validar ponta a ponta com um exame real, acrescente o smoke:
```powershell
py -3.13 tools\setup_real_env.py --smoke "C:\serie_dicom" --no-lesion
```
Já tem o ambiente e só quer reconferir? `py -3.13 tools\setup_real_env.py --verify-only`.

**Ambiente (B) — alternativa manual:**
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[seg]
```
> O extra `[seg]` baixa **TotalSegmentator + PyTorch** (vários GB). Demora.

### 1.3 Conferir o ambiente (preflight)

```powershell
.\.venv\Scripts\python.exe digital_twin.py doctor
```

O que esperar:
- **Ambiente (A):** todas as dependências do núcleo `[OK]`; TotalSegmentator e
  torch aparecem como `[FALTA]`/ausentes — **isso é o esperado** sem GPU.
- **Ambiente (B):** além do núcleo, `TotalSegmentator importável` e
  `torch device: cuda`. Se aparecer `cpu`, a GPU não está visível ao torch —
  resolva o driver/CUDA antes de continuar (rodar em CPU funciona, mas é lento).

---

## Parte 2 — Teste rápido em 5 minutos (caso sintético, qualquer máquina)

Prova o pipeline ponta a ponta (estágios 4b–7) **sem GPU, sem DICOM, sem Slicer**,
usando um caso fictício gerado em disco.

```powershell
# 1) gera um caso sintético (órgão + lesão fictícios)
.\.venv\Scripts\python.exe tools\make_synthetic_case.py --out casos\sintetico

# 2) finaliza: refino + malha + STL + manifesto
.\.venv\Scripts\python.exe digital_twin.py finalize casos\sintetico --profile profiles\figado.yaml
```

**Critério de passe:** ao final imprime `[OK] 'finalize' concluído` e a pasta
`casos\sintetico\outputs\` contém:
- `figado_orgao.stl`
- `figado_lesao.stl`
- `viewer_manifest.json`

Para ver no navegador, vá direto à **Parte 5** (use o caminho
`../casos/sintetico/outputs`).

> Rodar a **suíte de testes** (não precisa de GPU):
> ```powershell
> .\.venv\Scripts\python.exe -m pytest
> ```
> Esperado: tudo verde (43 testes).

---

## Parte 3 — Execução real, Fase 1: `prepare` (máquina com GPU)

Estágios 1–4a: ingestão + des-identificação + normalização + **segmentação
automática do órgão**.

> ### ⚠️ Onde entra a RM? **Não há "upload".**
> Este sistema é **linha de comando** — não existe site nem botão de enviar
> arquivo. A RM "entra" como o **caminho de uma pasta** que você passa ao
> `prepare`. Dois pontos:
> 1. **A RM é uma PASTA, não um arquivo.** Um exame de RM são vários `.dcm`
>    (uma fatia cada). Você aponta para a **pasta** que contém todos eles.
> 2. **Deixe essa pasta onde quiser no disco** (ex.: `C:\exames\paciente001\`)
>    e informe esse caminho como **primeiro argumento** do `prepare` (abaixo).
>    Nada é copiado para um servidor; o pipeline lê a pasta direto do disco e
>    grava a saída em `--case-dir` (a pasta do caso, que você escolhe).

Você precisa de uma **pasta com a série DICOM de RM** de um paciente (uma pasta
contendo os arquivos `.dcm` das fatias).

```powershell
.\.venv\Scripts\python.exe digital_twin.py prepare "C:\caminho\serie_dicom" `
    --case-dir casos\paciente001 `
    --profile profiles\figado.yaml
#                                  ^^^^^^^^^^^^^^^^^^^^^^
#   "C:\caminho\serie_dicom"  = a PASTA da RM (sua entrada — troque pelo caminho real)
#   --case-dir casos\paciente001 = onde a SAÍDA é gravada (você escolhe)
```

Opções úteis:
- Sem GPU, mas com o exame real: acrescente `--device cpu --fast` (lento, mas
  valida o caminho).
- A modalidade é checada: `figado.yaml` espera **RM (MR/MRI)**. Exame de outra
  modalidade é **rejeitado** de propósito.

**Critério de passe:** termina com `[OK] 'prepare' concluído` e cria, em
`casos\paciente001\`:
- `volume.nii.gz` — o exame anonimizado (cabeçalhos DICOM descartados);
- `mask_organ.nii.gz` — o órgão segmentado automaticamente (**não pode estar vazio**);
- `manifest.json` — metadados do caso.

Ao final, o programa **imprime as instruções exatas** da etapa manual. Leia-as.

> Se abortar com `TotalSegmentator não está instalado` → você está no ambiente
> errado; instale o extra `[seg]` (Parte 1.2 B). O pipeline **nunca inventa** uma
> máscara — ele aborta. Isso é proposital.

### 3.1 Smoke test automático (recomendado na 1ª execução)

Em vez de rodar `prepare`/`finalize` na mão e conferir os arquivos, use o smoke
que roda tudo e **valida cada artefato**, devolvendo passou/falhou:

```powershell
# com uma lesão já pronta (.nii.gz):
.\.venv\Scripts\python.exe tools\smoke_gpu.py --dicom "C:\serie_dicom" --lesion "C:\mask_lesion.nii.gz"

# ou validando só a mecânica, sem lesão:
.\.venv\Scripts\python.exe tools\smoke_gpu.py --dicom "C:\serie_dicom" --no-lesion
```

Imprime um relatório linha a linha e sai com código **0** (tudo OK) ou **1**
(alguma checagem falhou). É a forma mais rápida de provar que a segmentação real
+ ingestão DICOM + STL + manifesto funcionam ponta a ponta nesta máquina.

---

## Parte 4 — Etapa manual: marcar a LESÃO no 3D Slicer

O órgão é automático; **a lesão é marcada por um humano**. Faça no 3D Slicer:

1. Abra o **3D Slicer**.
2. **Carregue o volume:** `File ▸ Add Data ▸ Choose File(s)` →
   `casos\paciente001\volume.nii.gz` → **OK**.
3. **Carregue a máscara do órgão como segmentação:**
   `File ▸ Add Data` → `casos\paciente001\mask_organ.nii.gz`. Na janela de opções
   (`Description`), marque a coluna **LabelMap** para esse arquivo antes do OK
   (ou, depois, converta em segmento via `Segmentations`). Revise se o órgão está
   correto; corrija se necessário.
4. **Crie o segmento da lesão:**
   - Abra o módulo **Segment Editor** (menu de módulos).
   - `Add` para criar um novo segmento; renomeie para `lesao`.
   - Use as ferramentas sugeridas: **Threshold**, **Grow from seeds /
     Region growing**, ou **Paint**, para pintar a lesão sobre o volume.
5. **Exporte a lesão como NIfTI, no caminho exato:**
   - Módulo **Segmentations** → seção `Export to files` (ou
     `File ▸ Save`), formato **.nii.gz**, **referência geométrica = o volume**
     carregado no passo 2.
   - Salve **exatamente** como:
     ```
     casos\paciente001\mask_lesion.nii.gz
     ```
   - O nome e o caminho precisam ser **idênticos** a esse — o `finalize` procura
     esse arquivo.
6. (Opcional) Se você corrigiu o órgão no passo 3, sobrescreva
   `casos\paciente001\mask_organ.nii.gz`.

> A lesão precisa ter **o mesmo tamanho/geometria do volume** (por isso exporte
> usando o volume como referência). Tamanho divergente faz o `finalize` abortar.
> Caso **realmente não exista lesão**, pule esta parte e use `--no-lesion` no
> `finalize` (Parte 5).

---

## Parte 5 — Execução real, Fase 2: `finalize`

Estágios 4b–7: importa a lesão + refino + malha + STL + manifesto do visualizador.

```powershell
.\.venv\Scripts\python.exe digital_twin.py finalize casos\paciente001 --profile profiles\figado.yaml
```

Caso **sem lesão** (escolha explícita — o pipeline não fabrica nada):
```powershell
.\.venv\Scripts\python.exe digital_twin.py finalize casos\paciente001 --profile profiles\figado.yaml --no-lesion
```

**Critério de passe:** `[OK] 'finalize' concluído`. Em
`casos\paciente001\outputs\`:
- `figado_orgao.stl`, `figado_lesao.stl` (em LPS);
- `viewer_manifest.json`.

---

## Parte 6 — Visualizar no navegador

Há duas formas.

### 6.1 Servido (recarga automática) — recomendado

```powershell
.\.venv\Scripts\python.exe -m http.server 8000
```
Abra no navegador (ajuste o caminho do caso após `case=`):
```
http://localhost:8000/viewer/index.html?case=../casos/paciente001/outputs
```
Para o caso sintético da Parte 2:
```
http://localhost:8000/viewer/index.html?case=../casos/sintetico/outputs
```
Pare o servidor com **Ctrl+C** quando terminar.

### 6.2 Arrastar e soltar (sem servidor)

1. Dê duplo clique em `viewer\index.html`.
2. Arraste para a área indicada **todo o conteúdo** da pasta `outputs\` do caso
   (o `viewer_manifest.json` **e** os arquivos `.stl`).

**Critério de passe:** o órgão (translúcido) e a lesão aparecem; orbitam com o
mouse (arrastar = girar, scroll = zoom); o painel à direita mostra os metadados e
o aviso “NÃO destinado a decisão clínica”. Caixas de seleção e controles de
opacidade ligam/desligam cada estrutura.

---

## Parte 7 — Trocar de órgão (ex.: baço)

A regra de domínio mora em **perfil (config)**, não no código. Para um novo órgão:

1. Copie o perfil:
   ```powershell
   Copy-Item profiles\figado.yaml profiles\baco.yaml
   ```
2. Edite `profiles\baco.yaml`: ajuste `id: baco`, `nome_exibicao`, e o alvo da
   segmentação `segmentacao_orgao.rotulo_alvo: spleen`.
   (Para descobrir nomes de classe válidos:
   `.\.venv\Scripts\python.exe -m totalsegmentator ...` ou
   `totalseg_info --classes -ta total_mr`.)
3. Rode tudo com `--profile profiles\baco.yaml`. **O motor não muda.** As saídas
   sairão como `baco_orgao.stl` / `baco_lesao.stl`.

---

## Parte 8 — Solução de problemas

| Mensagem / sintoma | Causa provável | O que fazer |
|---|---|---|
| `TotalSegmentator não está instalado` | Ambiente sem o extra `[seg]` | `pip install -e .[seg]` (máquina com GPU) |
| `doctor` mostra `torch device: cpu` | GPU não visível ao torch | Conferir driver NVIDIA/CUDA; ou aceitar CPU (lento) |
| `Modalidade do exame (...) não bate` | Exame não é RM, ou perfil errado | Use o perfil correto; `figado.yaml` espera RM |
| `Saída de segmentação esperada não encontrada` | `rotulo_alvo` inválido p/ a task | Conferir nome: `totalseg_info --classes -ta total_mr` |
| `Segmentação automática não encontrou '<órgão>'` | O órgão não aparece no exame | Revisão humana; não há órgão a modelar |
| `Máscara de lesão ausente` | Faltou exportar a lesão no Slicer | Exporte em `mask_lesion.nii.gz`, ou use `--no-lesion` |
| `tamanho diferente do volume/órgão` | Lesão exportada com outra geometria | No Slicer, exporte usando **o volume** como referência |
| `Refino zerou a máscara...` | Parâmetros de refino mal calibrados | Afrouxe `refino` no perfil (min_volume_voxels) |
| Wheels falham ao instalar | Python ≠ 3.13 | Use `py -3.13`; 3.14 ainda não tem wheels |
| Viewer não carrega via `?case=` | `fetch` exige http | Use o `http.server` (6.1) ou o arrastar-e-soltar (6.2) |

---

## Apêndice — Checklist de uma execução real completa

```
[ ] (B) venv 3.13 + pip install -e .[seg]
[ ] doctor mostra "torch device: cuda"
[ ] prepare  → volume.nii.gz + mask_organ.nii.gz (não-vazio) + manifest.json
[ ] 3D Slicer → mask_lesion.nii.gz salvo no caminho exato
[ ] finalize → outputs/ com 2 STL + viewer_manifest.json
[ ] http.server + viewer → órgão + lesão renderizam e orbitam
```

Qualquer passo que **aborte** é um sinal para investigar — o pipeline foi feito
para parar, nunca para inventar dado.
