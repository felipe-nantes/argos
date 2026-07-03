# Sincronização entre workspaces (Windows ⇆ MAC)

Fluxo do projeto: **features são desenvolvidas no PC Windows**; **a execução só
roda no MAC** (Ollama + MedGemma 27B + segmentação). O Git é a ponte entre os
dois. Este documento é o contrato de sincronização — siga-o nas duas máquinas.

## Regra de ouro

> **Fez `push` na máquina onde editou → faça `pull` na outra antes de rodar.**

Nunca edite os mesmos arquivos nas duas máquinas sem sincronizar no meio.

## No PC Windows (onde se desenvolve)

```bash
git pull --rebase          # começa a sessão sempre atualizado
# ... edita as features ...
git add -A
git commit -m "feat: descrição curta da mudança"
git push
```

## No MAC (onde se executa)

```bash
git pull                   # traz o que veio do Windows

# SÓ se pyproject.toml ou requirements.txt tiverem mudado nesse pull:
.venv/bin/python -m pip install -e ".[webapp,medgemma,seg]"

# sobe os serviços (ordem):
#   1) ollama serve            (daemon + modelo medgemma:27b-it-bf16)
#   2) gateway  :8001          python tools/medgemma_server.py --config configs/medgemma_ollama_27b.yaml --port 8001
#   3) webapp   :8080          WEBAPP_MEDGEMMA_CONFIG=configs/medgemma_ollama_27b.yaml .venv/bin/python -m uvicorn webapp.server:app --port 8080
```

## O que NÃO é sincronizado (de propósito)

Cada máquina mantém o seu — estão no `.gitignore`:

| Item | Por quê |
|---|---|
| `.venv/`, `.venv-win/` | ambientes virtuais têm caminho absoluto por máquina; recrie localmente |
| `casos/`, `flywheel/`, `*.nii.gz`, `*.dcm`, `*.stl` | **dados de paciente / saídas** — nunca vão para o Git (LGPD) |
| `.claude/launch.json` | caminhos absolutos específicos da máquina |
| `__pycache__/`, `*.egg-info/`, `.pytest_cache/` | artefatos de build/execução |

**Consequência prática:** dependência nova = editar `pyproject.toml` /
`requirements.txt` (isso **é** versionado) → a outra máquina roda o `pip install`
depois do `pull`. O ambiente em si nunca viaja pelo Git.

## Situações comuns

- **Pasta nova com código** → `git add` normalmente (ela entra no repo).
- **Pasta nova com dados** → coloque sob `casos/` (já ignorado) ou adicione ao
  `.gitignore`. Dado de paciente **nunca** entra no Git.
- **As duas máquinas commitaram** → `git pull --rebase` mantém o histórico linear
  (sem "merge commits" de vaivém).
- **Conflito** → o Git marca o trecho; resolva no arquivo, `git add`, `git rebase --continue`.

## Primeira configuração do MAC (uma vez só)

### Cenário A — conectar a pasta que JÁ existe no MAC (recomendado agora)

O MAC já tem a pasta funcionando (`.venv`, `casos/`, dados). NÃO clone por cima —
conecte a pasta existente ao repositório. Como `.venv/` e `casos/` estão no
`.gitignore`, eles **não são tocados**; só os arquivos de código são atualizados
para a versão do repositório (que traz o visualizador 3D novo).

```bash
cd /Users/sander_gurgel/Documents/projetos_sander/argos-main

# segurança: backup dos arquivos de código que serão sobrescritos
mkdir -p /tmp/argos_bkp && cp -R webapp viewer /tmp/argos_bkp/ 2>/dev/null

git init -b main
git remote add origin <URL_DO_REPO>
git fetch origin
git reset --hard origin/main     # alinha o CÓDIGO ao repo; .venv e casos/ ficam intactos

# só se pyproject/requirements tiverem mudado:
.venv/bin/python -m pip install -e ".[webapp,medgemma,seg]"
```

### Cenário B — clone limpo (máquina nova, no futuro)

```bash
git clone <URL_DO_REPO> argos-main && cd argos-main
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[webapp,medgemma,seg]"
# recrie o .claude/launch.json local se for usar o preview (caminhos do MAC)
```
