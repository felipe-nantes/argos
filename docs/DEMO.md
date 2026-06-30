# Roteiro de Apresentação (demo em ~5 min, sem GPU)

> **Modo Pesquisa.** Demonstração com **dados fictícios** (caso sintético). Nada
> aqui é exame de paciente nem decisão clínica.

Esta demo prova o pipeline **de ponta a ponta** (refino → malha → STL → viewer)
**sem GPU, sem DICOM, sem 3D Slicer**. O viewer roda **offline** (Three.js
vendorizado), então não depende da internet do local.

## Antes (uma vez, com internet, na máquina da apresentação)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe digital_twin.py doctor   # confere o núcleo
```

## Preparar o caso da demo (regenerável a qualquer momento)

```powershell
.\.venv\Scripts\python.exe tools\make_synthetic_case.py --out casos\demo
.\.venv\Scripts\python.exe digital_twin.py finalize casos\demo --profile profiles\figado.yaml
```

Confirme que existe `casos\demo\outputs\` com `figado_orgao.stl`,
`figado_lesao.stl` e `viewer_manifest.json`.

## Durante a apresentação

1. **(opcional) Mostre o preflight:** `digital_twin.py doctor` — explica que o
   sistema checa o ambiente e que a segmentação automática (estágio 3) roda na
   caixa com GPU; o resto roda em qualquer máquina.
2. **Suba o viewer:**
   ```powershell
   .\.venv\Scripts\python.exe -m http.server 8000
   ```
   Abra: `http://localhost:8000/viewer/index.html?case=../casos/demo/outputs`
3. **Narrativa de 60s no viewer:**
   - Banner vermelho: “Modo Pesquisa — NÃO destinado a decisão clínica”.
   - Órgão (translúcido) + lesão (sólida, vermelha) em 3D.
   - Orbitar (arrastar), zoom (scroll), ligar/desligar e opacidade no painel.
   - Painel mostra caso, órgão, **coordenadas LPS** e o disclaimer.
4. **Mensagem-chave:** “Regra de domínio mora em **config** (perfil YAML), não no
   código. Trocar de órgão = trocar de perfil; o motor não muda. O pipeline
   **nunca fabrica dado** — se algo falha, ele **aborta**.”

## Plano B (se a demo ao vivo falhar)

- **Viewer não abre via `?case=`:** use o arrastar-e-soltar — abra
  `viewer\index.html` (duplo clique) e arraste o conteúdo de `casos\demo\outputs\`.
- **Porta 8000 ocupada:** troque para `--bind 127.0.0.1 8080` e ajuste a URL.
- **Falha total do navegador:** tenha **screenshots/gravação** prontos (capture
  antes, durante o ensaio). Sugestão: grave um GIF/vídeo curto do viewer girando o
  modelo e tenha-o no slide como fallback.

## Ensaio (faça pelo menos uma vez)

Rode os dois blocos acima **na máquina e na rede reais da apresentação**. Os dois
riscos clássicos — porta ocupada e visualizador — só aparecem no ensaio.
