# Visualizador (modo Pesquisa)

Visualizador 3D estático (Three.js, sem build) para os STLs gerados pelo pipeline.
**NÃO destinado a decisão clínica.** Coordenadas LPS.

> **Offline:** o Three.js (+ STLLoader + OrbitControls) está vendorizado em
> `viewer/vendor/` e resolvido por um importmap no `index.html`. Não há
> dependência de CDN — funciona sem internet (ex.: na apresentação).

## Uso rápido (drag & drop)

1. Abra `viewer/index.html` no navegador (duplo clique funciona).
2. Arraste para a área indicada o conteúdo da pasta `outputs/` de um caso
   (o `viewer_manifest.json` **e** os arquivos `.stl`).

## Uso servido (carregamento automático via ?case=)

Por restrição do navegador, `fetch` só funciona via http. Sirva a raiz do projeto:

```bash
python -m http.server 8000
```

Depois abra (ajuste o caminho do caso):

```
http://localhost:8000/viewer/index.html?case=../casos/sintetico/outputs
```

Controles: orbitar (arrastar), zoom (scroll), alternar visibilidade e opacidade de
órgão/lesão no painel à direita.

## Revisão integrada ao webapp

Quando aberto pelo botão do webapp, o viewer recebe também `?job=<id>` e mostra
as ações **Aprovar segmentação** e **Solicitar revisão**. A decisão é enviada ao
backend local e persistida no caso como `outputs/approval.json`.
