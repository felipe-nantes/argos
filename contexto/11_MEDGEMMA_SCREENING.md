# 11 · Triagem visual hepática com MedGemma

> **Modo Pesquisa.** Este fluxo gera somente uma **hipótese visual exploratória**.
> Não é diagnóstico, não é laudo médico, não recomenda conduta e não substitui
> radiologista, cirurgião ou médico pesquisador. Toda saída fica
> `pending_review` e exige revisão humana.

## Posição na arquitetura

O MedGemma é um fluxo **paralelo e opcional após o `prepare`**. Ele consome os
artefatos já des-identificados e revisáveis do caso:

```text
volume.nii.gz + mask_organ.nii.gz + manifest.json
    -> painel 2D sem máscara de lesão
    -> adaptador MedGemma configurável
    -> resposta JSON validada
    -> medgemma_report.json (pending_review)
```

Ele não altera os sete estágios existentes, não participa da segmentação do órgão
e não recebe `mask_lesion.nii.gz`. A montagem e os gates estão em
`dtwin/medgemma_panel.py`; configuração, prompt, cliente e parser em
`dtwin/medgemma_client.py`; a CLI/orquestração em
`dtwin/medgemma_screening.py`.

## Montagem 2D

O painel `medgemma_liver_screening_panel.png` contém:

- nove fatias axiais distribuídas uniformemente pela extensão do fígado (3×3);
- uma vista coronal e uma sagital no centroide da máscara;
- RM em escala de cinza com janela robusta por percentis;
- somente o **contorno** hepático, sem esconder o sinal do parênquima;
- nenhum texto identificável e nenhum metadado textual no PNG;
- aviso visível de modo Pesquisa.

O manifest `medgemma_liver_screening_manifest.json` registra as vistas, modelo
configurado, `lesion_pre_marked: false`, rastreabilidade e revisão obrigatória.

PHI queimada diretamente nos pixels não pode ser excluída com segurança por uma
regra automática genérica. Por isso o painel pode ser gerado para inspeção, mas a
inferência só é liberada após revisão visual explícita com
`--confirm-no-visible-phi`.

## Gerar e revisar somente o painel

Na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m dtwin.medgemma_screening `
  --case-dir casos\caso_real_001 `
  --medgemma-config configs\medgemma_4b.yaml `
  --panel-only
```

## Configurar o backend 4B

A configuração inicial é `configs/medgemma_4b.yaml`:

- versão: `MedGemma 1.5 4B Instruction-Tuned`;
- id oficial: `google/medgemma-1.5-4b-it`;
- provider: `http_json_v1`;
- endpoint local padrão: `http://127.0.0.1:8001/generate`;
- timeout: 120 s;
- backend/modelo começam como **indisponíveis**, pois o modelo gated não está
  instalado automaticamente por este repositório.

O gateway local já está implementado em `tools/medgemma_server.py`. A config
`medgemma_4b.yaml` permanece fail-closed; para o backend operacional use
`medgemma_local_4b.yaml`, depois de aceitar a licença e baixar os pesos.

```powershell
$env:MEDGEMMA_BACKEND_CONFIGURED="true"
$env:MEDGEMMA_MODEL_AVAILABLE="true"
$env:MEDGEMMA_ENDPOINT_URL="http://127.0.0.1:8001/generate"
```

O gateway `dtwin-medgemma-v1` recebe JSON com `model_id`, `model_version`,
`prompt`, PNG em base64 e limites de geração. Ele deve devolver:

```json
{
  "model_id": "google/medgemma-1.5-4b-it",
  "model_version": "MedGemma 1.5 4B Instruction-Tuned",
  "report": {
    "resultado_hipotese": "POSITIVA | NEGATIVA | INCONCLUSIVA",
    "resumo_do_achado": "string",
    "localizacao_aproximada": "string",
    "sinais_visuais_observados": ["string"],
    "confianca": "baixa | moderada | alta",
    "limitacoes_da_analise": ["string"],
    "necessidade_de_revisao_humana": true
  }
}
```

Se o backend não ecoar exatamente o id e a versão solicitados, a resposta é
descartada.

### Instalação local completa (RTX 4060 / Windows)

O runtime usa a mesma API do exemplo oficial: Transformers multimodal,
`AutoModelForImageTextToText`, `AutoProcessor` e BitsAndBytes 4-bit. Nesta máquina,
as dependências já podem ser instaladas/recriadas com:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[medgemma]"
```

O modelo é **gated**. A aceitação dos termos e o login pertencem à conta do
usuário e não podem ser automatizados pelo projeto:

1. Acesse `https://huggingface.co/google/medgemma-1.5-4b-it`, entre na sua conta e
   aceite os termos de acesso.
2. Crie um token de leitura em `https://huggingface.co/settings/tokens`.
3. No PowerShell, na raiz do projeto, execute:

   ```powershell
   .\.venv\Scripts\hf.exe auth login
   ```

4. Cole o token **somente no terminal**. Nunca grave o token em YAML, Git, log,
   prompt ou conversa.
5. Verifique a autorização e baixe os pesos:

   ```powershell
   .\.venv\Scripts\python.exe tools\setup_medgemma.py --verify-only
   .\.venv\Scripts\python.exe tools\setup_medgemma.py
   .\.venv\Scripts\python.exe tools\setup_medgemma.py --local-only
   ```

6. Inicie o backend oculto e aguarde o modelo carregar:

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\start_medgemma.ps1
   ```

   Estado e logs ficam em `.medgemma/` (gitignored). Confirme manualmente, se
   necessário:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8001/health
   ```

7. Para encerrar:

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\stop_medgemma.ps1
   ```

O perfil operacional é `configs/medgemma_local_4b.yaml`, com timeout de 300 s e
quantização `bitsandbytes-nf4`. Não há fallback para CPU ou para outro modelo.

## Executar a triagem completa

Após revisar visualmente o painel:

```powershell
.\.venv\Scripts\python.exe -m dtwin.medgemma_screening `
  --case-dir casos\caso_real_001 `
  --medgemma-config configs\medgemma_local_4b.yaml `
  --confirm-no-visible-phi
```

Sem backend/modelo configurado a execução **aborta** e não cria relatório falso.
Quando válida, a saída é `medgemma_report.json`, sempre com status
`pending_review`, versão do modelo e disclaimer de pesquisa.

## Migração futura para 27B

Use `configs/medgemma_27b.yaml`. O MedGemma 1.5 oficial está disponível somente
como 4B multimodal; as variantes 27B atuais pertencem à geração MedGemma 1.
Portanto, não existe um identificador oficial `MedGemma 1.5 27B`: o arquivo
mantém `model_id: null` e
`model_available: false`. Isso é um gate deliberado. Quando a versão correta
existir, basta ajustar o arquivo (ou as variáveis `MEDGEMMA_MODEL_ID`,
`MEDGEMMA_MODEL_VERSION`, `MEDGEMMA_MODEL_AVAILABLE`) e o endpoint/adaptador do
backend, sem alterar painel, parser ou pipeline. O timeout inicial é 300 s.

O `google/medgemma-27b-it` existente não é rotulado automaticamente como “1.5
27B”, porque isso quebraria a rastreabilidade solicitada.

## Painel multifásico (fusão RGB) — maior acurácia

O painel de fase única em cinza tem baixa sensibilidade: em testes com o dataset
público TCGA-LIHC (pacientes com HCC confirmado, cujas máscaras de tumor **não**
são mostradas ao modelo), o 4B chamava lesões óbvias de `NEGATIVA`. A causa não é
só o tamanho do modelo — uma fase única em cinza descarta a **dinâmica de realce**
(realce arterial + *washout* tardio) que define o HCC.

O modo `panel.mode: multiphase_fusion` (`configs/medgemma_local_4b_multiphase.yaml`,
módulo `dtwin/medgemma_panel_multiphase.py`) pré-computa esse sinal nos pixels:

- funde fases de RM **co-registradas** em canais RGB (padrão: R=arterial,
  G=portal-venoso, B=tardio) — lesões com washout aparecem avermelhadas;
- **recorta** no bounding-box do fígado (usa só a máscara do órgão) e **janela**
  a intensidade dentro do fígado, por fase, ampliando a conspícuidade;
- o prompt explica a codificação de cor ao modelo;
- `max_output_tokens` maior evita que o raciocínio estoure antes do JSON.

Executar (as fases precisam estar na mesma grade/geometria da máscara):

```powershell
.\.venv\Scripts\python.exe -m dtwin.medgemma_screening `
  --case-dir casos\caso_real_001 `
  --phase art=casos\caso_real_001\phase_art.nii.gz `
  --phase pv=casos\caso_real_001\phase_pv.nii.gz `
  --phase del=casos\caso_real_001\phase_del.nii.gz `
  --medgemma-config configs\medgemma_local_4b_multiphase.yaml `
  --confirm-no-visible-phi
```

Invariantes preservados: Cenário A (nenhuma máscara de lesão é lida ou renderizada),
PNG sem metadados, **revisão humana do painel e confirmação de PHI obrigatórias**,
`lesion_pre_marked: false`, rastreabilidade do modelo. O painel é uma **imagem
derivada** e é rotulado como tal (`input_type: mri_multiphase_rgb_fusion_liver_crop`,
`preserve_grayscale_signal: false`).

Ressalvas: as fases precisam estar co-registradas (a mesma grade não garante
ausência de movimento respiratório); o `prepare` atual ingere **uma** série, então
alimentar múltiplas fases em produção exige estender a ingestão; e a melhora de
sensibilidade observada é um sinal promissor em n pequeno — medir
sensibilidade/especificidade no conjunto rotulado (com controles sem tumor) é o
próximo passo antes de qualquer conclusão.

## Gates implementados

- caso anônimo, modalidade RM e estado PESQUISA;
- volume/máscara existentes, 3D, mesma geometria e máscara plausível/não vazia;
- ao menos nove fatias hepáticas;
- PNG sem metadados textuais;
- revisão visual de PHI antes de inferência;
- nenhuma entrada ou renderização de lesão pré-marcada;
- backend, modelo, endpoint, timeout e tamanho de entrada validados;
- endpoint local obrigatório quando `execution_mode: local`;
- resposta com schema exato e somente três estados;
- revisão humana sempre `true`;
- diagnóstico definitivo e recomendação de conduta rejeitados;
- relatório gravado somente após todos os gates.

## Parser tolerante (canonicalização)

O validador de resposta canonicaliza saída **semanticamente válida** ao vocabulário
exigido, sem fabricar conteúdo nem afrouxar a segurança:

- estado em outro idioma/caixa (`"NEGATIVE"`, `"Positiva"`) → `NEGATIVA`/`POSITIVA`;
- confiança (`"low"`, `"Moderate"`) → `baixa`/`moderada`;
- campo de lista que veio como string → lista de uma string; chaves extras descartadas.

O que **não** é recuperado (continua rejeitado): estados/valores não reconhecidos,
diagnóstico definitivo ou conduta clínica (gate de segurança), e resposta sem JSON.
O ganho é honestidade estatística: recupera respostas reais que a checagem estrita
descartava — inclusive respostas *erradas* do modelo, que assim deixam de se
esconder como "erro de formato".

## Limitações

- Uma montagem 2D perde informação do volume 3D completo.
- O modelo pode errar, omitir ou alucinar; o parser reduz risco textual, mas não
  transforma a saída em evidência clínica.
- A qualidade depende da sequência de RM, da máscara e da revisão humana.
- Há gateway FastAPI local, mas não há UI clínica, PACS nem serviço multiusuário.
- Execução remota fica bloqueada por padrão para evitar transmissão acidental de
  dado sensível.
