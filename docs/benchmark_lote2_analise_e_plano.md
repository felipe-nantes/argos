# Análise do benchmark do lote 2 e plano de melhoria do pipeline

Data: 2026-07-08  
Projeto: ARGOS / Digital Twin  
Contexto: benchmark backend local com lote positivo de RMs hepáticas em `C:\Users\profurg\Desktop\sander\dicoms\lote 2`

## 1. Resumo executivo

O benchmark do lote 2 mostrou um desempenho inicial abaixo do desejado para detecção de alterações hepáticas em RM, especialmente nos cenários com painel único e RAG textual. A melhor configuração testada foi a cobertura volumétrica sem RAG, que detectou 2 de 6 casos positivos.

O resultado ruim não deve ser interpretado simplesmente como “o MedGemma não funciona”. A leitura mais correta é que o pipeline ainda não entrega ao modelo evidência visual suficiente, consistente e estruturada para a tarefa. Quando o modelo recebeu cobertura volumétrica completa do fígado, houve melhora objetiva em relação ao baseline.

Resultado resumido:

| Cenário | Positivos detectados | Falhas técnicas | Sensibilidade principal |
|---|---:|---:|---:|
| Baseline, painel único `uniform_9` | 0/6 | 1 | 0.0% |
| RAG textual + painel único | 0/6 | 2 | 0.0% |
| Volumétrico `volumetric_blocks` | 2/6 | 1 | 33.3% |
| Volumétrico + RAG textual | 0/6 | 6 | 0.0% |

Como o lote continha apenas casos positivos, este benchmark avalia sensibilidade. Ele não permite estimar especificidade.

## 2. O que o benchmark realmente demonstrou

O teste demonstrou quatro pontos importantes:

1. O painel baseline com poucos cortes é insuficiente para lesões focais hepáticas.
2. A cobertura volumétrica melhora a chance de o modelo enxergar a alteração.
3. O RAG textual, no formato atual, não melhorou a detecção e aumentou falhas de schema.
4. O benchmark runner oficial ainda precisa de ajuste para rodar corretamente com NIfTI já preparado e política de anonimização.

O achado mais relevante é que o modo volumétrico transformou alguns casos antes classificados como negativos em positivos. Isso sugere que parte relevante do erro vem da representação visual, não apenas do raciocínio do modelo.

## 3. Por que o resultado foi ruim

### 3.1. O baseline mostra poucos cortes do fígado

O baseline usa uma estratégia limitada de amostragem, com poucos cortes distribuídos pelo volume hepático. Esse tipo de painel é rápido e simples, mas tem uma limitação óbvia: uma lesão pequena ou localizada fora dos cortes escolhidos pode simplesmente não aparecer.

Se a lesão não está visível no painel, o MedGemma não tem como detectá-la. Nesse cenário, o erro não é exatamente do modelo, mas da evidência visual fornecida a ele.

Essa hipótese foi reforçada pelo resultado do cenário volumétrico:

- baseline: 0 positivos detectados;
- volumétrico: 2 positivos detectados;
- nos relatórios válidos do volumétrico, o gate de cobertura hepática foi 100%.

Portanto, a primeira prioridade deve ser melhorar aquilo que o modelo enxerga.

### 3.2. A tarefa depende de exame multiparamétrico, não de uma única série

RM hepática não costuma ser interpretada apenas por uma imagem isolada. A avaliação de lesões focais depende da combinação de fases e sequências:

- pré-contraste;
- arterial;
- portal/venosa;
- tardia/equilíbrio;
- T2;
- DWI;
- ADC;
- eventualmente sequências adicionais.

No benchmark do lote 2, alguns casos exigiram seleção manual ou alternativa de séries, porque havia séries MPR inadequadas, incompatibilidades de dimensão ou problemas de amostragem. Isso significa que, mesmo quando o painel foi tecnicamente válido, ele pode não ter representado a melhor evidência radiológica disponível no exame.

Uma lesão pode ser pouco evidente em uma fase e muito evidente em outra. Se o pipeline envia ao modelo apenas uma série menos informativa, o resultado tende a ser falso negativo.

### 3.3. Alguns casos tinham estrutura DICOM difícil

Durante a preparação do lote, foram observados problemas como:

- avisos de amostragem não uniforme;
- possível ausência de cortes;
- séries reconstruídas/MPR que não eram ideais para análise axial primária;
- incompatibilidade de dimensões em algumas séries;
- necessidade de selecionar manualmente séries alternativas.

Esses problemas não impedem necessariamente a inferência, mas reduzem a confiabilidade do benchmark. Eles também mostram que o pipeline precisa de uma etapa mais forte de seleção, validação e classificação das séries antes da geração do painel.

### 3.4. O benchmark local usou MedGemma 4B, não o 27B externo

O teste foi executado com o MedGemma 1.5 4B local, em CUDA, com quantização NF4. Esse modelo é útil para desenvolvimento e benchmark operacional, mas não deve ser tratado como substituto direto do MedGemma 27B que vinha sendo usado em outro equipamento.

O 4B pode ser suficiente para validar o pipeline, testar painéis, medir falhas técnicas e comparar estratégias de representação visual. Porém, a conclusão final sobre acurácia deve ser repetida com o 27B quando a infraestrutura estiver disponível.

### 3.5. O RAG textual piorou a aderência ao schema

O RAG textual não melhorou a detecção neste lote. Mais importante: ele aumentou a taxa de respostas inválidas.

Foram observadas falhas como:

- resposta sem JSON válido;
- retorno literal de template, como `POSITIVA | NEGATIVA | INCONCLUSIVA`;
- valores inválidos em campos estruturados;
- inconsistência no campo de confiança.

Isso indica que o contexto textual está competindo com a instrução de saída estruturada. Em vez de ajudar o modelo a classificar melhor a imagem, o RAG parece ter aumentado a complexidade do prompt e reduzido a disciplina de resposta.

Nesse momento, o RAG não deve ser usado como parte central da decisão visual. Ele deve ser reposicionado como apoio posterior à organização do relatório, padronização de critérios e explicação, não como reforço direto da primeira inferência visual.

### 3.6. Falhas técnicas entram como erro na métrica principal

No protocolo correto, um caso que falha tecnicamente não pode ser ignorado nem convertido em negativo. Se o modelo não entrega uma resposta válida, o pipeline não produziu uma decisão confiável.

Por isso, respostas inválidas, JSON quebrado ou schema inválido devem contar como erro na métrica principal.

Isso torna a métrica mais dura, mas mais honesta.

## 4. Problema identificado no benchmark runner oficial

A primeira tentativa de rodar o benchmark usando `dtwin.medgemma_benchmark` não produziu métrica real de inferência.

O motivo foi uma incompatibilidade entre componentes:

1. o gerador de painel exige que o manifesto do caso tenha `case_id` começando com `anon-*`;
2. ele também exige `policy: anonymize`;
3. ao importar NIfTI já preparado, o benchmark runner recria o manifesto com `policy: public_dataset_sanitized`;
4. isso faz o pipeline abortar antes da inferência.

Portanto, os resultados iniciais do runner oficial foram descartados. Para obter inferência real neste lote, os casos foram executados diretamente por `dtwin.medgemma_screening` usando os manifests originais anonimizados, gerados pelo `Engine.prepare`.

Esse ponto precisa ser corrigido antes de usar o benchmark runner como ferramenta oficial para comparações maiores.

## 5. Melhorias recomendadas

### 5.1. Corrigir o benchmark runner para NIfTI anonimizados

Primeira correção recomendada:

- permitir que o benchmark runner preserve `policy: anonymize` quando o volume já vem de uma preparação anonimizada;
- ou permitir que datasets públicos sanitizados passem por uma política explicitamente autorizada;
- evitar que o runner recrie um manifesto incompatível com o gerador de painel;
- manter as salvaguardas de PHI, sem relaxar a proteção por conveniência.

Critério de aceite:

- `dtwin.medgemma_benchmark` deve conseguir rodar os mesmos casos até a inferência real;
- os manifests devem continuar sem PHI;
- `case_id` deve permanecer anônimo;
- a execução deve produzir `medgemma_report.json` por caso válido.

### 5.2. Adicionar retry e reparo de JSON

O pipeline deve tratar respostas fora do schema de forma mais robusta.

Fluxo sugerido:

1. primeira chamada normal ao MedGemma;
2. se a resposta não for JSON válido, fazer uma segunda chamada curta pedindo apenas correção para JSON;
3. se o JSON for válido mas tiver campo inválido, tentar reparo controlado;
4. se ainda falhar, marcar como `INVALID_RESPONSE`;
5. registrar texto bruto, erro, tentativa e motivo da falha para auditoria.

Essa melhoria não deve mascarar erro clínico. Ela serve apenas para reduzir falhas técnicas de formatação.

Critério de aceite:

- falhas por JSON inválido devem cair;
- nenhuma resposta reparada deve inventar achados;
- o relatório deve indicar se houve reparo;
- o benchmark deve separar erro técnico de falso negativo.

### 5.3. Usar o volumétrico como cenário principal sem RAG

O cenário `volumetric_blocks` foi o único que trouxe ganho objetivo de sensibilidade.

Recomendação:

- manter o baseline `uniform_9` apenas como controle reprodutível;
- usar o volumétrico como candidato principal para novos benchmarks;
- sempre registrar número de painéis;
- sempre validar `covered_liver_voxels == total_liver_voxels`;
- interromper inferência se a cobertura estiver incompleta.

Critério de aceite:

- todos os casos volumétricos devem demonstrar cobertura hepática 100%;
- cada painel deve ter hash persistido;
- cada resposta por painel deve ser auditável;
- uma falha em painel intermediário deve invalidar o caso inteiro.

### 5.4. Melhorar seleção automática de séries

Antes de gerar qualquer painel, o pipeline precisa entender melhor o exame.

Objetivo:

- identificar quais séries representam fases e sequências relevantes;
- evitar séries MPR inadequadas como entrada principal;
- rejeitar ou sinalizar séries com amostragem ruim;
- escolher automaticamente as séries mais úteis para análise hepática.

Metadados úteis:

- `SeriesDescription`;
- `ProtocolName`;
- `ImageType`;
- tempo pós-contraste, quando disponível;
- orientação;
- espessura de corte;
- número de slices;
- consistência de spacing;
- presença de `b-value` para DWI;
- séries ADC derivadas.

Critério de aceite:

- o pipeline deve classificar séries por provável tipo: arterial, portal, tardia, T2, DWI, ADC, pré-contraste ou desconhecida;
- deve registrar a escolha no manifesto;
- deve avisar quando a escolha for fraca ou ambígua;
- deve permitir revisão humana da seleção.

### 5.5. Criar painéis por fase e sequência

O próximo avanço natural é deixar de enviar apenas um volume ou um painel composto. Para RM hepática, o modelo deve receber evidências separadas por fase.

Estratégia sugerida:

- painel volumétrico arterial;
- painel volumétrico portal/venoso;
- painel volumétrico tardio;
- painel T2;
- painel DWI;
- painel ADC;
- painel resumo/fusão apenas como complemento.

O modelo deve analisar cada painel como uma evidência parcial. Depois, a decisão final deve ser agregada de forma determinística.

Regra inicial possível:

1. qualquer painel com achado positivo consistente → caso suspeito/positivo;
2. se nenhum painel positivo, mas algum painel inconclusivo → caso inconclusivo;
3. negativo apenas quando todos os painéis forem negativos e a qualidade for suficiente.

Critério de aceite:

- cada painel deve indicar fase/sequência;
- cada painel deve ter hash e manifesto;
- o relatório final deve dizer qual fase sustentou a decisão;
- a agregação deve ser reproduzível.

### 5.6. Adicionar mapas determinísticos de realce e washout

Sem treinar modelo, ainda é possível criar evidências derivadas úteis.

Exemplos:

- subtração pré-contraste versus arterial;
- arterial versus portal;
- arterial versus tardio;
- mapa de realce relativo;
- mapa de washout;
- harmonização de DWI/ADC;
- normalização por parênquima hepático.

Esses mapas não devem ser tratados como diagnóstico automático. Eles devem ser apresentados ao MedGemma como evidência visual auxiliar.

Objetivo:

- tornar padrões sutis de realce mais visíveis;
- reduzir dependência da percepção direta em imagens originais;
- ajudar especialmente em lesões pequenas ou pouco contrastadas.

Critério de aceite:

- mapas devem ser determinísticos;
- fórmula e parâmetros devem ser versionados;
- não devem usar máscara de lesão;
- devem ser auditáveis e reproduzíveis.

### 5.7. Reposicionar o RAG

O RAG textual não deve ser removido do projeto, mas seu papel precisa mudar.

Uso não recomendado neste momento:

- inserir muito contexto textual dentro da chamada visual principal;
- pedir ao modelo para combinar imagem, literatura e decisão em um único passo;
- usar RAG para tentar compensar painel visual incompleto.

Uso recomendado:

- padronizar linguagem do relatório;
- fornecer critérios radiológicos após a decisão visual;
- ajudar a explicar limitações;
- apoiar a revisão humana;
- gerar checklist estruturado de critérios;
- reduzir alucinação textual no laudo final.

Fluxo sugerido:

1. visão primeiro: MedGemma analisa painéis sem RAG pesado;
2. decisão estruturada;
3. RAG textual recupera critérios relevantes;
4. relatório final é organizado com apoio do RAG;
5. revisão humana obrigatória.

Critério de aceite:

- RAG não deve aumentar falhas de schema;
- saída visual deve continuar JSON estrito;
- contexto recuperado deve ser citado/auditável;
- o relatório deve separar “achado observado” de “critério textual recuperado”.

### 5.8. Separar erro técnico de erro clínico

O benchmark deve reportar mais de uma métrica.

Métricas recomendadas:

- sensibilidade principal, contando falhas e inconclusivos como erro;
- sensibilidade entre respostas válidas;
- taxa de falha técnica;
- taxa de inconclusivos;
- taxa de JSON inválido;
- tempo médio por caso;
- tempo médio por painel;
- número médio de painéis;
- cobertura hepática;
- taxa de casos com seleção de série ambígua.

Isso evita duas distorções:

1. esconder falhas técnicas removendo casos difíceis;
2. confundir falso negativo real com falha de infraestrutura/schema.

## 6. Ordem prática recomendada

### Etapa 1 — Corrigir infraestrutura do benchmark

Prioridade máxima.

Tarefas:

- corrigir incompatibilidade de manifesto no runner oficial;
- garantir que NIfTI preparado com política anonimizada rode sem abortar;
- preservar salvaguardas de PHI;
- gerar relatório oficial por cenário;
- validar que o webapp e CLI usam a mesma lógica.

Resultado esperado:

- benchmark reproduzível sem execução manual caso a caso.

### Etapa 2 — Robustecer schema e retries

Tarefas:

- implementar retry para JSON inválido;
- implementar reparo controlado de schema;
- registrar tentativas;
- diferenciar `INVALID_RESPONSE` de `NEGATIVA`;
- garantir que falha técnica não vire decisão clínica.

Resultado esperado:

- menor perda de casos por formatação;
- métricas mais confiáveis.

### Etapa 3 — Consolidar volumétrico sem RAG

Tarefas:

- manter `volumetric_blocks` como cenário principal;
- validar cobertura 100%;
- auditar painéis e hashes;
- medir sensibilidade em lote maior;
- comparar contra baseline.

Resultado esperado:

- medir o ganho real da cobertura volumétrica.

### Etapa 4 — Seleção multiparamétrica de séries

Tarefas:

- classificar séries por fase/sequência;
- selecionar automaticamente melhores séries;
- criar warnings de qualidade;
- permitir revisão humana da seleção.

Resultado esperado:

- modelo recebe as imagens certas, não apenas uma série qualquer.

### Etapa 5 — Painéis por fase

Tarefas:

- gerar painéis separados por fase;
- criar prompt parcial por painel;
- agregar decisão por regra determinística;
- registrar fase que sustentou a positividade.

Resultado esperado:

- maior sensibilidade para lesões dependentes de realce/washout.

### Etapa 6 — Mapas determinísticos

Tarefas:

- gerar mapas de subtração e realce;
- validar registro entre fases;
- incluir mapas como evidência complementar;
- medir ganho versus painéis originais.

Resultado esperado:

- evidências visuais mais explícitas para o modelo.

### Etapa 7 — Reavaliar RAG com papel restrito

Tarefas:

- reduzir contexto na chamada visual;
- mover RAG para etapa pós-decisão;
- validar se melhora consistência do relatório;
- não usar RAG se ele aumentar erro de schema.

Resultado esperado:

- RAG útil para relatório e revisão, não prejudicial para classificação.

### Etapa 8 — Repetir com MedGemma 27B

Tarefas:

- rodar os mesmos cenários no 27B externo;
- comparar 4B versus 27B;
- manter mesmo lote, mesmas configurações e mesmos manifests;
- comparar tempo, sensibilidade, falhas técnicas e consistência.

Resultado esperado:

- separar limitação do modelo 4B de limitação do pipeline.

## 7. Próximo candidato de benchmark

O próximo candidato recomendado é:

`MedGemma 4B + volumetric_blocks sem RAG`

Motivo:

- foi o único cenário com melhora mensurável;
- manteve cobertura hepática 100% nos relatórios válidos;
- não sofreu colapso completo de schema como o volumétrico + RAG;
- é uma base mais clara para comparar com o MedGemma 27B.

Esse cenário ainda está longe da meta de 75%, mas é o melhor ponto de partida.

## 8. Conclusão

O benchmark do lote 2 foi ruim em termos de acurácia, mas útil em termos de diagnóstico do pipeline.

Ele mostrou que:

- o baseline com poucos cortes provavelmente perde lesões por falta de cobertura;
- a cobertura volumétrica melhora a detecção;
- o RAG textual atual atrapalha mais do que ajuda na inferência visual;
- o benchmark runner precisa de correção para produzir métricas oficiais;
- a próxima grande melhoria deve ser visual e estrutural, não treinamento de IA.

A direção recomendada é fortalecer o pipeline antes de qualquer fine-tuning:

1. benchmark oficial funcional;
2. schema robusto;
3. volumétrico estável;
4. seleção multiparamétrica de séries;
5. painéis por fase;
6. mapas determinísticos;
7. RAG apenas como apoio textual/revisão;
8. repetição no MedGemma 27B.

Somente depois disso fará sentido avaliar se o teto das melhorias sem treino foi atingido.

