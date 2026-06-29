# 07 · Infraestrutura e Custos

## Topologia do MVP: estação de trabalho única

O MVP roda **inteiro em uma máquina** (a workstation do HU): backend + modelo +
processamento + servidor web local. O navegador acessa a interface na própria
máquina ou na rede local. **Nenhum dado sai da máquina** — o que é ideal para
privacidade (`03_REGULATORIO_LGPD.md`) e zera custo de nuvem.

## Hardware necessário

O time informou ter **hardware considerável** e sem teto de orçamento fixo (avaliar
por necessidade). Para esta carga:

- **GPU** — é o item crítico. A inferência do nnU-Net (TotalSegmentator MRI) em
  CPU é lenta a ponto de inviabilizar o uso fluido; em GPU é prática. Uma GPU com
  **VRAM suficiente** (tipicamente ≥ 8–12 GB para inferência confortável; mais
  ajuda em volumes grandes) deve ser priorizada.
- **RAM** — volumes 3D de RM + processamento de máscara + mesh consomem memória;
  32 GB é um piso razoável, 64 GB confortável.
- **Armazenamento** — exames DICOM/NIfTI são pesados. Reservar espaço para: exames
  des-identificados, máscaras, meshes/STL e o **acervo do flywheel** (que cresce a
  cada caso). SSD para desempenho.
- **3D Slicer** roda na mesma estação para a etapa de revisão/lesão.

> Recomendação: dimensionar a GPU pela inferência do nnU-Net; é o gargalo. CPU/RAM
> são secundários. Validar a VRAM disponível antes de assumir que qualquer GPU
> serve.

## Modelo de custo

O MVP é essencialmente **capex (hardware que já existe) + opex perto de zero**:

- **Sem custo de nuvem** (roda local).
- **Sem custo de licença** das ferramentas centrais (SimpleITK, nnU-Net,
  TotalSegmentator, 3D Slicer, PyVista, FastAPI, Niivue/VTK.js são abertos).
- Custo real está em **tempo de engenharia** e em **hardware** (já disponível).

Isto preserva o padrão enxuto do time **sem** forçar arquitetura de nuvem cara — a
diferença em relação a projetos web típicos é que o "peso" aqui é local
(GPU/armazenamento), não recorrente.

## Origem dos exames

- **MVP:** **upload manual** de uma pasta DICOM. Simples, suficiente, e mantém o
  dado sob controle.
- **Futuro:** **integração com PACS/DICOMweb** do hospital, para puxar exames sem
  intervenção manual. Isto é fase posterior (`08_ROADMAP.md`) e adiciona requisitos
  de rede, autenticação e conformidade — por isso não entra no MVP.

A fronteira já está isolada: como o pipeline recebe "uma série DICOM" como entrada,
trocar a **fonte** (upload manual → PACS) no futuro não reescreve o pipeline.

## Caminho de escalonamento (e seus custos futuros)

Quando o MVP provar valor, os saltos prováveis e o que cada um implica:

1. **Estação única → servidor interno multiusuário.** Vários cirurgiões acessam
   pela rede do HU. Custo: um servidor (com GPU) + autenticação + fila de
   processamento. O pipeline não muda; muda onde roda e como controla acesso.
2. **Upload manual → PACS/DICOMweb.** Integração com o sistema do hospital. Custo:
   trabalho de integração + conformidade + acordos institucionais.
3. **Pseudonimização ativa.** Para o uso clínico, a chave paciente↔modelo entra em
   operação (ponto de extensão já reservado — `03_REGULATORIO_LGPD.md`).
4. **GPU dedicada / mais robusta** conforme volume de casos e modelos próprios
   (treino, não só inferência, é bem mais pesado).

## Resumo de decisões

- MVP roda local, em uma máquina, sem nuvem.
- GPU é o investimento-chave; dimensionar pela inferência do nnU-Net.
- Reservar armazenamento crescente para o acervo do flywheel.
- Upload manual agora; PACS depois, sem reescrever o pipeline.
