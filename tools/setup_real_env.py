#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bootstrap da máquina de execução REAL (caixa com GPU) em um comando.

Prepara o ambiente do jeito mais rápido: cria o venv, instala o núcleo + o extra
[seg] (TotalSegmentator + torch), e VERIFICA que está tudo pronto para rodar a
segmentação real — torch enxergando a GPU e o rótulo do órgão válido na task do
perfil. Opcionalmente já dispara o smoke ponta a ponta num exame real.

Use o Python do sistema (não precisa de venv ainda):

  # Windows
  py -3.13 tools\\setup_real_env.py
  # Linux/Mac
  python3.13 tools/setup_real_env.py

Opções úteis:
  --verify-only            só checa um ambiente já instalado (não cria/instala).
  --recreate               apaga o .venv e recria do zero.
  --smoke "C:/serie_dicom" após preparar, roda tools/smoke_gpu.py nesse exame.
  --lesion <arquivo> | --no-lesion   repassados ao smoke.
  --profile profiles/figado.yaml     perfil do órgão (default).
  --device gpu             device do TotalSegmentator (gpu|cpu|gpu:N).

Códigos de saída: 0 = ambiente pronto; 1 = algo falhou (ver o relatório).
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(cmd, capture=False) -> subprocess.CompletedProcess:
    print(">>", " ".join(str(c) for c in cmd))
    return subprocess.run(
        cmd, cwd=ROOT, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def step(msg: str) -> None:
    print("\n" + "=" * 70 + f"\n {msg}\n" + "=" * 70)


class Report:
    def __init__(self) -> None:
        self.fail = 0
        self.warn = 0

    def ok(self, m: str) -> None:
        print(f"  [OK]   {m}")

    def warning(self, m: str) -> None:
        self.warn += 1
        print(f"  [AVISO] {m}")

    def error(self, m: str) -> None:
        self.fail += 1
        print(f"  [FALHA] {m}")


def check_python_version(rep: Report) -> None:
    v = sys.version_info
    label = f"{v.major}.{v.minor}.{v.micro}"
    if (3, 10) <= (v.major, v.minor) < (3, 14):
        rep.ok(f"Python {label} (compatível: >=3.10,<3.14)")
    else:
        rep.error(f"Python {label} fora da faixa suportada (>=3.10,<3.14). Use py -3.13.")


def ensure_venv(rep: Report, recreate: bool) -> bool:
    if recreate and VENV.exists():
        print(f"  removendo venv existente: {VENV}")
        shutil.rmtree(VENV)
    if VENV.exists():
        rep.ok(f"venv já existe: {VENV}")
        return True
    print(f"  criando venv em {VENV} ...")
    venv.create(VENV, with_pip=True)
    if venv_python().exists():
        rep.ok(f"venv criado: {VENV}")
        return True
    rep.error("falha ao criar o venv.")
    return False


def install_seg(rep: Report) -> None:
    py = venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    r = run([py, "-m", "pip", "install", "-e", ".[seg]"])
    if r.returncode == 0:
        rep.ok("instalação do núcleo + extra [seg] concluída.")
    else:
        rep.error("pip install -e .[seg] falhou (veja a saída acima).")


def check_doctor(rep: Report) -> None:
    py = venv_python()
    r = run([py, "digital_twin.py", "doctor"], capture=True)
    out = r.stdout or ""
    print(out)
    if "torch device: cuda" in out:
        rep.ok("torch enxerga a GPU (cuda).")
    elif "torch device: cpu" in out:
        rep.warning("torch só vê CPU — segmentação roda, mas LENTA. Confira driver/CUDA.")
    else:
        rep.error("torch ausente/sem device — o extra [seg] não está pronto.")
    if "TotalSegmentator importável" in out:
        rep.ok("TotalSegmentator importável.")
    else:
        rep.error("TotalSegmentator não importável — rode com instalação (sem --verify-only).")


def check_organ_label(rep: Report, profile_path: str) -> None:
    """Confirma que o rótulo do órgão do perfil é classe válida da task (best-effort)."""
    py = venv_python()
    snippet = (
        "import sys, yaml\n"
        "p = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))\n"
        "seg = p['segmentacao_orgao']; label = seg['rotulo_alvo']; task = seg.get('motor_task','total_mr')\n"
        "try:\n"
        "    from totalsegmentator.map_to_binary import class_map\n"
        "    classes = set(class_map[task].values()) if task in class_map else set()\n"
        "    if not classes:\n"
        "        print('SKIP task', task); sys.exit(2)\n"
        "    print('OK' if label in classes else 'MISS', label, task)\n"
        "    sys.exit(0 if label in classes else 3)\n"
        "except Exception as e:\n"
        "    print('SKIP', e); sys.exit(2)\n"
    )
    r = run([py, "-c", snippet, profile_path], capture=True)
    out = (r.stdout or "").strip()
    if r.returncode == 0:
        rep.ok(f"rótulo do órgão válido na task — {out}")
    elif r.returncode == 3:
        rep.error(f"rótulo do órgão NÃO existe na task — {out}. Ajuste o perfil.")
    else:
        rep.warning(
            f"não foi possível validar o rótulo automaticamente ({out}). "
            "Confira manualmente: totalseg_info --classes -ta total_mr"
        )


def run_smoke(rep: Report, args) -> None:
    py = venv_python()
    cmd = [py, "tools/smoke_gpu.py", "--dicom", args.smoke,
           "--profile", args.profile, "--device", args.device]
    if args.lesion:
        cmd += ["--lesion", args.lesion]
    elif args.no_lesion:
        cmd += ["--no-lesion"]
    r = run(cmd)
    if r.returncode == 0:
        rep.ok("smoke ponta a ponta passou.")
    else:
        rep.error("smoke ponta a ponta falhou (veja o relatório do smoke acima).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bootstrap da caixa GPU (execução real).")
    ap.add_argument("--profile", default="profiles/figado.yaml", help="Perfil do órgão (YAML).")
    ap.add_argument("--device", default="gpu", help="Device do TotalSegmentator (gpu|cpu|gpu:N).")
    ap.add_argument("--verify-only", action="store_true", help="Só verificar; não criar/instalar.")
    ap.add_argument("--recreate", action="store_true", help="Apagar e recriar o .venv.")
    ap.add_argument("--smoke", metavar="DICOM", help="Após preparar, rodar o smoke nesse exame.")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--lesion", help="Máscara de lesão pronta para o smoke.")
    grp.add_argument("--no-lesion", action="store_true", help="Rodar o smoke sem lesão.")
    args = ap.parse_args(argv)

    print("=" * 70)
    print(" SETUP DO AMBIENTE REAL — Digital Twin (caixa com GPU)")
    print(" MODO PESQUISA — saída NÃO destinada a decisão clínica.")
    print("=" * 70)

    rep = Report()

    step("1/5 Python")
    check_python_version(rep)

    if not args.verify_only:
        step("2/5 venv")
        if not ensure_venv(rep, args.recreate):
            return _final(rep)
        step("3/5 Instalação (núcleo + [seg]) — pode demorar (torch é grande)")
        install_seg(rep)
    else:
        step("2-3/5 (--verify-only: pulando criação/instalação)")
        if not venv_python().exists():
            rep.error(f"venv não encontrado em {VENV}. Rode sem --verify-only primeiro.")
            return _final(rep)

    step("4/5 Verificação do ambiente (doctor + GPU + rótulo do órgão)")
    check_doctor(rep)
    check_organ_label(rep, args.profile)

    if args.smoke:
        step("5/5 Smoke ponta a ponta no exame real")
        run_smoke(rep, args)
    else:
        step("5/5 Smoke (pulado)")
        print("  Sem --smoke. Quando tiver um exame real, rode:")
        print(f"    {venv_python()} tools/smoke_gpu.py --dicom \"<serie_dicom>\" --no-lesion")

    return _final(rep)


def _final(rep: Report) -> int:
    print("\n" + "-" * 70)
    if rep.fail == 0:
        extra = f" ({rep.warn} aviso[s] — revise acima)" if rep.warn else ""
        print(f"AMBIENTE PRONTO{extra}.")
        return 0
    print(f"AMBIENTE NÃO PRONTO: {rep.fail} falha(s), {rep.warn} aviso(s). Resolva acima.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
