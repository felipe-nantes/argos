#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dtwin — motor determinístico e órgão-agnóstico do pipeline de Digital Twin.

API pública mínima. O comportamento por órgão vem dos perfis em profiles/
(config), nunca do código. Ver contexto/04_ARQUITETURA.md.
"""
from .core import Case, PipelineError, load_profile
from .engine import Engine

__all__ = ["Engine", "Case", "PipelineError", "load_profile"]
__version__ = "0.1.0"
