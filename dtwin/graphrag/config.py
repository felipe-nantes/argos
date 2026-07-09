"""Configuração do GraphRAG Neo4j."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dtwin.core import PipelineError


CONFIG_SCHEMA = "argos-graphrag-neo4j-config-v1"


@dataclass(frozen=True)
class Neo4jConnectionConfig:
    uri: str
    user: str
    password_env: str
    database: str = "neo4j"

    @property
    def password(self) -> str:
        value = os.environ.get(self.password_env)
        if not value:
            raise PipelineError(f"Variável de ambiente ausente para Neo4j: {self.password_env}")
        return value


@dataclass(frozen=True)
class GraphRagConfig:
    neo4j: Neo4jConnectionConfig
    research_only: bool = True
    clinical_use_allowed: bool = False

    def validate(self) -> None:
        if self.research_only is not True or self.clinical_use_allowed is not False:
            raise PipelineError("GraphRAG v1 deve ser research_only=true e clinical_use_allowed=false.")
        if not self.neo4j.uri or not self.neo4j.user or not self.neo4j.password_env:
            raise PipelineError("Config Neo4j exige uri, user e password_env.")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"Config GraphRAG inválida ({path}): {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"Config GraphRAG deve ser objeto YAML: {path}")
    return value


def load_graphrag_config(path: Path) -> GraphRagConfig:
    data = _read_yaml(path)
    if data.get("schema") != CONFIG_SCHEMA:
        raise PipelineError(f"schema inválido em {path}: esperado {CONFIG_SCHEMA}.")
    neo4j = data.get("neo4j") or {}
    safety = data.get("safety") or {}
    if not isinstance(neo4j, dict) or not isinstance(safety, dict):
        raise PipelineError("Config GraphRAG exige blocos neo4j e safety.")
    config = GraphRagConfig(
        neo4j=Neo4jConnectionConfig(
            uri=str(neo4j.get("uri") or "").strip(),
            user=str(neo4j.get("user") or "").strip(),
            password_env=str(neo4j.get("password_env") or "").strip(),
            database=str(neo4j.get("database") or "neo4j").strip(),
        ),
        research_only=bool(safety.get("research_only", True)),
        clinical_use_allowed=bool(safety.get("clinical_use_allowed", False)),
    )
    config.validate()
    return config
