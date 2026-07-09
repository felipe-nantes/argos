"""Construção de contexto metadata GraphRAG seguro."""
from __future__ import annotations

from typing import Any

from .schema import TARGET_FINDING


def build_metadata_graphrag_context(
    store,
    *,
    negative_subtype: str | None = None,
    phenotype_tag: str | None = None,
    target: str = TARGET_FINDING,
    limit: int = 10,
) -> dict[str, Any]:
    query = store.query_context(
        negative_subtype=negative_subtype,
        phenotype_tag=phenotype_tag,
        target=target,
        limit=limit,
    )
    return {
        "query_mode": "metadata_graphrag",
        "target_condition": "focal_liver_lesion_suspicion",
        "target": target,
        "filters": {
            "negative_subtype": negative_subtype,
            "phenotype_tag": phenotype_tag,
        },
        "retrieved_cases": query.get("retrieved_cases") or [],
        "mimic_context": query.get("mimic_context") or [],
        "limitations": query.get("limitations") or [],
        "research_only": True,
        "clinical_use_allowed": False,
    }
