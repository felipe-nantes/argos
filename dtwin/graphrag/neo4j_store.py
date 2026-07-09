"""Camada fina de acesso ao Neo4j para o metadata GraphRAG."""
from __future__ import annotations

from typing import Any

from dtwin.core import PipelineError

from .config import GraphRagConfig
from .schema import CONSTRAINTS, MIMIC_RELATIONS, NEGATIVE_TARGET_RELATIONS, registry_record_to_graph_params


UPSERT_CASE_CYPHER = """
MERGE (d:Dataset {dataset_id: $dataset_id})
SET d.name = $dataset_name
MERGE (c:Case {case_id: $case_id})
SET c.series_id = $series_id,
    c.review_status = $review_status,
    c.has_segmentation = $has_segmentation,
    c.research_only = true,
    c.clinical_use_allowed = false,
    c.metadata = $metadata
MERGE (c)-[:FROM_DATASET]->(d)
MERGE (rc:RagClass {name: $rag_class})
MERGE (c)-[:HAS_RAG_CLASS]->(rc)
MERGE (gt:GroundTruthLabel {name: $label})
MERGE (c)-[:HAS_LABEL]->(gt)
MERGE (m:Modality {name: $modality})
MERGE (c)-[:HAS_MODALITY]->(m)
MERGE (sf:SourceFormat {name: $source_format})
MERGE (c)-[:HAS_SOURCE_FORMAT]->(sf)
MERGE (p:Phase {name: $phase})
MERGE (c)-[:HAS_PHASE]->(p)
WITH c
FOREACH (value IN CASE WHEN $negative_subtype IS NULL THEN [] ELSE [$negative_subtype] END |
  MERGE (ns:NegativeSubtype {name: value})
  MERGE (c)-[:HAS_NEGATIVE_SUBTYPE]->(ns)
)
FOREACH (value IN CASE WHEN $positive_subtype IS NULL THEN [] ELSE [$positive_subtype] END |
  MERGE (ps:PositiveSubtype {name: value})
  MERGE (c)-[:HAS_POSITIVE_SUBTYPE]->(ps)
)
FOREACH (tag IN $phenotype_tags |
  MERGE (pt:PhenotypeTag {name: tag})
  MERGE (c)-[:HAS_PHENOTYPE_TAG]->(pt)
)
FOREACH (limitation IN $limitations |
  MERGE (l:Limitation {text: limitation})
  MERGE (c)-[:HAS_LIMITATION]->(l)
)
FOREACH (warning IN $warnings |
  MERGE (w:Limitation {text: warning})
  MERGE (c)-[:HAS_LIMITATION]->(w)
)
"""


class Neo4jStore:
    def __init__(self, config: GraphRagConfig):
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise PipelineError("Pacote neo4j não instalado. Instale o extra GraphRAG antes de usar Neo4j real.") from exc
        self._database = config.neo4j.database
        self._driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.user, config.neo4j.password),
        )

    def close(self) -> None:
        self._driver.close()

    def run_write(self, cypher: str, parameters: dict[str, Any] | None = None) -> None:
        with self._driver.session(database=self._database) as session:
            session.execute_write(lambda tx: tx.run(cypher, **(parameters or {})).consume())

    def run_read(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:
            result = session.execute_read(lambda tx: list(tx.run(cypher, **(parameters or {}))))
        return [dict(record) for record in result]

    def ensure_schema(self) -> None:
        for statement in CONSTRAINTS:
            self.run_write(statement)
        for phenotype_tag, finding in MIMIC_RELATIONS:
            self.run_write(
                """
                MERGE (pt:PhenotypeTag {name: $phenotype_tag})
                MERGE (f:Finding {name: $finding})
                MERGE (pt)-[:CAN_MIMIC]->(f)
                """,
                {"phenotype_tag": phenotype_tag, "finding": finding},
            )
        for negative_subtype, finding in NEGATIVE_TARGET_RELATIONS:
            self.run_write(
                """
                MERGE (ns:NegativeSubtype {name: $negative_subtype})
                MERGE (f:Finding {name: $finding})
                MERGE (ns)-[:NEGATIVE_FOR_TARGET]->(f)
                """,
                {"negative_subtype": negative_subtype, "finding": finding},
            )

    def upsert_registry_record(self, record: dict[str, Any]) -> None:
        self.run_write(UPSERT_CASE_CYPHER, registry_record_to_graph_params(record))

    def query_context(
        self,
        *,
        negative_subtype: str | None,
        phenotype_tag: str | None,
        target: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        rows = self.run_read(
            """
            MATCH (c:Case)
            OPTIONAL MATCH (c)-[:FROM_DATASET]->(d:Dataset)
            OPTIONAL MATCH (c)-[:HAS_NEGATIVE_SUBTYPE]->(ns:NegativeSubtype)
            OPTIONAL MATCH (c)-[:HAS_PHENOTYPE_TAG]->(pt:PhenotypeTag)
            WHERE ($negative_subtype IS NULL OR ns.name = $negative_subtype)
              AND ($phenotype_tag IS NULL OR pt.name = $phenotype_tag)
            RETURN c.case_id AS case_id,
                   d.dataset_id AS dataset_id,
                   ns.name AS negative_subtype,
                   collect(DISTINCT pt.name) AS phenotype_tags,
                   c.review_status AS review_status
            ORDER BY case_id
            LIMIT $limit
            """,
            {
                "negative_subtype": negative_subtype,
                "phenotype_tag": phenotype_tag,
                "target": target,
                "limit": int(limit),
            },
        )
        mimics = self.run_read(
            """
            MATCH (pt:PhenotypeTag)-[:CAN_MIMIC]->(f:Finding)
            WHERE ($phenotype_tag IS NULL OR pt.name = $phenotype_tag)
              AND ($target IS NULL OR f.name = $target)
            RETURN pt.name AS phenotype_tag, f.name AS finding
            ORDER BY phenotype_tag, finding
            """,
            {"phenotype_tag": phenotype_tag, "target": target},
        )
        limitations = self.run_read(
            """
            MATCH (c:Case)-[:HAS_LIMITATION]->(l:Limitation)
            OPTIONAL MATCH (c)-[:HAS_NEGATIVE_SUBTYPE]->(ns:NegativeSubtype)
            WHERE ($negative_subtype IS NULL OR ns.name = $negative_subtype)
            RETURN DISTINCT l.text AS limitation
            ORDER BY limitation
            LIMIT 20
            """,
            {"negative_subtype": negative_subtype},
        )
        return {
            "retrieved_cases": rows,
            "mimic_context": mimics,
            "limitations": [row.get("limitation") for row in limitations if row.get("limitation")],
        }
