"""Fundações locais do RAG do ARGOS.

Este pacote é deliberadamente isolado do fluxo MedGemma. A F0 do RAG prepara
corpus, chunks e índices auditáveis antes de qualquer integração com inferência.
"""

from .chunking import CorpusChunk, CorpusDocument, chunk_document, validate_chunks

__all__ = ["CorpusChunk", "CorpusDocument", "chunk_document", "validate_chunks"]
