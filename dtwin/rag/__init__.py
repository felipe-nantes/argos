"""Fundações locais do RAG do ARGOS.

O pacote mantém corpus, chunks, índices, retrieval e grounding auditáveis. A
integração com MedGemma é opcional e controlada por configuração versionada.
"""

from .chunking import CorpusChunk, CorpusDocument, chunk_document, validate_chunks
from .grounding import append_rag_to_prompt, build_rag_prompt_addendum
from .retriever import build_rag_context, persist_rag_context

__all__ = [
    "CorpusChunk",
    "CorpusDocument",
    "append_rag_to_prompt",
    "build_rag_context",
    "build_rag_prompt_addendum",
    "chunk_document",
    "persist_rag_context",
    "validate_chunks",
]
