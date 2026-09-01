"""LlamaIndex adapters for ReDevOps (frameworks plan Section 7).

Wraps a REAL LlamaIndex `VectorStoreIndex` retriever as a ReDevOps Context-Runtime *representation*. LlamaIndex
keeps owning indexing, embeddings, retrieval and its query engine / Workflow control flow; ReDevOps registers the
LlamaIndex retriever as ONE representation the optimizer may compose with structural closure, and adds authority,
replay, verification and governance around the answer. The retriever is not replaced — it is composed.

Embeddings run locally and offline (the shipped redevops_rag bge-small encoder, wrapped as a LlamaIndex
BaseEmbedding), and the LLM is disabled for retrieval so these runs need no external service.
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

from adapter import NodeSpan  # noqa: E402

from llama_index.core import Document, Settings, VectorStoreIndex  # noqa: E402
from llama_index.core.embeddings import BaseEmbedding  # noqa: E402

try:
    import llama_index.core as _lic
    LI_VERSION = getattr(_lic, "__version__", "0.14.24")
except Exception:
    LI_VERSION = "0.14.24"


_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        from redevops_rag.embed import Embedder
        _ENC = Embedder()
    return _ENC


class LocalEmbedding(BaseEmbedding):
    """A real LlamaIndex embedding backed by the shipped local bge-small encoder — offline, CPU, deterministic."""

    def _get_text_embedding(self, text: str):
        return [float(x) for x in _encoder().encode([text])[0]]

    def _get_query_embedding(self, query: str):
        return [float(x) for x in _encoder().encode([query])[0]]

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)


def _configure():
    Settings.embed_model = LocalEmbedding()
    Settings.llm = None            # retrieval only — no external LLM


class LlamaIndexRetrieverCapability:
    """WorkflowAdapter over a real LlamaIndex VectorStoreIndex retriever. Given a corpus {id -> text}, it builds
    the index once and answers `retrieve(query, k)` with ranked (id, score) hits — the ReDevOps optimizer treats
    these hits as one representation among {llama-index, structural, composed}."""

    framework = "llama-index"
    framework_version = LI_VERSION

    def __init__(self, corpus: dict[str, str], *, name: str = "li_retriever"):
        _configure()
        self._name = name
        self._ids = list(corpus)
        docs = [Document(text=corpus[i], id_=i, doc_id=i) for i in self._ids]
        self._index = VectorStoreIndex.from_documents(docs, show_progress=False)
        self.model_id = "bge-small-en (local)"
        self._last_span: NodeSpan | None = None

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        r = self._index.as_retriever(similarity_top_k=top_k)
        hits = r.retrieve(query)
        self._last_span = NodeSpan(
            node_id=self.node_identity("retrieve", {"q": query}), kind="retriever",
            name=f"llama-index:{self._name}", attrs={"framework": self.framework, "top_k": top_k},
            children=[NodeSpan(node_id="embed", kind="model", name=self.model_id)])
        out = []
        for h in hits:
            sid = h.node.ref_doc_id or h.node.id_
            out.append((sid, float(h.score if h.score is not None else 0.0)))
        return out

    def invoke(self, capability: str, inputs: dict) -> dict:
        hits = self.retrieve(inputs.get("query") or inputs.get("prompt") or "", inputs.get("top_k", 10))
        return {"hits": hits, "framework": self.framework, "model": self.model_id}

    def node_identity(self, capability: str, inputs: dict) -> str:
        h = hashlib.sha256((capability + str(sorted(inputs.items()))).encode()).hexdigest()[:10]
        return f"llama-index:{self._name}:{h}"

    def state_projection(self) -> dict:
        return {"framework": self.framework, "retriever": self._name, "corpus": len(self._ids),
                "model": self.model_id}

    def telemetry(self) -> NodeSpan:
        return self._last_span or NodeSpan(node_id=self._name, kind="retriever", name=f"llama-index:{self._name}")
