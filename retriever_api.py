import os
import sys
import json
import logging
import faiss
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HippoRAG-style Retriever for VERL")

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = 3
    return_scores: bool = False

SAVE_DIR = os.getenv("SAVE_DIR", "verl/hipporag/outputs/server")
DATA_PATH = os.path.join(SAVE_DIR, "openie_results_ner_qwen7B_4096:latest.json")

class SimpleHippoRAGRetriever:
    def __init__(self, data_path: str, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        self.data_path = data_path
        self.embedding_model = embedding_model
        self.docs = []
        self.doc_embeddings = None
        self.embedding_dim = 1024

        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model, device="cuda")

        self._load_data()
        self._build_index()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.docs = [e["passage"] for e in data.get("docs", []) if "passage" in e]
        logger.info(f"Loaded {len(self.docs)} documents")

    def _build_index(self):
        import faiss
        logger.info("Building FAISS index...")
        self.doc_embeddings = self.model.encode(self.docs, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

        self.embedding_dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings)
        logger.info(f"Index built with {self.index.ntotal} vectors")

    def retrieve(self, queries: List[str], topk: int = 3):
        query_embeddings = self.model.encode(queries, batch_size=32, convert_to_numpy=True)
        faiss.normalize_L2(query_embeddings)

        scores, indices = self.index.search(query_embeddings, k=topk)

        results = []
        for i, (score_list, idx_list) in enumerate(zip(scores, indices)):
            query_results = []
            for score, idx in zip(score_list, idx_list):
                if idx >= 0 and idx < len(self.docs):
                    query_results.append({
                        "document": self.docs[idx],
                        "score": float(score)
                    })
            results.append(query_results)

        return results

global_retriever = None

@app.on_event("startup")
async def load_index():
    global global_retriever
    try:
        logger.info("Initializing HippoRAG-style Retriever...")
        global_retriever = SimpleHippoRAGRetriever(
            data_path=DATA_PATH,
            embedding_model="BAAI/bge-large-en-v1.5"
        )
        logger.info("Retriever ready")
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        raise

@app.post("/retrieve")
async def retrieve(request: QueryRequest):
    if global_retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    try:
        results = global_retriever.retrieve(
            queries=request.queries,
            topk=request.topk
        )

        if request.return_scores:
            return {"result": results}
        else:
            return {"result": [[doc["document"] for doc in r] for r in results]}
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "service": "HippoRAG-style Retriever"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)