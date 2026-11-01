"""
Local Sentence Embeddings Module using SentenceTransformers
Model: sentence-transformers/all-MiniLM-L6-v2 (Local, CPU-optimized, dimension=384)
"""

from typing import List, Union
import numpy as np


class EmbeddingModel:
    """
    Singleton-style wrapper for local sentence embedding inference.
    """
    _instance = None

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Encodes text(s) into 384-dimensional dense vectors."""
        model = self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        embeddings = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
        return embeddings


_GLOBAL_EMBEDDING_MODEL = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingModel:
    global _GLOBAL_EMBEDDING_MODEL
    if _GLOBAL_EMBEDDING_MODEL is None:
        _GLOBAL_EMBEDDING_MODEL = EmbeddingModel(model_name=model_name)
    return _GLOBAL_EMBEDDING_MODEL
