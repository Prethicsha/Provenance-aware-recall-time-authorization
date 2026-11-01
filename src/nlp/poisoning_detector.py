"""
Semantic Poisoning Detection Baselines
Implements:
1. TF-IDF + Logistic Regression (Baseline 1)
2. Sentence-Embedding + Logistic Regression (Baseline 2)
Computes genuine NLP classification metrics (Precision, Recall, F1, AUROC, FPR, FNR).
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from src.nlp.embeddings import get_embedding_model


class PoisoningDetector:
    """Base interface for NLP memory poisoning detectors."""
    def fit(self, texts: List[str], labels: List[int]) -> "PoisoningDetector":
        raise NotImplementedError

    def predict(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError

    def load(self, filepath: str):
        raise NotImplementedError


class TfidfPoisoningDetector(PoisoningDetector):
    """TF-IDF N-gram feature extraction + Logistic Regression Classifier."""
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            stop_words="english"
        )
        self.clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> "TfidfPoisoningDetector":
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted.")
        X = self.vectorizer.transform(texts)
        return self.clf.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted.")
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)[:, 1]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "clf": self.clf}, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.clf = data["clf"]
            self.is_fitted = True
        return self


class EmbeddingPoisoningDetector(PoisoningDetector):
    """Dense Sentence-Transformer Embeddings + Logistic Regression Classifier."""
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> "EmbeddingPoisoningDetector":
        embedder = get_embedding_model(self.embedding_model_name)
        X = embedder.encode(texts, normalize=True)
        self.clf.fit(X, labels)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted.")
        embedder = get_embedding_model(self.embedding_model_name)
        X = embedder.encode(texts, normalize=True)
        return self.clf.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted.")
        embedder = get_embedding_model(self.embedding_model_name)
        X = embedder.encode(texts, normalize=True)
        return self.clf.predict_proba(X)[:, 1]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"clf": self.clf, "embedding_model_name": self.embedding_model_name}, f)

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.clf = data["clf"]
            self.embedding_model_name = data.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
            self.is_fitted = True
        return self


def compute_classification_metrics(y_true: List[int], y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Computes exact, un-fabricated classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.5

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": len(y_true)
    }


def train_and_evaluate_detectors(
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    save_dir: str = "data/processed/models"
) -> Dict[str, Any]:
    """Trains both TF-IDF and Embedding baselines and returns evaluation metrics."""
    results = {}

    # 1. TF-IDF Baseline
    tfidf_detector = TfidfPoisoningDetector()
    tfidf_detector.fit(train_texts, train_labels)
    tfidf_preds = tfidf_detector.predict(test_texts)
    tfidf_probs = tfidf_detector.predict_proba(test_texts)
    tfidf_metrics = compute_classification_metrics(test_labels, tfidf_preds, tfidf_probs)
    tfidf_path = os.path.join(save_dir, "tfidf_detector.pkl")
    tfidf_detector.save(tfidf_path)
    results["tfidf"] = {
        "metrics": tfidf_metrics,
        "model_path": tfidf_path,
        "detector": tfidf_detector
    }

    # 2. Sentence-Embedding Baseline
    emb_detector = EmbeddingPoisoningDetector()
    emb_detector.fit(train_texts, train_labels)
    emb_preds = emb_detector.predict(test_texts)
    emb_probs = emb_detector.predict_proba(test_texts)
    emb_metrics = compute_classification_metrics(test_labels, emb_preds, emb_probs)
    emb_path = os.path.join(save_dir, "embedding_detector.pkl")
    emb_detector.save(emb_path)
    results["embedding"] = {
        "metrics": emb_metrics,
        "model_path": emb_path,
        "detector": emb_detector
    }

    return results
