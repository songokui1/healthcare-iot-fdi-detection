"""A small curated, local reference corpus for retrieval-augmented analysis.

These notes are general explanatory context, NOT examples with test labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCUMENTS = {
    "K01": "False data injection modifies observed data in transit or at the sensor. "
           "Suspicious network features do not establish an attack without corroboration.",
    "K02": "Man-in-the-middle interception can affect confidentiality and integrity. "
           "Authentication and network provenance help evaluate the event.",
    "K03": "Packet timing, port and jitter shifts can arise from benign network "
           "congestion, reconfiguration, or device maintenance; avoid overclaiming.",
    "K04": "Clinical biometric measurements vary naturally. Atypical readings "
           "do not prove malicious manipulation or indicate a medical diagnosis.",
    "K05": "A false negative is a missed attack. A false positive is a benign "
           "event flagged as suspicious. Both must be measured on unseen data.",
    "K06": "The WUSTL-EHMS-2020 benchmark combines network-flow and biometric "
           "features. Its labels were generated using attacker MAC addresses. "
           "Exclude identifiers, label fields and obvious leakage proxies.",
}

@dataclass
class Retriever:
    corpus: dict[str, str]
    vectorizer: TfidfVectorizer
    matrix: object

    @classmethod
    def create(cls, corpus: dict[str, str] | None = None) -> "Retriever":
        docs = corpus if corpus is not None else DOCUMENTS
        if not docs:
            raise ValueError("knowledge corpus cannot be empty")
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(docs.values())
        return cls(dict(docs), vectorizer, matrix)

    def search(self, query: str, k: int = 3) -> list[dict]:
        if k < 1:
            raise ValueError("k must be positive")
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        keys = list(self.corpus)
        return [{"id": keys[i], "text": self.corpus[keys[i]],
                 "score": round(float(scores[i]), 4)}
                for i in ranked[:k]]
