from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class ResumeEmbedder:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.names = []
        self.texts = []

    def build_index(self, resumes):
        """
        resumes : dictionary
        {
            filename : resume_text
        }
        """

        self.names = list(resumes.keys())
        self.texts = list(resumes.values())

        embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True
        ).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        print(f"Indexed {len(self.texts)} resumes.")

    def search(self, query, top_k=3):

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(query_embedding)

        top_k = min(top_k, len(self.names))

        scores, indices = self.index.search(query_embedding, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "name": self.names[idx],
                "text": self.texts[idx],
                "score": float(score)
            })

        return results