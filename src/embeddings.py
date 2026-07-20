import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class ResumeEmbedder:
    """
    Creates resume embeddings using SentenceTransformer
    and stores them in a FAISS index.
    """

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index = None

        self.resume_names = []

        self.resume_texts = []

    # -------------------------------------------------------
    # Build FAISS Index
    # -------------------------------------------------------

    def build_index(self, resumes):
        """
        Parameters
        ----------
        resumes : dict

        {
            filename : resume_text
        }
        """

        if len(resumes) == 0:
            raise ValueError("No resumes found.")

        self.resume_names = list(resumes.keys())

        self.resume_texts = list(resumes.values())

        embeddings = self.model.encode(
            self.resume_texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        print(f"Indexed {len(self.resume_names)} resumes.")

    # -------------------------------------------------------
    # Search Similar Resumes
    # -------------------------------------------------------

    def search(self, query, top_k=5):

        if self.index is None:
            raise ValueError("Index has not been built.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        top_k = min(top_k, len(self.resume_names))

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            results.append(
                {
                    "name": self.resume_names[idx],
                    "text": self.resume_texts[idx],
                    "score": round(float(score), 4)
                }
            )

        return results

    # -------------------------------------------------------
    # Number of indexed resumes
    # -------------------------------------------------------

    def __len__(self):

        return len(self.resume_names)