import os

import faiss
import streamlit as st

from sentence_transformers import SentenceTransformer

from src.config import settings
from src.utils import save_json, logger


# --------------------------------------------------
# Cached Embedding Model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    logger.info(
        "Loading embedding model: %s",
        settings.EMBEDDING_MODEL
    )


    model = SentenceTransformer(
        settings.EMBEDDING_MODEL
    )


    logger.info(
        "Embedding model loaded successfully"
    )


    return model


# --------------------------------------------------
# Resume Embedder
# --------------------------------------------------

class ResumeEmbedder:

    def __init__(self):

        self.model = (
            load_embedding_model()
        )

        self.index = None

        self.metadata = []


    # --------------------------------------------------
    # Build FAISS Index
    # --------------------------------------------------

    def build_index(self, resumes):

        if not resumes:

            raise ValueError(
                "No resumes available "
                "to build the index."
            )


        texts = list(
            resumes.values()
        )


        logger.info(
            "Generating embeddings for %d resumes",
            len(texts)
        )


        embeddings = self.model.encode(

            texts,

            normalize_embeddings=True,

            show_progress_bar=True
        )


        embeddings = embeddings.astype(
            "float32"
        )


        dimension = embeddings.shape[1]


        logger.info(
            "Embedding dimension: %d",
            dimension
        )


        self.index = faiss.IndexFlatIP(
            dimension
        )


        self.index.add(
            embeddings
        )


        self.metadata = [

            {
                "name": name,
                "text": text
            }

            for name, text in resumes.items()

        ]


        os.makedirs(
            settings.FAISS_DIR,
            exist_ok=True
        )


        faiss.write_index(

            self.index,

            settings.FAISS_PATH
        )


        save_json(

            self.metadata,

            settings.METADATA_PATH
        )


        logger.info(
            "FAISS index built with %d resumes",
            len(self.metadata)
        )


    # --------------------------------------------------
    # Semantic Search
    # --------------------------------------------------

    def search(
        self,
        query,
        top_k=5
    ):

        if self.index is None:

            raise ValueError(
                "FAISS index has not been built."
            )


        if self.index.ntotal == 0:

            return []


        k = min(
            top_k,
            self.index.ntotal
        )


        logger.info(
            "Generating query embedding"
        )


        query_embedding = self.model.encode(

            [query],

            normalize_embeddings=True
        )


        query_embedding = (
            query_embedding.astype(
                "float32"
            )
        )


        scores, ids = self.index.search(

            query_embedding,

            k
        )


        results = []


        for score, idx in zip(

            scores[0],

            ids[0]

        ):

            if idx < 0:
                continue


            results.append({

                **self.metadata[idx],

                "similarity":
                    float(score)

            })


        return results