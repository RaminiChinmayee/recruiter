import numpy as np
import streamlit as st

from sentence_transformers import CrossEncoder

from src.config import settings
from src.utils import logger


# --------------------------------------------------
# Cached CrossEncoder
# --------------------------------------------------

@st.cache_resource
def load_reranker_model():

    logger.info(
        "Loading CrossEncoder: %s",
        settings.RERANKER_MODEL
    )


    model = CrossEncoder(
        settings.RERANKER_MODEL
    )


    logger.info(
        "CrossEncoder loaded successfully"
    )


    return model


# --------------------------------------------------
# Resume Reranker
# --------------------------------------------------

class ResumeReranker:

    def __init__(self):

        self.model = (
            load_reranker_model()
        )


    # --------------------------------------------------
    # Normalize Score
    # --------------------------------------------------

    @staticmethod
    def normalize_score(score):

        return float(

            1 /

            (
                1 +

                np.exp(
                    -score
                )

            )

        )


    # --------------------------------------------------
    # Rerank
    # --------------------------------------------------

    def rerank(
        self,
        query,
        candidates,
        top_k=5
    ):

        if not candidates:

            return []


        logger.info(
            "Reranking %d candidates",
            len(candidates)
        )


        pairs = [

            (
                query,
                item["text"]
            )

            for item in candidates

        ]


        scores = self.model.predict(

            pairs,

            show_progress_bar=True
        )


        for item, score in zip(

            candidates,

            scores

        ):

            raw_score = float(
                score
            )


            normalized_score = (
                self.normalize_score(
                    raw_score
                )
            )


            item["rerank_score"] = (
                raw_score
            )


            item["score"] = (
                normalized_score
            )


        candidates.sort(

            key=lambda x: x["score"],

            reverse=True

        )


        logger.info(
            "Reranking completed"
        )


        return candidates[:top_k]