from sentence_transformers import CrossEncoder
import numpy as np

from src.config import settings



class ResumeReranker:


    def __init__(self):

        self.model = CrossEncoder(
            settings.RERANKER_MODEL
        )


    # --------------------------------------------------
    # Convert raw CrossEncoder score to 0-1 probability
    # --------------------------------------------------

    @staticmethod
    def normalize_score(score):

        return float(
            1 / (1 + np.exp(-score))
        )


    # --------------------------------------------------
    # Rerank Candidates
    # --------------------------------------------------

    def rerank(
        self,
        query,
        candidates,
        top_k=5
    ):


        if not candidates:

            return []



        pairs = []


        for item in candidates:

            pairs.append(

                (
                    query,
                    item["text"]
                )

            )



        # Raw CrossEncoder scores

        scores = self.model.predict(
            pairs
        )



        for item, score in zip(
            candidates,
            scores
        ):


            raw_score = float(score)


            normalized_score = self.normalize_score(
                raw_score
            )


            item["rerank_score"] = raw_score


            item["score"] = normalized_score



        # Sort using normalized relevance

        candidates.sort(

            key=lambda x:x["score"],

            reverse=True

        )


        return candidates[:top_k]