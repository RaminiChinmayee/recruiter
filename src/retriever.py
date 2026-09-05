from rank_bm25 import BM25Okapi

from src.embeddings import ResumeEmbedder
from src.utils import logger


class HybridRetriever:

    def __init__(self, resumes):

        if not resumes:

            raise ValueError(
                "No resumes provided."
            )


        self.resumes = resumes


        # --------------------------------------------------
        # Semantic Search
        # --------------------------------------------------

        logger.info(
            "Initializing semantic retriever"
        )


        self.embedder = ResumeEmbedder()


        self.embedder.build_index(
            resumes
        )


        # --------------------------------------------------
        # BM25 Keyword Search
        # --------------------------------------------------

        logger.info(
            "Building BM25 index"
        )


        corpus = [

            text.lower().split()

            for text in resumes.values()

        ]


        self.bm25 = BM25Okapi(
            corpus
        )


        self.names = list(
            resumes.keys()
        )


        logger.info(
            "Hybrid retriever initialized"
        )


    # --------------------------------------------------
    # Hybrid Search
    # --------------------------------------------------

    def search(
        self,
        query,
        top_k=20
    ):

        if not self.resumes:

            return []


        logger.info(
            "Starting hybrid search"
        )


        # --------------------------------------------------
        # Semantic Search
        # --------------------------------------------------

        semantic_results = (
            self.embedder.search(
                query,
                top_k=top_k
            )
        )


        semantic_scores = {

            item["name"]:
                item["similarity"]

            for item in semantic_results

        }


        # --------------------------------------------------
        # BM25 Search
        # --------------------------------------------------

        query_tokens = (
            query.lower().split()
        )


        bm25_scores = (
            self.bm25.get_scores(
                query_tokens
            )
        )


        max_bm25 = max(
            bm25_scores
        ) if len(bm25_scores) else 0


        # --------------------------------------------------
        # Combine Scores
        # --------------------------------------------------

        results = []


        for i, (name, text) in enumerate(
            self.resumes.items()
        ):

            semantic_score = (
                semantic_scores.get(
                    name,
                    0.0
                )
            )


            keyword_score = (
                float(
                    bm25_scores[i]
                )
                if i < len(bm25_scores)
                else 0.0
            )


            normalized_keyword = (

                keyword_score / max_bm25

                if max_bm25 > 0

                else 0.0

            )


            hybrid_score = (

                0.7 * semantic_score

                +

                0.3 * normalized_keyword

            )


            results.append({

                "name": name,

                "text": text,

                "similarity":
                    semantic_score,

                "keyword_score":
                    keyword_score,

                "score":
                    hybrid_score

            })


        results.sort(

            key=lambda x: x["score"],

            reverse=True

        )


        logger.info(
            "Hybrid search completed"
        )


        return results[:top_k]