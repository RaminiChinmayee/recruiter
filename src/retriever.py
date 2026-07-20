from rank_bm25 import BM25Okapi

from src.embeddings import ResumeEmbedder



class HybridRetriever:


    def __init__(self,resumes):


        self.resumes=resumes


        # Semantic search

        self.embedder=ResumeEmbedder()

        self.embedder.build_index(
            resumes
        )


        # Keyword search

        corpus=[

            text.lower().split()

            for text in resumes.values()

        ]


        self.bm25=BM25Okapi(
            corpus
        )


        self.names=list(
            resumes.keys()
        )



    def search(
        self,
        query,
        top_k=20
    ):


        # FAISS

        semantic_results=self.embedder.search(
            query,
            top_k
        )



        semantic_scores={}


        for item in semantic_results:

            semantic_scores[
                item["name"]
            ] = item["similarity"]




        # BM25


        bm25_scores=self.bm25.get_scores(

            query.lower().split()

        )



        keyword_scores={}


        for i,score in enumerate(bm25_scores):

            keyword_scores[
                self.names[i]
            ]=score




        # Combine


        results=[]


        for name,text in self.resumes.items():


            semantic=semantic_scores.get(
                name,
                0
            )


            keyword=keyword_scores.get(
                name,
                0
            )


            hybrid=(

                0.7*semantic

                +

                0.3*(keyword/
                     max(keyword_scores.values()
                         or [1]))

            )


            results.append({

                "name":name,

                "text":text,

                "score":hybrid

            })



        results.sort(

            key=lambda x:x["score"],

            reverse=True

        )


        return results[:top_k]