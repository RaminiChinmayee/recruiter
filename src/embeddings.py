import faiss
import numpy as np
import os

from sentence_transformers import SentenceTransformer

from src.config import settings
from src.utils import save_json,load_json



class ResumeEmbedder:


    def __init__(self):

        self.model=SentenceTransformer(
            settings.EMBEDDING_MODEL
        )

        self.index=None

        self.metadata=[]



    def build_index(self,resumes):


        texts=list(resumes.values())


        embeddings=self.model.encode(

            texts,

            normalize_embeddings=True

        ).astype(
            "float32"
        )


        dim=embeddings.shape[1]


        self.index=faiss.IndexFlatIP(dim)


        self.index.add(
            embeddings
        )


        self.metadata=[

        {
        "name":name,
        "text":text
        }

        for name,text in resumes.items()

        ]



        os.makedirs(
            "data/faiss_index",
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



    def search(
        self,
        query,
        top_k=5
    ):


        query_embedding=self.model.encode(

            [query],

            normalize_embeddings=True

        ).astype(
            "float32"
        )


        scores,ids=self.index.search(

            query_embedding,

            top_k

        )


        results=[]


        for score,idx in zip(
            scores[0],
            ids[0]
        ):


            results.append({

            **self.metadata[idx],

            "similarity":
            float(score)

            })


        return results