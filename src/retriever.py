from src.embeddings import ResumeEmbedder


class ResumeRetriever:

    def __init__(self, resumes):
        self.embedder = ResumeEmbedder()
        self.embedder.build_index(resumes)

    def retrieve(self, job_description, top_k=5):
        return self.embedder.search(
            job_description,
            top_k
        )