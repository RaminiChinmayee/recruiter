from src.embeddings import ResumeEmbedder


class ResumeRetriever:

    def __init__(self, resumes):
        """
        resumes : dictionary
        {
            filename : resume_text
        }
        """

        self.embedder = ResumeEmbedder()
        self.embedder.build_index(resumes)

    def retrieve(self, job_description, top_k=3):
        """
        Returns the top matching resumes.
        """

        return self.embedder.search(job_description, top_k)