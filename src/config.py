import os
from dotenv import load_dotenv


load_dotenv()


class Settings:

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )

    LLM_MODEL = (
        "llama-3.3-70b-versatile"
    )

    EMBEDDING_MODEL = (
        "all-MiniLM-L6-v2"
    )

    RERANKER_MODEL = (
        "cross-encoder/"
        "ms-marco-MiniLM-L-6-v2"
    )


    FAISS_PATH = (
        "data/faiss_index/"
        "resume.index"
    )

    METADATA_PATH = (
        "data/faiss_index/"
        "metadata.json"
    )


    RESUME_FOLDER = (
        "data/resumes"
    )



settings = Settings()