import os
from dotenv import load_dotenv


# Load variables from .env
load_dotenv()


class Settings:

    # --------------------------------------------------
    # Groq
    # --------------------------------------------------

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    LLM_MODEL = "llama-3.1-8b-instant"


    # --------------------------------------------------
    # Embedding Model
    # --------------------------------------------------

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"


    # --------------------------------------------------
    # Cross Encoder
    # --------------------------------------------------

    RERANKER_MODEL = (
        "cross-encoder/"
        "ms-marco-MiniLM-L-6-v2"
    )


    # --------------------------------------------------
    # FAISS
    # --------------------------------------------------

    FAISS_DIR = "data/faiss_index"

    FAISS_PATH = os.path.join(
        FAISS_DIR,
        "resume.index"
    )

    METADATA_PATH = os.path.join(
        FAISS_DIR,
        "metadata.json"
    )


    # --------------------------------------------------
    # Resume Storage
    # --------------------------------------------------

    RESUME_FOLDER = "data/resumes"


    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    RETRIEVAL_TOP_K = 20

    RERANK_TOP_K = 10

    FINAL_TOP_K = 5


settings = Settings()