import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embedding Config
EMBEDDING_MODEL = "all-mpnet-base-v2"
BATCH_SIZE = 16
EMBEDDING_DIMENSION = 768
CHUNK_SIZE = 300
OVERLAP = 50

# FAISS and Data Paths
FILE_PATH = "data"
DATA_FOLDER = "data/vector_db"
FAISS_INDEX_PATH = os.path.join(DATA_FOLDER, "index.faiss")
CHUNKS_PATH = os.path.join(DATA_FOLDER, "chunks.pkl")
CHUNKS_JSON_PATH = os.path.join(DATA_FOLDER, "chunks.json")

# GenerationClient
MODEL = "llama-3.3-70b-versatile"