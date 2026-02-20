import os
import pickle
import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from src.config import (
    BATCH_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    DATA_FOLDER,
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    CHUNKS_JSON_PATH,
    CHUNK_SIZE,
    OVERLAP,
)


class Chunking:
    def semantic_chunking(self, text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
        """
        Splits text into chunks.
        This uses a sliding window approach with sentences to respect semantic boundaries.
        """
        # Split into sentences (simple regex)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > chunk_size and current_chunk:
                # Join the current chunk and add to list
                chunks.append(" ".join(current_chunk))

                # Create overlap
                # Keep the last few sentences that fit within overlap size
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) < overlap:
                        overlap_chunk.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(sentence)
            current_length += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def generate_embedding(self, chunks, batch_size=BATCH_SIZE):

      
    #     all_embeddings = []

    #     # Process the list in chunks of batch_size
    #     for i in range(0, len(chunks), batch_size):
    #         batch = chunks[i : i + batch_size]

    #         config = types.EmbedContentConfig(
    #             output_dimensionality=EMBEDDING_DIMENSION,
    #             task_type="RETRIEVAL_DOCUMENT",  # Recommended for RAG storage
    #         )

    #         result = self.client.models.embed_content(
    #             model=EMBEDDING_MODEL, contents=batch, config=config
    #         )

    #         # Extract numerical values from the response
    #         for embedding in result.embeddings:
    #             all_embeddings.append(embedding.values)

    #     return np.array(all_embeddings, dtype=np.float32)
        """
        Generates embeddings for a list of text chunks using Sentence Transformers.
        """
        embeddings = self.model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)


class FaissStore:
    def save_index(self, embeddings, chunks):
        os.makedirs(DATA_FOLDER, exist_ok=True)

        # Normalize embeddings for Inner Product (effectively Cosine Similarity)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, FAISS_INDEX_PATH)

        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)

        # Also save as JSON for transparency/debugging
        with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)

    def load_index(self):
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
            return None, None

        index = faiss.read_index(FAISS_INDEX_PATH)

        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)

        return index, chunks
