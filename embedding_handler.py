from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, CHUNKS_PKL_PATH

# Load the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Dimension of the embedding vectors (must match your model)
embedding_dim = 384


def get_faiss_index():
    """
    Load existing FAISS index if it exists, otherwise create a new one.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"Loaded FAISS index from {FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"Failed to read FAISS index, creating a new one. Error: {e}")
            index = faiss.IndexFlatL2(embedding_dim)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
    return index


def get_document_chunks():
    """
    Load existing document chunks from pickle file, or return empty list.
    """
    if os.path.exists(CHUNKS_PKL_PATH):
        try:
            with open(CHUNKS_PKL_PATH, 'rb') as f:
                chunks = pickle.load(f)
            print(f"Loaded {len(chunks)} chunks from {CHUNKS_PKL_PATH}")
            return chunks
        except Exception as e:
            print(f"Failed to load chunks, returning empty list. Error: {e}")
            return []
    else:
        return []


def embed_and_store(chunks):
    """
    Embed new document chunks, add them to FAISS index and pickle file.
    """
    if not chunks:
        return

    # Always load latest index and chunks
    index = get_faiss_index()
    document_chunks = get_document_chunks()

    # Embed new chunks
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Add to FAISS index and update chunks
    index.add(embeddings)
    document_chunks.extend(chunks)

    # Save updated index and chunks
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PKL_PATH, 'wb') as f:
        pickle.dump(document_chunks, f)

    print(f"Stored {len(chunks)} new chunks. Total chunks: {len(document_chunks)}")
