
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, CHUNKS_PKL_PATH
import os

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

embedding_dim = 384

## Removed erroneous top-level call to faiss.write_index(index, FAISS_INDEX_PATH)

def get_faiss_index():
    """
    Load existing FAISS index or create a new one if not found.
    """
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    else:
        return faiss.IndexFlatL2(embedding_dim)

def get_document_chunks():
    """
    Load existing document chunks or return empty list if not found.
    """
    if os.path.exists(CHUNKS_PKL_PATH):
        with open(CHUNKS_PKL_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        return []


def embed_and_store(chunks):
    """
    Embed chunks and store them in FAISS index and pickle file.
    Handles concurrent Streamlit sessions by always loading latest index/chunks.
    """
    index = get_faiss_index()
    document_chunks = get_document_chunks()

    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    index.add(embeddings)
    document_chunks.extend(chunks)

    # Save index and chunks using config paths
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PKL_PATH, 'wb') as f:
        pickle.dump(document_chunks, f)
