from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, CHUNKS_PKL_PATH

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_dim = 384

def get_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
        except Exception:
            index = faiss.IndexFlatL2(embedding_dim)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
    return index

def get_document_chunks():
    if os.path.exists(CHUNKS_PKL_PATH):
        try:
            with open(CHUNKS_PKL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return []
    return []

def embed_and_store(chunks):
    if not chunks:
        return
    index = get_faiss_index()
    document_chunks = get_document_chunks()
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)
    document_chunks.extend(chunks)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_PKL_PATH, 'wb') as f:
        pickle.dump(document_chunks, f)
