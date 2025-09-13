from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, CHUNKS_PKL_PATH
import os

FAISS_INDEX_PATH = "faiss_index.index"  # path where your FAISS index will be saved

# Load model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Save index
faiss.write_index(index, FAISS_INDEX_PATH)

# Load embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize FAISS index (L2 distance)
embedding_dim = 384  # Depends on the model used
index = faiss.IndexFlatL2(embedding_dim)

# In-memory storage for document chunks (can persist with pickle)
document_chunks = []

def embed_and_store(chunks):
    global document_chunks
    
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    index.add(embeddings)
    document_chunks.extend(chunks)

    # Save index and chunks
    faiss.write_index(index, 'faiss_index.bin')
    with open('document_chunks.pkl', 'wb') as f:
        pickle.dump(document_chunks, f)
