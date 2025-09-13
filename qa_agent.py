from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import pipeline
import os
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, CHUNKS_PKL_PATH

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_dim = 384

# --- Safely load FAISS index ---
if os.path.exists(FAISS_INDEX_PATH):
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"Loaded FAISS index from {FAISS_INDEX_PATH}")
    except Exception as e:
        print(f"Failed to read FAISS index. Creating new index. Error: {e}")
        index = faiss.IndexFlatL2(embedding_dim)
else:
    index = faiss.IndexFlatL2(embedding_dim)

# --- Safely load document chunks ---
if os.path.exists(CHUNKS_PKL_PATH):
    try:
        with open(CHUNKS_PKL_PATH, 'rb') as f:
            document_chunks = pickle.load(f)
        print(f"Loaded {len(document_chunks)} document chunks")
    except Exception as e:
        print(f"Failed to load chunks. Using empty list. Error: {e}")
        document_chunks = []
else:
    document_chunks = []

# Load HuggingFace QA pipeline
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')


def query_document(user_question, top_k=3):
    """
    Given a user question, finds top-k relevant chunks using FAISS
    and returns an answer from HuggingFace QA pipeline.
    """
    if len(document_chunks) == 0 or index.ntotal == 0:
        return "No document chunks found. Please process a document first."

    # Embed the question
    question_embedding = embedding_model.encode([user_question])
    question_embedding = np.array(question_embedding).astype('float32')

    # Search top-k relevant chunks
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [document_chunks[idx] for idx in indices[0] if idx < len(document_chunks)]

    if not relevant_chunks:
        return "No relevant chunks found."

    # Prepare context
    context = " ".join(relevant_chunks)

    # Get answer
    try:
        result = qa_pipeline(question=user_question, context=context)
        return result.get('answer', "No answer found.")
    except Exception as e:
        return f"Error in QA pipeline: {e}"
