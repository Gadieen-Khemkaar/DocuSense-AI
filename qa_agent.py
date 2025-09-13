from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import pipeline
import os
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, CHUNKS_PKL_PATH

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_dim = 384

def load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            index = faiss.read_index(FAISS_INDEX_PATH)
        except Exception:
            index = faiss.IndexFlatL2(embedding_dim)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
    return index

def load_document_chunks():
    if os.path.exists(CHUNKS_PKL_PATH):
        try:
            with open(CHUNKS_PKL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return []
    return []

index = load_faiss_index()
document_chunks = load_document_chunks()
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def query_document(user_question, top_k=3):
    if len(document_chunks) == 0 or index.ntotal == 0:
        return "No document chunks found. Please process a document first."
    question_embedding = embedding_model.encode([user_question])
    question_embedding = np.array(question_embedding).astype('float32')
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [document_chunks[idx] for idx in indices[0] if idx < len(document_chunks)]
    if not relevant_chunks:
        return "No relevant chunks found."
    context = " ".join(relevant_chunks)
    try:
        result = qa_pipeline(question=user_question, context=context)
        return result.get('answer', "No answer found.")
    except Exception as e:
        return f"Error in QA pipeline: {e}"
