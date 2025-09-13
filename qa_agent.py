
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import pipeline
import os
from config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, CHUNKS_PKL_PATH

# Load FAISS index and document chunks safely
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_dim = 384

if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    index = faiss.IndexFlatL2(embedding_dim)

if os.path.exists(CHUNKS_PKL_PATH):
    with open(CHUNKS_PKL_PATH, 'rb') as f:
        document_chunks = pickle.load(f)
else:
    document_chunks = []

# Load HuggingFace QA pipeline
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def query_document(user_question, top_k=3):
    question_embedding = embedding_model.encode([user_question])
    question_embedding = np.array(question_embedding).astype('float32')

    # Search top-k relevant chunks
    if len(document_chunks) == 0 or index.ntotal == 0:
        return "No document chunks found. Please process a document first."

    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [document_chunks[idx] for idx in indices[0]]

    # Prepare context
    context = " ".join(relevant_chunks)

    # Get answer
    result = qa_pipeline(question=user_question, context=context)
    return result['answer']
