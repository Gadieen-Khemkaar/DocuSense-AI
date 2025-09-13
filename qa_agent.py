from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import pipeline

# Load saved index and chunks
index = faiss.read_index('faiss_index.bin')
with open('document_chunks.pkl', 'rb') as f:
    document_chunks = pickle.load(f)

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load HuggingFace QA pipeline
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def query_document(user_question, top_k=3):
    question_embedding = embedding_model.encode([user_question])
    question_embedding = np.array(question_embedding).astype('float32')
    
    # Search top-k relevant chunks
    distances, indices = index.search(question_embedding, top_k)
    relevant_chunks = [document_chunks[idx] for idx in indices[0]]
    
    # Prepare context
    context = " ".join(relevant_chunks)
    
    # Get answer
    result = qa_pipeline(question=user_question, context=context)
    return result['answer']
