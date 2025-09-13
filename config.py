# -------------------------------
# Model & Embedding Config
# -------------------------------

# Sentence-Transformers model for embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# HuggingFace summarization model
SUMMARIZATION_MODEL_NAME = "facebook/bart-large-cnn"

# HuggingFace QA model
QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"

# -------------------------------
# Text Splitting Config
# -------------------------------

# Maximum number of characters per chunk
CHUNK_SIZE = 1000

# Overlap between chunks to maintain context
CHUNK_OVERLAP = 200

# -------------------------------
# FAISS / Storage Config
# -------------------------------

# FAISS index file
FAISS_INDEX_PATH = "faiss_index.bin"

# Document chunks pickle file
CHUNKS_PKL_PATH = "document_chunks.pkl"
