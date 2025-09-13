import streamlit as st
from embedding_handler import embed_and_store, get_document_chunks
from qa_agent import query_document, index

st.set_page_config(page_title="DocuSense-AI", layout="wide")
st.title("📄 DocuSense-AI: Document Q&A")

st.header("1️⃣ Upload Documents")
uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for uploaded_file in uploaded_files:
        text = uploaded_file.read().decode("utf-8")
        chunks = [p for p in text.split("\n") if p.strip()]
        all_chunks.extend(chunks)
    if all_chunks:
        embed_and_store(all_chunks)
        st.success(f"Processed and stored {len(all_chunks)} chunks from uploaded files.")

st.header("2️⃣ Ask Questions")
user_question = st.text_input("Enter your question here:")

if st.button("Get Answer") and user_question.strip():
    with st.spinner("Searching for answer..."):
        answer = query_document(user_question)
    st.markdown(f"**Answer:** {answer}")

st.sidebar.header("📊 Info")
chunks = get_document_chunks()
st.sidebar.write(f"Total stored chunks: {len(chunks)}")
st.sidebar.write(f"FAISS index total vectors: {index.ntotal}")
