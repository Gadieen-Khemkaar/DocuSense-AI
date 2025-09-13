import streamlit as st
from document_loader import load_pdf, load_txt
from text_splitter import split_text
from embedding_handler import embed_and_store
from qa_agent import query_document
from summarizer import summarize_document

st.set_page_config(page_title="🌟 DocuSense AI", layout="wide")

st.title("🌟 DocuSense AI - Document Summarizer & Q&A Agent")

uploaded_file = st.file_uploader("📄 Upload your PDF or TXT document", type=['pdf', 'txt'])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        document_text = load_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        document_text = load_txt(uploaded_file)
    else:
        st.error("❌ Unsupported file type.")
        st.stop()

    st.success("✅ Document loaded successfully!")
    st.write("---")

    # Step 1: Summarize document
    if st.button("📝 Summarize Document"):
        with st.spinner("Generating summary..."):
            summary = summarize_document(document_text)
            st.subheader("📑 Document Summary:")
            st.write(summary)

    st.write("---")

    # Step 2: Process embeddings
    if st.button("🔧 Process Document for Q&A"):
        with st.spinner("Processing and storing embeddings..."):
            chunks = split_text(document_text)
            embed_and_store(chunks)
            st.success(f"✅ Processed {len(chunks)} text chunks into FAISS embeddings.")

    st.write("---")

    # Step 3: Ask Questions
    user_question = st.text_input("❓ Ask a question related to the document:")
    if st.button("💬 Get Answer"):
        with st.spinner("Searching for the best answer..."):
            answer = query_document(user_question)
            st.subheader("🤖 Answer:")
            st.write(answer)
