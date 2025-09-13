# 🌟 DocuSense AI

**DocuSense AI** is a fully open-source document summarization and Q\&A agent.
It allows you to upload PDF or TXT documents, generate summaries, and ask questions about the document using semantic search.

---

## ✅ Features

* Upload PDF or TXT documents.
* Generate a concise summary of the document.
* Split the document into chunks for embeddings.
* Generate embeddings with **Sentence-Transformers**.
* Store embeddings in **FAISS** for efficient semantic search.
* Ask questions and get context-aware answers using HuggingFace QA models.
* Fully free and open-source — no OpenAI API required.
* Interactive interface built with **Streamlit**.

---

## 🗂 Project Structure

```
document_qa_agent/
│
├── app.py                 ← Streamlit interface (upload, summarize, Q&A)
├── document_loader.py     ← Load PDF / TXT files
├── text_splitter.py       ← Split documents into chunks
├── embedding_handler.py   ← Generate embeddings + store in FAISS
├── qa_agent.py            ← Semantic search + Q&A logic
├── summarizer.py          ← Document summarization
├── config.py              ← Configuration (models, chunk size, paths)
├── requirements.txt       ← Python dependencies
└── README.md
```

---

## 🛠 Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd document_qa_agent
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv docusense_env
# Activate (Windows)
docusense_env\Scripts\activate
# Activate (Linux/Mac)
source docusense_env/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

* Upload a PDF or TXT document.
* Click **“Summarize Document”** to get a summary.
* Click **“Process Document for Q\&A”** to generate embeddings.
* Ask questions in the input box and click **“Get Answer”**.

---

## ⚡ Dependencies

* [sentence-transformers](https://www.sbert.net/)
* [faiss-cpu](https://github.com/facebookresearch/faiss)
* [transformers](https://huggingface.co/transformers/)
* [PyPDF2](https://pythonhosted.org/PyPDF2/)
* [python-docx](https://python-docx.readthedocs.io/en/latest/) (optional for DOCX support)
* [Streamlit](https://streamlit.io/)

---

## 📝 Configuration

* **Models, chunk sizes, and paths** are stored in `config.py`.
* You can change the summarization or QA models easily without touching the main code.

---

## 💡 Notes

* The first run downloads pre-trained models; may take a few minutes.
* FAISS index and document chunks are saved locally for faster subsequent queries.
* Fully offline once models are downloaded.


