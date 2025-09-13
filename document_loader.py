from PyPDF2 import PdfReader

def load_pdf(file):
    """
    Extract text from a PDF file.

    Args:
        file: Uploaded file (Streamlit file uploader or local path)

    Returns:
        str: Extracted text from all PDF pages
    """
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def load_txt(file):
    """
    Read text from a TXT file.

    Args:
        file: Uploaded file (Streamlit file uploader or local path)

    Returns:
        str: Full text content of the TXT file
    """
    file.seek(0)  # Ensure pointer is at the start
    return file.read().decode("utf-8")
