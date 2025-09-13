from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(document_text, chunk_size=1000, chunk_overlap=200):
    """
    Splits the document text into smaller chunks for embedding.

    Args:
        document_text (str): Full text of the document.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(document_text)
    return chunks
