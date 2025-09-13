from transformers import pipeline

# Summarization pipeline
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

def summarize_document(document_text):
    summary = summarizer(document_text, max_length=200, min_length=50, do_sample=False)
    return summary[0]['summary_text']
