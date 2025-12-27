
import os
from pypdf import PdfReader

import re
from typing import List, Dict

PDF_DIR = "data/pdfs"


def load_pdfs(pdf_dir = PDF_DIR):
    documents = []

    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue

        path = os.path.join(pdf_dir, fname)
        reader = PdfReader(path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            documents.append({
                "source" : fname,
                "page" : page_num + 1,
                "text" : text
            })

    return documents


def clean_text(text: str) -> str: # variable : typehints -> output
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip() # similar to java's .trim() but you can specify exactly what to remove


def chunk_text(
        documents: List[Dict],
        chunk_size: int = 800,
        chunk_overlap: int = 200
):
    chunks = []

    for doc in documents:
        text = clean_text(doc["text"])
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            if len(chunk) < 100:
                break

            chunks.append ({
                "source" : doc["source"],
                "page" : doc["page"],
                "text" : chunk
            })

            start = end - chunk_overlap
    
    return chunks

# if __name__ == "__main__":
#     docs = load_pdfs()
#     chunks = chunk_text(docs)
#     print(f"Loaded {len(docs)} pages.")
#     print(f"Created {len(chunks)} chunks.")


import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "research_papers"


def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2") # general purpose model


def get_chroma_client():
    client = chromadb.Client(Settings(
        persist_directory = CHROMA_DIR, 
        anonymized_telemetry = False
        ))
    return client


def build_vector_store(chunks):
    client = get_chroma_client()

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    model = get_embedding_model()

    texts = [c["text"] for c in chunks]
    metas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    ids = [f"{i}" for i in range (len(chunks))]

    embeddings = model.encode(texts, show_progress_bar = True).tolist()

    collection.add (
        ids = ids,
        documents = texts,
        embeddings = embeddings,
        metadatas = metas
    )

    # client.persist()
    print(f"Stored {len(texts)} chunks in Chroma.")


if __name__ == "__main__":
    docs = load_pdfs()
    chunks = chunk_text(docs)
    print(f"Loaded {len(docs)} pages.")
    print(f"Created {len(chunks)} chunks.")
    build_vector_store(chunks)
