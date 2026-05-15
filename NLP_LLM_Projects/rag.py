import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384
index = faiss.IndexFlatL2(dimension)

texts = []


def add_documents(docs):
    global texts, index

    texts = [d[1] for d in docs]  # ONLY store strings

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    index.reset()  # IMPORTANT: prevent mismatch bugs
    index.add(embeddings)


def retrieve(query, k=3):
    global texts

    if len(texts) == 0:
        return []

    query_vec = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vec, k)

    results = []

    for i in indices[0]:
        if 0 <= i < len(texts):   # 🔥 SAFE GUARD (fixes crash)
            results.append((i, texts[i]))

    return results