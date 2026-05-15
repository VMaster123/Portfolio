from db import insert_doc, get_docs
from rag import add_documents

def ingest_file(path="data/sample.txt"):
    print("🔥 Starting ingestion...")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text.split("\n\n")
    print(f"📦 Chunks found: {len(chunks)}")

    for c in chunks:
        insert_doc(c)

    docs = get_docs()
    print(f"🗄️ Docs in SQLite: {len(docs)}")

    add_documents(docs)

    print("✅ Ingestion complete")

if __name__ == "__main__":
    ingest_file()