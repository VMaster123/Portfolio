from fastapi import FastAPI
from pydantic import BaseModel
import requests

from db import get_docs
from rag import add_documents, retrieve

app = FastAPI(title="Local RAG System (FAISS + Ollama)")


# ----------------------------
# Request schema
# ----------------------------
class QueryRequest(BaseModel):
    q: str


# ----------------------------
# Load FAISS on startup
# ----------------------------
@app.on_event("startup")
def startup_event():
    docs = get_docs()
    add_documents(docs)
    print(f"🔥 FAISS loaded with {len(docs)} documents")


# ----------------------------
# Main endpoint
# ----------------------------
@app.post("/query")
def query(req: QueryRequest):
    question = req.q

    docs = retrieve(question)
    context = "\n".join([d[1] for d in docs]) if docs else ""

    answer = generate_answer(context, question)

    return {"question": question, "context": context, "answer": answer}


# ----------------------------
# OLLAMA LLM FUNCTION (THIS IS WHERE YOUR SNIPPET GOES)
# ----------------------------
def generate_answer(context: str, question: str) -> str:
    prompt = f"""
You are a strict physics and engineering tutor.

Rules:
- Use ONLY the provided context
- Be direct and technical
- No greetings or fluff

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
            },
            timeout=120,
        )

        response.raise_for_status()
        return response.json()["response"]

    except Exception as e:
        return f"ERROR calling Ollama: {str(e)}"
