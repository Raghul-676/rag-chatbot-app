import fitz 
import os
import numpy as np
import requests
import chromadb
from sentence_transformers import SentenceTransformer

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded")


chroma_client = chromadb.PersistentClient(path="./my_vector_store")
collection = chroma_client.get_or_create_collection(name="policy_chunks")
print("Connected to ChromaDB")


def get_top_k_context(query, k=5):
    query_embedding = embedding_model.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0]

def query_groq(question, context, model="llama3-8b-8192"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are an AI assistant that reviews insurance policy documents. "
        "Your goal is to return a structured JSON with the following:\n\n"
        "Format:\n"
        "{\n"
        '  "decision": "If the question is covered by the policy approved else rejected",\n'
        '  "amount": "INR value or N/A",\n'
        '  "justification": "Explain the decision by directly referencing the policy document as short as possible using basic english."\n'
        "}\n\n"
        "Only base your answer on the context provided. Do NOT assume anything. "
        "If you lack information, reject the claim with justification."
    )

    prompt = f"""Context:\n{context}\n\nQuestion: {question}\n\nReturn only the JSON result."""

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 700
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.status_code}\n{response.text}")

    return response.json()["choices"][0]["message"]["content"]

def chat_with_pdf():
    print("Ask policy related questions, Type 'exit' to stop.\n")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        try:
            context_chunks = get_top_k_context(question)
            context = "\n\n".join(context_chunks)
            structured_answer = query_groq(question, context)
            print(f"\n Structured Decision:\n{structured_answer}\n")
        except Exception as e:
            print("Error:", e)
            
# chat_with_pdf()