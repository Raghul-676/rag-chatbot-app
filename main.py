import json
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import List
from app import get_top_k_context, query_groq 
import os

from dotenv import load_dotenv
load_dotenv()  

API_KEY = os.getenv("API_KEY","my-secure-api-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(title="RAG API")

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key")

class QueryRequest(BaseModel):
    questions: List[str]


@app.post("/rag-query")
async def rag_query(req: QueryRequest, api_key: str = Depends(get_api_key)):
    try:
        results = []
        for q in req.questions:
            context_chunks = get_top_k_context(q)
            context = "\n\n".join(context_chunks)
            structured_answer = query_groq(q, context)
            results.append({
                "question": q,
                "answer": json.loads(structured_answer)
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
async def root():
    return {"message": "Welcome to RAG API. Use POST /rag-query"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # 8000 for local dev, Render injects PORT in production
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
