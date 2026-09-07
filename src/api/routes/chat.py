from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.talk_to_data.nl_to_sql import ask

router = APIRouter()

class ChatRequest(BaseModel):
    query: str

@router.post("/chat")
def chat(request: ChatRequest):
    try:
        result = ask(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
