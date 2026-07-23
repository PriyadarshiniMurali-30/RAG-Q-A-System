from fastapi import FastAPI
from pydantic import BaseModel
from query import answer_question

app = FastAPI(title="Document Q&A API")


# This defines what a valid request must look like
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def root():
    return {"message": "Document Q&A API is running. POST to /ask with a question."}


@app.post("/ask")
def ask(request: QuestionRequest):
    result = answer_question(request.question, top_k=request.top_k)
    return result