import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException  # type: ignore[import]
from pydantic import BaseModel

from app import InterviewAssistant

# Load environment variables from the project .env file
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

assistant = InterviewAssistant()
app = FastAPI(
    title="DSPy Gemini AI Assistant",
    description="Ask technical questions via FastAPI and explore the interactive Swagger UI at /docs.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {
        "message": "DSPy Gemini AI Assistant is running. Open /docs to use Swagger UI."
    }


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    try:
        result = assistant(question=payload.question)
        return AnswerResponse(answer=result.answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
