from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import app_router
import uvicorn
import logging

app = FastAPI(
    title="RAG-based QA Service",
    description="A FastAPI application for document upload and RAG-based question answering.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
app.include_router(app_router.router)

@app.get("/")
async def root():
    return {"message": "RAG API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
