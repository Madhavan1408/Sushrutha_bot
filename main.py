from fastapi  import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI(title="Susuhrutha_bot AI Assistant",description="An AI assistant that can answer questions based on the provided context and generate responses using a RAG (Retrieval-Augmented Generation) process.",version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
    
)