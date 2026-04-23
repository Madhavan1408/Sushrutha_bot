from splitter import split_documents, medical_data
from langchain_huggingface import HuggingFaceEmbeddings  # updated import
from langchain_community.vectorstores import FAISS       # store embeddings properly

print(f"Splitting {len(medical_data)} documents into chunks...")
chunks = split_documents(medical_data)
print(f"Total chunks created: {len(chunks)}")

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # fixed capitalization
    model_kwargs={'device': 'cpu'}
)

# Store chunks + embeddings in a vector store (FAISS)
vector_store = FAISS.from_documents(chunks, embeddings)
print(f"Generated and stored embeddings for {len(chunks)} chunks.")
vector_store.save_local("faiss_index")
print("FAISS index saved locally as 'faiss_index'.")