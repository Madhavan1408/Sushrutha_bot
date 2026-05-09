from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re 
def clean_text(text : Document) -> str:
    text=text.page_content
    data=re.sub(r"\s+"," ",text)
    data=re.sub(r"\n+"," ",data)
    data=re.sub(r"[^\x20-\x7E]"," ",data)
    return data.strip()
def load_documents():
    folder=r"D:\AI Course\medial_bot\data"
    all_texts=[]
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            file_path=os.path.join(folder,file)
            loader=PyPDFLoader(file_path)
            pages=loader.load()
            for page in pages:
                cl_text=clean_text(page)
                all_texts.append(
                    Document(
                    page_content=cl_text,
                    metadata={"source":file}
                        )
                    )
    return all_texts
def embeddings_vector(all_texts):
    chunks=RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        seperators=["\n\n","\n",". "," "]
    )
    chunkes=chunks.split_documents(all_texts)
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )
    # BUilding FAISS Index this handles embedding internally
    vector_store=FAISS.from_documents(
        documents=chunkes,
        emdbedding=embeddings
    )
    
    # Save this embeddings vectors to local disk
    save_path=r"D:\AI Course\medial_bot"
    vector_store.save_local(save_path)
    print(f"the vector data stored in local device in path {save_path}")
    return vector_store
def load_vector_store(save_path : str) -> FAISS:
    embeddings=HuggingFaceEmbeddings(
        model_name="senetence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device':'cpu'}
    )
    vectorstore=FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore
if __name__ =="__main__":
    folder="D:\AI Course\medial_bot\data"
    index_path=r"D:\AI Course\medial_bot\faiss_index"
    if os.path.exists(index_path):
        vectorstore=load_vector_store(index_path)
    else:
        documents=load_documents()
        vectorstore=embeddings_vector(documents)
    test_query="what is Gynacoleogy"
    results=vectorstore.similarity_search(test_query)
    print(results)