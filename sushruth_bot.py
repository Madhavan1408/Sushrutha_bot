from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda,RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# --- SETUP ---
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "faiss_index",
    embed_model,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={'k': 3})
search = DuckDuckGoSearchRun()

# --- SMART RETRIEVAL ---
def smart_retrieval(question: str) -> str:
    pdf_docs = retriever.invoke(question)
    pdf_context = "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]: {d.page_content}"
        for d in pdf_docs
    )

    # Check if PDF context is actually relevant
    if len(pdf_context.strip()) < 200:
        try:
            web_result = search.run(question)
            return f"[Web Search Result]:\n{web_result}"
        except:
            return pdf_context

    return f"[Document Knowledge]:\n{pdf_context}"

# --- PROMPT ---
SYSTEM_PROMPT = """You are a friendly expert who explains anything 
in simple, clear language that anyone can understand.

HOW YOU COMMUNICATE:
1. Give the simple answer first in 1-2 sentences
2. Then explain with an everyday analogy or example
3. Break down any complex terms immediately
4. Use short sentences
5. End with one practical takeaway

HOW YOU USE KNOWLEDGE:
- Use the context provided if relevant
- Use your general knowledge for anything else
- Be honest when uncertain

Context:
{context}"""

# --- LLM (Upgraded) ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

prompt = ChatPromptTemplate([
    ('system', SYSTEM_PROMPT),
    ('human', "{question}")
])

def add_disclaimer(answer: str) -> str:
    if any(word in answer.lower() for word in
           ['medicine', 'drug', 'disease', 'symptom',
            'doctor', 'medical', 'treatment']):
        return answer + "\n\n⚠️ For medical topics, always consult a doctor."
    return answer

# --- CHAIN ---
smart_chain = (
    RunnableParallel({
        "context": RunnableLambda(smart_retrieval),
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(add_disclaimer)
)

# --- CHAT LOOP WITH MEMORY ---
chat_history = []

def chat(user_input: str) -> str:
    try:
        answer = smart_chain.invoke(user_input)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))
        return answer
    except Exception as e:
        return f"Error: {e}"

# --- RUN ---
print("Bot is ready! Type 'quit' to exit.\n")
while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() == 'quit':
        break
    answer = chat(question)
    print(f"\nBot: {answer}\n")
    print("-" * 50)