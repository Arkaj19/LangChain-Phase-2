from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Load the document
loader = TextLoader('docs.txt')
documents = loader.load()

# 2. Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 3. Create Gemini embeddings with model name
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 4. Convert text into embeddings & store in FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Create a Retriever
retriever = vectorstore.as_retriever()

# 6. Retrieve relevant documents
query = "When did the Chernobyl disaster occur?"
retrieved_docs = retriever.get_relevant_documents(query)
retrieved_text = "\n".join(doc.page_content for doc in retrieved_docs)

# 7. Initialize Gemini LLM
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# 8. Ask Gemini with retrieved context
prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"
answer = llm.invoke(prompt)

print("Answer:", getattr(answer, "text", answer))
