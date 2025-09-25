from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
import os

load_dotenv()

model = GoogleGenerativeAI(
    model='gemini-1.5-flash',
    google_api_key = os.getenv('GEMINI_API_KEY')
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

loader = TextLoader('docs.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
docs = text_splitter.split_documents(documents)

vector_store = FAISS.from_documents(docs, embeddings)

retriever = vector_store.as_retriever()

#Create qachain
qa_chain = RetrievalQA.from_chain_type(llm = model, retriever= retriever)

query = "When did the Chernobyl disaster happen?"
answer = qa_chain.run(query)

print(answer)