from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = GoogleGenerativeAI(
    model='gemini-1.5-flash',
    google_api_key = os.getenv('GEMINI_API_KEY')
)

loader = TextLoader('cricket.txt')

docs = loader.load()

prompt = PromptTemplate(
    template="Write a poem from the report : {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'text' : docs[0].page_content})
print(result)