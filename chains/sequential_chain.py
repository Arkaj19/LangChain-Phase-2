from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    google_api_key= os.getenv('GEMINI_API_KEY')
)

template1 = PromptTemplate(
    template="Give me a detailed report of \n {topic}",
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template = "Give me 5 interesting short points on the report \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({ 'topic' : 'American Civil War'})

print(result)

chain.get_graph().print_ascii()