from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about \n {topic}",
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(    
    model="gemini-1.5-flash",   # or another model name
    google_api_key=os.getenv("GEMINI_API_KEY"))

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic' : 'Demon Slayer'})

print(result)

chain.get_graph().print_ascii()