from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(    
    model="gemini-1.5-flash",   # or another model name
    google_api_key=os.getenv("GEMINI_API_KEY"))

parser = JsonOutputParser() # <- We need to instantiate it with that ()

template = PromptTemplate(
    template="Give me the name, age, location and city of a fictional character of Harry Potter \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({})
# Everytime we do invoke we need to send a dictionary, but since we arent passing anything we need to send empty at least
print( result)