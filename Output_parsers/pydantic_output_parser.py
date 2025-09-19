from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal
import os

load_dotenv()

model = ChatGoogleGenerativeAI(    
    model="gemini-1.5-flash",   # or another model name
    google_api_key=os.getenv("GEMINI_API_KEY"))

class Wizard(BaseModel):
    name: str = Field(description='Name of the Wizard')
    age: int = Field(gt=10,description='Age of the Wizard')
    house: str = Field(description='Which house does he belong to like Griffyndor, Slytherin, HufflePuff, etc.')

parser = PydanticOutputParser(pydantic_object=Wizard)

template = PromptTemplate(
    template='Generate the name, age and house name of a Wizard of {house_name} in Hogwarts \n {format_instruction}',
    input_variables=['house_name'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


chain = template | model | parser
result = chain.invoke({'house_name': 'Slytherin'})

print(result)