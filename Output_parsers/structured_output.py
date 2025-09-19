from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
 
load_dotenv()

model = ChatGoogleGenerativeAI(    
    model="gemini-1.5-flash",   # or another model name
    google_api_key=os.getenv("GEMINI_API_KEY"))

schema = [
    ResponseSchema(name='Fact-1', description="Fact 1 about the topic"),
    ResponseSchema(name='Fact-2', description="Fact 2 about the topic"),
    ResponseSchema(name='Fact-3', description="Fact 3 about the topic"),
    ResponseSchema(name='Fact-4', description="Fact 4 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 4 interesting facts about the topic \n {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic':'Hogwarts'})
print(result)