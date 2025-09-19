from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(  
    repo_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

#1st Prompt -> Detailed Report
template = PromptTemplate(
    template='Write a detailed Report on {topic}',
    input_variables=['topic']
)

#2nd Prompt -> Summary
template2 = PromptTemplate(
    template='Write a short 5 line summary on the following. \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'Black Hole'})

print(result)
