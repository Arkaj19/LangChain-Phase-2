from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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

prompt1 = template.invoke({'topic': 'Black Hole'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text' : result1.content})
result2 = model.invoke(prompt2)

print(result2.content)