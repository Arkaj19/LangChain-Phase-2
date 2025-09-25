from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables= ['topic']
)

model = GoogleGenerativeAI(
    model= 'gemini-1.5-flash',
    google_api_key = os.getenv('GEMINI_API_KEY')
)

parser = StrOutputParser()

prompt_explain = PromptTemplate(
    template="Elaborate on the line:  {text}",
    input_variables= ['text']
)

chain = RunnableSequence(prompt, model, parser, prompt_explain, model, parser)
# Here it suggests that the model should send the prompt to the model and then the model shoud be sending the output tot the parser

print(chain.invoke({'topic' : 'Computer'}))

#We will be making this more complex by taking the output being generated from the model and then again sending it to the model to elaborate on the generated joke