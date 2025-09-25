from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = GoogleGenerativeAI(
    model= 'gemini-1.5-flash',
    google_api_key = os.getenv('GEMINI_API_KEY')
)

prompt1 = PromptTemplate(
    template="Write a tweet about {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template="Write a linkedin post about {topic}",
    input_variables= ['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence( prompt1, model, parser),
    'linkedin': RunnableSequence( prompt2, model, parser),
})

result = parallel_chain.invoke({'topic' : 'Job Recession'})

print(result)