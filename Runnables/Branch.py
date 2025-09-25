from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template="Write a report about {topic}",
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the report created within 500 words:  {text}",
    input_variables=['text']
)

model = GoogleGenerativeAI(
    model= 'gemini-1.5-flash',
    google_api_key = os.getenv('GEMINI_API_KEY')
)

parser = StrOutputParser()

report_gen_chain = RunnableSequence( prompt, model, parser)

branch_chain = RunnableBranch(
    ( lambda x: len(x.split()) > 500, RunnableSequence( prompt2, model, parser)),
    RunnablePassthrough
)

final_chain = RunnableSequence( report_gen_chain, branch_chain)

print(final_chain.invoke({'topic' : 'Cricket'}))

