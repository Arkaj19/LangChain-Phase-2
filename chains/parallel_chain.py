from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=os.getenv('GEMINI_API_KEY')
)

report_prompt = PromptTemplate(
    template="Give me a detailed report on the \n {text}",
    input_variables=['text']
)

summary_prompt = PromptTemplate(
    template="Give me summary notes on the \n {text}",
    input_variables=['text']
)

quiz_prompt = PromptTemplate(
    template="Give me 10 quiz questions on the topic \n {text}",
    input_variables=['text']
)

merge_prompt = PromptTemplate(
    template="Merge the provided notes and the quiz questions in a single document and return the user \n  notes ->{notes} and questions -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

# First we need to generate the chain where we get the big report
report_chain = report_prompt | model | parser

# Now we will generate the parallel chain
parallel_chain = RunnableParallel({
    'notes' : summary_prompt | model | parser,
    'quiz' : quiz_prompt | model | parser
})

merge_chain = merge_prompt | model | parser

chain = report_chain | parallel_chain | merge_chain

result = chain.invoke({ 'text' : 'Mohenjo Daro Civilization'})

print(result)

chain.get_graph().print_ascii()