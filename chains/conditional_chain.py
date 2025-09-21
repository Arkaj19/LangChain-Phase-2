from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key= os.getenv('GEMINI_API_KEY')
)

class Sentiment(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description="This is the sentiment that is being conveyed by the text")

parser = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=Sentiment)

clasify_prompt = PromptTemplate(
    template="Classify whether the input is positive or negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = clasify_prompt | model | parser2

# Prompt for response on positive feedback
pos_prompt = PromptTemplate(
    template="In one short sentence, thank the user for this positive feedback: {feedback}",
    input_variables= ['feedback']
)

# Prompt for response on negative feedback
neg_prompt = PromptTemplate(
    template="Give a single, empathetic sentence responding to this negative feedback: {feedback}",
    input_variables= ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', pos_prompt | model | parser),
    (lambda x:x.sentiment == 'negative', neg_prompt | model | parser),
    RunnableLambda(lambda x: "Could not find Sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({ 'feedback': "This is an wonderful phone"})
print(result)

chain.get_graph().print_ascii()