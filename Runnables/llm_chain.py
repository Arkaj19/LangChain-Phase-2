from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    google_api_key = os.getenv('GEMINI_API_KEY')
)

prompt = PromptTemplate(
    template = "Suggest a catchy blog title for the {topic}",
    input_variables=['topic']
)

chain = LLMChain(llm = model , prompt = prompt )

topic = input('Enter the topic')
output = chain.run(topic)

print("Generated Blog title", output)