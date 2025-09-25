from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = GoogleGenerativeAI(
    model = 'gemini-1.5-flash',
    google_api_key= os.getenv('GEMINI_API_KEY')
)

#Define the input
topic = input('Enter the topic')

prompt = PromptTemplate(
    template = "Suggest a catchy title for my blog post \n {topic}",
    input_variables= ['topic']
)

formatted_prompt = prompt.format(topic = topic)

blog_title = model.invoke(formatted_prompt)

print('Generated blog title: ', blog_title) 