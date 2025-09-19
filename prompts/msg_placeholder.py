from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

#Chat Template
chat_template = ChatPromptTemplate([
    ('system','You are a helpful Customer Support Agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history = []

#Load Chat History
with open('chathistory.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

#Create Prompts
prompt = chat_template.invoke({'chathistory': chat_history, 'query': 'Where is my Refund?'})

print(prompt)