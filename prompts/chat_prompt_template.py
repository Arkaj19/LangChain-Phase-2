from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Help me in this {topic}')
])

prompt = chat_template.invoke({'domain':'Java', 'topic':'Destructuring'} ) #We are sending a Dictionary of values
print(prompt)   