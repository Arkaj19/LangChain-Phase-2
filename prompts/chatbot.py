from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(    
    model="gemini-1.5-flash",   # or another model name
    google_api_key=os.getenv("GEMINI_API_KEY"))

chat_history = [
    SystemMessage(content="You are a very good Assistant")
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('AI: ', result.content)

print(chat_history)


# # Streamlit UI
# st.title("Gemini Chat App")

# # Keep conversation history
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# # Display previous messages
# for message in st.session_state["messages"]:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # Input box for the user
# user_input = st.chat_input("Type your message")

# if user_input:
#     # Show user message immediately
#     st.session_state["messages"].append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.write(user_input)

#     # Get model response
#     result = model.invoke(user_input)
#     bot_reply = result.content

#     # Show AI reply
#     st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
#     with st.chat_message("assistant"):
#         st.write(bot_reply)