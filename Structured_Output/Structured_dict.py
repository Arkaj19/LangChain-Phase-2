from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, EmailStr
from typing import Literal
import os

load_dotenv()

model = ChatGoogleGenerativeAI(    
    model="gemini-1.5-flash",   # or another model name
    google_api_key=os.getenv("GEMINI_API_KEY"))

#Class
class Review(BaseModel):
    summary : str = Field(description="Here it will contain ta short summary of the entire review")
    sentiment : Literal["pos", "neg"] = Field(description='This defines whether it is postive or negative')
    name : str = Field(description="This is the name of the reviewer")

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.
Given by : Arkajyoti""")

print(dict(result))

