from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-pro")
result=model.invoke('Write a shayari on mouse in english')

print(result.content)

