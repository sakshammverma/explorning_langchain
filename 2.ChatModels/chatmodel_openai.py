from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model="gpt-4",temperature=0.8)

result=model.invoke('Write a shayari on Keyboard')

print(result)