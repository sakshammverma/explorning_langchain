from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

template1 = PromptTemplate(
    template='Write a detailed note on the given {topic}',
    input_variables=['topic']
)


template2 = PromptTemplate(
    template='Write a short 2 line summary on the given {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic' : 'Royal Challenger Bangalore'})

print(result)