from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

json_schema = {
    "title": "Review",
    "description": "Extract information from a smartphone review",
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A brief overview"
        },
        "sentiment": {
            "type": "string",
            "enum": ["Positive", "Negative"],
            "description": "Overall tone"
        }
    },
    "required": ["summary", "sentiment"]
}

# Bind the schema
structured_model = model.with_structured_output(json_schema)

review_text = "The screen is beautiful."
result = structured_model.invoke(review_text)

# Result is returned as a Python Dictionary
print(result)