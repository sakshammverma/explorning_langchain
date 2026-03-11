from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# LangChain will automatically look for GOOGLE_API_KEY in the environment
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Define the Schema using Pydantic BaseModel
class ReviewSummary(BaseModel):
    summary: str = Field(description="A brief overview of the main points")
    
    sentiment: Literal["Positive", "Negative"] = Field(description="The overall tone")
    
    # Adding Constraints (e.g. rating out of 5)
    rating: float = Field(ge=0, le=5, description="A rating out of 5 based on the review")
    
    # Optional Fields
    cons: Optional[list[str]] = Field(default=None, description="List of negatives")

# Bind the schema
structured_model = model.with_structured_output(ReviewSummary)

review_text = """ I've been using the new Aether Phone X for about two weeks now, and overall, I'm pretty impressed. The 120Hz OLED display is absolutely gorgeous—super smooth for scrolling and bright enough for direct sunlight. Battery life is stellar; I easily get through a heavy day of work with 20% left by bedtime.
However, the camera system is just 'okay' for this price point. In good lighting, the 50MP main sensor takes great, sharp photos. But in low light, there is significant noise, and the shutter lag is noticeable. The software is clean, which I love, but it’s missing a few customization features I had on my last phone.
Pros:
Incredible battery life (lasts all day +).
Fast 120Hz screen.
Clean, bloatware-free software.
Cons:
Mediocre low-light camera performance.
The phone feels a bit slippery without a case.
18W charging feels slow in 2026.
Verdict: Great if you prioritize battery and screen, but photographers might want to look elsewhere."""
result = structured_model.invoke(review_text)

# The result is a Pydantic Object (you access variables using dot notation)
print(result.summary)
print(result.rating) # Output: 4.0
print(result.sentiment) # Output: Positive

# You can easily convert it back to a dictionary if needed
dict_result = result.model_dump()
