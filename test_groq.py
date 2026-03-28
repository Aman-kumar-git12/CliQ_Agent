import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

key = os.getenv("GROQ_API_KEY")
print(f"Key found: {key[:10]}...{key[-5:]}" if key else "Key NOT found")

try:
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=key)
    response = llm.invoke([HumanMessage(content="Hello")])
    print("Response successful!")
    print(f"Content: {response.content}")
except Exception as e:
    print(f"Error calling Groq: {e}")
