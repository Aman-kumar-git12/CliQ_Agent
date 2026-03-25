from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Stable model for classification and question rewriting
classifier_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    streaming=True
)

# Slightly creative model for final user-facing answers
response_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    streaming=True
)

# Backward-compatible export for any older code paths
llm = response_llm
