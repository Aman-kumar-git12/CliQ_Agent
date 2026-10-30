import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

classifier_model = os.getenv("MESSAGE_ASSISTANT_CLASSIFIER_MODEL", "llama-3.1-8b-instant")
response_model = os.getenv("MESSAGE_ASSISTANT_RESPONSE_MODEL", "llama-3.1-8b-instant")
fallback_response_model = os.getenv("MESSAGE_ASSISTANT_FALLBACK_MODEL", classifier_model)

# Stable model for classification and question rewriting
classifier_llm = ChatGroq(
    model=classifier_model,
    temperature=0,
    streaming=True
)

# Slightly creative model for final user-facing answers
response_llm = ChatGroq(
    model=response_model,
    temperature=0.7,
    streaming=True
)

# Lower-variance fallback for structured outputs and degraded-mode recovery
fallback_response_llm = ChatGroq(
    model=fallback_response_model,
    temperature=0.2,
    streaming=True
)

# Help chatbot router: rewrites + splits + classifies in one JSON call.
# Groq JSON mode cannot stream, so streaming is disabled even inside astream_events.
# max_retries=1: the SDK already retries once; more retries only delay the error when a limit is hit.
help_router_llm = ChatGroq(
    model=classifier_model,
    temperature=0,
    disable_streaming=True,
    max_retries=1,
).bind(response_format={"type": "json_object"})

# Help chatbot answers: low temperature so the same question gets the same facts
help_answer_llm = ChatGroq(
    model=response_model,
    temperature=0.2,
    streaming=True,
    max_retries=1,
)

# Backward-compatible export for any older code paths
llm = response_llm
