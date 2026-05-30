# Forward exports from the modularized message_assistant sub-package to keep external imports intact.
from .message_assistant_impl.service import (
    answer_about_conversation,
    generate_message_assistant_response,
    stream_answer_about_conversation,
)

__all__ = [
    "generate_message_assistant_response",
    "answer_about_conversation",
    "stream_answer_about_conversation",
]
