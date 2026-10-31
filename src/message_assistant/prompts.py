"""Prompt text for the message assistant LLM calls; wording here is tuned, so change it deliberately."""

# Shared JSON shape the reply generators ask the model to return.
REPLY_JSON_SHAPE = (
    "{"
    '"top_reply": string, '
    '"reply_suggestions": string[], '
    '"emoji_replies_with_emojis": string[], '
    '"emoji_replies_without_emojis": string[], '
    '"emoji_replies": string[], '
    '"rewrites": {"clean": string, "short": string, "warm": string, "confident": string}, '
    '"same_message_variants": string[], '
    '"grouped_replies": {"top_reply": string, "safe_reply": string, "warm_reply": string, '
    '"playful_reply": string, "curious_reply": string, "direct_reply": string}, '
    '"context_focus": string, '
    '"conversation_summary": string, '
    '"detected_intent": string, '
    '"tone": string'
    "}"
)

# Per-tone style guidance appended to reply prompts.
TONE_INSTRUCTIONS = {
    "casual": "Keep it very relaxed and informal. Use everyday language, feeling like a quick text to a peer.",
    "polite": "Be courteous, respectful, and well-mannered.",
    "formal": "Use proper grammar, full sentences, and maintain a professional or respectful distance.",
    "flirty": "Be playful, slightly teasing, and show romantic interest without being overly aggressive or explicit.",
    "professional": "Keep it strictly business-appropriate, clear, objective, and focused on facts.",
    "witty": "Use clever humor, puns, or mild sarcasm to be amusing, smart, and quick on your feet.",
    "direct": "Be straightforward, clear, and to the point. No fluff or beating around the bush.",
    "friendly": "Be warm, approachable, positive, and pleasant, like talking to a good friend.",
    "empathetic": "Show deep understanding, warmth, validation, and care for the other person's feelings.",
    "confident": "Speak with self-assurance, certainty, and decisiveness, without sounding arrogant.",
}

# Per-preference emoji guidance appended to reply prompts.
EMOJI_INSTRUCTIONS = {
    "both": "Generate two parallel styles: one set with emojis and one set without emojis.",
    "with": "Prefer emoji-rich replies and make the emoji version the primary style.",
    "without": "Prefer emoji-free replies and keep emoji usage minimal or absent.",
}

# Safety classifier: decides whether a messaging request should be refused.
SAFETY_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a safety classifier for a private messaging assistant. "
    "Return JSON only with this shape: "
    '{"allow": boolean, "reason": string}. '
    "Set allow to false if the user is asking for help with coercion, abuse, harassment, manipulation, "
    "sexual exploitation, deceit in a harmful way, stalking, threats, or evading consent. "
    "Set allow to true for benign messaging help, boundary-setting, de-escalation, or safety-seeking requests."
)

# Ask-mode router: is the question about this chat or general texting advice.
QUESTION_MODE_SYSTEM_PROMPT = (
    "You classify whether a user question in a chat assistant is about a specific conversation "
    "or is general messaging advice. Return JSON only with this shape: "
    '{"mode": string, "reason": string}. '
    'mode must be either "contextual" or "general".'
)

# Reply-mode router: contextual replies versus sparse-chat opener suggestions.
REPLY_MODE_SYSTEM_PROMPT = (
    "You classify reply-generation strategy for a chat assistant. Return JSON only with this shape: "
    '{"mode": string, "reason": string}. '
    'mode must be either "contextual" or "general". '
    "Choose contextual when the best reply depends on the actual chat context. "
    "Choose general only for sparse first-message/opening-message situations with too little context."
)

# General reply suggestions for sparse or opening conversations.
GENERAL_REPLY_SYSTEM_PROMPT = (
    "You are a private messaging assistant inside a chat app. "
    "The user needs strong reply or opener suggestions for a sparse or early conversation. "
    "Generate actual sendable messages, not advice. "
    "The outputs should feel natural, modern, confident, and easy to reply to. "
    "Do not produce dry or generic lines. "
    "Rank replies by realism, not positivity. "
    "Avoid filler-first replies like 'That's great', 'Sounds good', 'Nice', or 'Awesome' unless they are "
    "expanded with specific context. "
    "Generate both emoji-backed replies and emoji-free replies as separate parallel lists. "
    "If emoji-only would feel weak or confusing, then use short text-plus-emoji messages instead. "
    "Return valid JSON only with this exact shape: " + REPLY_JSON_SHAPE + "."
)

# Ask-mode JSON answer grounded in the conversation.
CONVERSATION_ANSWER_SYSTEM_PROMPT = (
    "You are a messaging assistant inside a private chat app. "
    "Answer only using the recent conversation context. "
    "Be direct, helpful, and concise. "
    "Return valid JSON only with this exact shape: "
    "{"
    '"answer": string, '
    '"suggested_actions": string[]'
    "}."
)

# Ask-mode JSON answer for general messaging advice.
GENERAL_HELP_SYSTEM_PROMPT = (
    "You are a messaging coach inside a private chat app. "
    "The user is asking for general messaging help, not analysis of a specific conversation. "
    "Return valid JSON only with this exact shape: "
    "{"
    '"answer": string, '
    '"suggested_actions": string[], '
    '"suggested_replies": string[], '
    '"mode": string, '
    '"tone": string'
    "}."
)

# Streaming plain-text answer grounded in the conversation.
STREAM_CONVERSATION_SYSTEM_PROMPT = (
    "You are a messaging assistant inside a private chat app. "
    "Answer only using the provided conversation context. "
    "Do not return JSON. Respond in a concise, useful paragraph. "
    "Never mention JSON, schemas, formatting, or internal system limitations."
)

# Streaming plain-text general messaging advice.
STREAM_GENERAL_HELP_SYSTEM_PROMPT = (
    "You are a messaging coach inside a private chat app. "
    "The user wants general messaging help rather than analysis of a specific conversation. "
    "Do not return JSON. Respond with practical advice and a few short example messages when relevant. "
    "Never mention JSON, schemas, formatting, or internal system limitations."
)

# Fixed task list for contextual reply generation.
CONTEXTUAL_REPLY_TASKS = (
    "Tasks:\n"
    "1. Suggest the single best next reply.\n"
    "2. Suggest several alternate replies.\n"
    "3. Provide emoji-friendly replies. Use emoji-only replies when they naturally fit this exact chat, "
    "otherwise use short text-plus-emoji messages.\n"
    "4. Rewrite the user's draft in cleaner ways.\n"
    "5. Provide other ways to say the same thing.\n"
    "6. Briefly summarize the older context if it matters.\n"
    "7. Detect the main intent of this conversation using a short label such as interest, invitation, planning, "
    "dreaming, casual_mention, question, apology, follow_up, negotiation, support, casual, or general.\n"
    "8. Mention what part of the context these replies are based on.\n"
    "9. If the last message is a direct personal question like age, place, job, visit, plan, or preference, "
    "generate specific human replies the user could realistically send right away.\n"
    "10. Prefer replies that sound like a real person texting, not a support bot.\n"
    "11. Do not add a next step like flights, bookings, research, or a committed plan unless the chat clearly "
    "suggests that level of commitment.\n"
    "12. Make the grouped replies meaningfully different in strategy, not just wording.\n"
    "13. Avoid filler-only openings such as 'That's great', 'Sounds good', 'Nice', or 'Awesome' unless followed "
    "by context-specific substance.\n"
    "14. Choose the top reply based on realism and conversational fit, not on maximum enthusiasm.\n"
)


def safety_classifier_user_prompt(text: str) -> str:
    """User turn for the safety classifier."""
    return f"User request:\n{text}"


def question_mode_user_prompt(question: str) -> str:
    """User turn for the ask-mode router."""
    return (
        f"Question: {question}\n\n"
        "Choose contextual if the answer should depend on the user's current chat history. "
        "Choose general if it is broad texting advice, openers, greetings, pickup lines, or generic "
        "message-writing help."
    )


def reply_mode_user_prompt(
    older_text: str, conversation_text: str, last_other: str, draft: str, confidence: str
) -> str:
    """User turn for the reply-mode router."""
    return (
        f"Older context:\n{older_text}\n\n"
        f"Recent conversation:\n{conversation_text}\n\n"
        f"Last other-person message: {last_other or '[Not available]'}\n"
        f"Draft: {draft.strip() or '[No draft provided]'}\n"
        f"Context confidence: {confidence}\n"
    )


# Opening rules for replies grounded in the actual conversation.
CONTEXTUAL_REPLY_INTRO = (
    "You are a private messaging assistant inside a chat app. "
    "Use only the recent conversation to craft helpful replies for the user. "
    "Your job is to generate natural, human-sounding, ready-to-send replies, not generic advice. "
    "If there is any real chat context, treat it as the primary source and avoid falling back to generic "
    "opener advice. "
    "When the other person asks a simple question, answer it directly first and optionally add a light "
    "follow-up. "
    "If the draft is empty, still generate contextual replies from the conversation instead of inventing a "
    "fresh opener. "
    "Anchor your replies to the last message from the other person whenever one exists. "
    "Avoid robotic, vague, repetitive, or placeholder wording. "
    "Do not explain your reasoning. Do not give advice bullets. Only generate actual messages the user can send. "
    "Make replies feel like modern real chat messages. "
    "Rank replies by realism, not positivity. "
    "The best reply should sound like what a real person would actually send next in this exact conversation. "
    "Avoid filler-first replies like 'That's great', 'Sounds good', 'Nice', or 'Awesome' unless they are "
    "expanded with specific context. "
    "Emoji replies should be standalone emoji-only replies when that would feel natural in the conversation. "
    "If emoji-only would feel awkward or unclear, then use short text-plus-emoji messages instead. "
    "Return valid JSON only with this exact shape: " + REPLY_JSON_SHAPE + ". "
)

# Closing grounding and grouped-reply rules for contextual replies.
CONTEXTUAL_REPLY_RULES = (
    "Do not invent actions, plans, bookings, travel arrangements, promises, or commitments unless they are "
    "clearly supported by the conversation. "
    "If the conversation only shows interest or casual discussion, keep replies low-assumption. "
    "If context confidence is low, be extra conservative and avoid introducing any commitment, logistics, or "
    "emotional intensity that is not clearly present. "
    "Classify the conversation using one of these labels when possible: interest, invitation, planning, "
    "dreaming, casual_mention, question, apology, follow_up, negotiation, support, casual, general. "
    "Grouped replies must be genuinely different from each other: "
    "top_reply should be the best overall send-now reply, "
    "safe_reply should be low-risk and grounded only in explicit context, "
    "warm_reply should feel friendly and emotionally open, "
    "playful_reply should be light and a little fun without changing the facts, "
    "curious_reply should move the conversation forward by asking a natural question, "
    "direct_reply should be clear and straightforward without sounding pushy."
)


def contextual_reply_system_prompt(
    max_suggestions: int, other_name: str, tone: str, tone_instruction: str, emoji_instruction: str
) -> str:
    """System prompt for replies grounded in the actual conversation."""
    return (
        CONTEXTUAL_REPLY_INTRO
        + f"Provide exactly {max_suggestions} items in reply_suggestions when possible, 3 items in "
        "emoji_replies_with_emojis, "
        "3 items in emoji_replies_without_emojis, and 4 items in same_message_variants. "
        "Keep the with-emojis and without-emojis lists parallel in meaning. "
        "Make the replies specific to the conversation with "
        f"{other_name}. Keep them natural, concise, and ready to send. "
        f"Default tone for replies should be {tone} unless a rewrite field naturally needs a different nuance. "
        f"{tone_instruction}\n"
        f"{emoji_instruction}\n" + CONTEXTUAL_REPLY_RULES
    )


def contextual_reply_user_prompt(
    older_text: str, conversation_text: str, other_name: str, last_other: str, confidence: str, draft: str
) -> str:
    """User turn for replies grounded in the actual conversation."""
    return (
        f"Older conversation summary or snippets:\n{older_text}\n\n"
        f"Recent conversation:\n{conversation_text}\n\n"
        f"Last message from {other_name}: {last_other or '[Not available]'}\n\n"
        f"Context confidence: {confidence}\n\n"
        f"Current draft from the user: {draft.strip() or '[No draft provided]'}\n\n" + CONTEXTUAL_REPLY_TASKS
    )


def general_reply_user_prompt(
    draft: str, tone: str, tone_instruction: str, emoji_instruction: str, max_suggestions: int
) -> str:
    """User turn for sparse-chat opener suggestions."""
    return (
        f"User draft or goal: {draft.strip() or '[No draft provided]'}\n\n"
        f"Preferred tone: {tone}\n"
        f"{tone_instruction}\n"
        f"{emoji_instruction}\n\n"
        f"Generate exactly {max_suggestions} strong messaging suggestions for sparse-chat or first-message "
        "situations. "
        "Prioritize greetings, openers, light flirty starters, and easy-to-reply-to messages. "
        "Avoid bland lines like only 'hi' or 'hello' unless they are improved. "
        "If the draft is a simple message like hi, hello, what is your age, where are you from, or have you been "
        "there, turn it into natural reply options a real person would send. "
        "Do not invent actions, plans, bookings, or commitments unless the user explicitly asked for that level "
        "of action. "
        "Keep the suggestions grounded and low-assumption when the chat is sparse. "
        "Because context is weak here, apply a confidence guard: prefer lower-assumption replies over enthusiastic "
        "or committed ones. "
        "Make sure grouped_replies contains clearly different values for safe_reply, warm_reply, playful_reply, "
        "curious_reply, and direct_reply."
    )


def conversation_answer_user_prompt(
    older_text: str, other_name: str, conversation_text: str, history_text: str, question: str, tone: str
) -> str:
    """User turn for the ask-mode JSON answer about the conversation."""
    return (
        f"Older conversation summary or snippets:\n{older_text}\n\n"
        f"Conversation with {other_name}:\n{conversation_text}\n\n"
        f"Recent Ask AI follow-up context:\n{history_text}\n\n"
        f"User question: {question.strip() or 'Help me with this conversation.'}\n\n"
        "Answer specifically from the chat. If the answer is unclear, say that clearly. "
        "Never mention JSON, formatting, schemas, or internal system limitations to the user. "
        "Use the Ask AI follow-up context only to preserve continuity with the user's earlier assistant questions, "
        "but ground the answer in the real chat messages. "
        "If the user asks for the chat or conversation itself, provide a clean transcript-style recap or concise "
        "summary from the messages. "
        f"Keep the answer tone {tone}. Also suggest up to 3 helpful next actions."
    )


def general_help_user_prompt(question: str, tone: str) -> str:
    """User turn for the ask-mode JSON general messaging advice."""
    return (
        f"User question: {question.strip() or 'Help me write general messages.'}\n\n"
        f"Preferred tone: {tone}\n\n"
        "Give practical advice for texting or messaging in general. "
        "If the user asks for hello, hi, flirty, or opening-message help, provide 3 short ready-to-send examples. "
        "Never mention JSON, schemas, formatting, or internal model limitations."
    )


def stream_conversation_user_prompt(
    older_text: str, other_name: str, conversation_text: str, history_text: str, question: str, tone: str
) -> str:
    """User turn for the streaming answer about the conversation."""
    return (
        f"Older conversation summary or snippets:\n{older_text}\n\n"
        f"Conversation with {other_name}:\n{conversation_text}\n\n"
        f"Recent Ask AI follow-up context:\n{history_text}\n\n"
        f"User question: {question.strip() or 'Help me with this conversation.'}\n\n"
        f"Keep the answer tone {tone}. If context is unclear, say that clearly. "
        "Use the Ask AI follow-up context only to preserve continuity, but ground the answer in the real chat "
        "messages."
    )


def stream_general_help_user_prompt(question: str, tone: str) -> str:
    """User turn for the streaming general messaging advice."""
    return (
        f"User question: {question.strip() or 'Help me write general messages.'}\n\n"
        f"Preferred tone: {tone}\n\n"
        "If they ask for hi, hello, flirty, romantic, or opening-message help, include 3 short ready-to-send "
        "examples."
    )
