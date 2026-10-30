"""User-facing messages for help-chat failures, plus a breaker that skips the LLM while a limit is active.

Messages are templates (no LLM call), so they work even when every model is out of quota.
"""
import math
import re
import threading
import time

import groq

DURATION_RE = re.compile(r"try again in ((?:\d+h)?(?:\d+m)?(?:\d+(?:\.\d+)?s)?)")
DURATION_PART_RE = re.compile(r"(\d+(?:\.\d+)?)([hms])")
SCOPE_RE = re.compile(r"per (day|minute) \((?:TPD|TPM|RPD|RPM)\)")
MODEL_RE = re.compile(r"for model `([^`]+)`")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
# Common Hindi words written in English letters; two or more in a message means Hinglish
HINGLISH_WORDS = {
    "kaise", "kaisa", "kya", "kyu", "kyun", "karu", "karun", "karo", "karna", "karte", "karein", "kare", "mera", "meri",
    "mere", "mujhe", "apna", "apni", "apne", "aap", "hai", "hain", "ho", "nahi", "nahin", "kahan", "kidhar", "yaha",
    "yahan", "gaya", "gayi", "raha", "rahi", "chahiye", "bhai", "batao", "bata", "kar", "se", "ka", "ki", "ke", "ko",
}

# kind -> language -> template. {wait} is a humanized duration such as "about 1 hour 20 minutes".
MESSAGES = {
    # Groq's daily cap is a rolling 24h window, so the wait can be minutes or hours: say "for now", not "today"
    "daily_limit": {
        "english": "⏳ CliQ AI has reached its usage limit for now, so I can't answer at the moment. Please come back in {wait}.",
        "hinglish": "⏳ CliQ AI ki usage limit abhi ke liye khatam ho gayi hai, isliye abhi jawab nahi de sakta. Please {wait} baad wapas aana.",
        "hindi": "⏳ CliQ AI की उपयोग सीमा अभी के लिए पूरी हो गई है, इसलिए अभी जवाब नहीं दे सकता। कृपया {wait} बाद फिर आएँ।",
    },
    "busy": {
        "english": "CliQ AI is getting a lot of questions right now. Please try again in {wait}.",
        "hinglish": "CliQ AI ke paas abhi bahut saare sawaal aa rahe hain. Please {wait} baad dobara try karo.",
        "hindi": "CliQ AI के पास अभी बहुत सारे सवाल आ रहे हैं। कृपया {wait} बाद फिर कोशिश करें।",
    },
    "setup": {
        "english": "CliQ AI is unavailable because of a configuration problem on our side. The CliQ team needs to fix it, so please try again later.",
        "hinglish": "Humari taraf ek setup problem ki wajah se CliQ AI abhi available nahi hai. CliQ team ise fix karegi, please baad mein try karo.",
        "hindi": "हमारी तरफ़ एक सेटअप समस्या की वजह से CliQ AI अभी उपलब्ध नहीं है। CliQ टीम इसे ठीक करेगी, कृपया बाद में कोशिश करें।",
    },
    "timeout": {
        "english": "CliQ AI took too long to respond. Please try again in a moment.",
        "hinglish": "CliQ AI ne jawab dene mein bahut der laga di. Please thodi der baad dobara try karo.",
        "hindi": "CliQ AI ने जवाब देने में बहुत देर लगा दी। कृपया थोड़ी देर बाद फिर कोशिश करें।",
    },
    "unreachable": {
        "english": "CliQ AI can't reach its AI service right now. Please try again in a few minutes.",
        "hinglish": "CliQ AI abhi apni AI service tak nahi pahunch pa raha. Please kuch minute baad try karo.",
        "hindi": "CliQ AI अभी अपनी AI सेवा तक नहीं पहुँच पा रहा। कृपया कुछ मिनट बाद कोशिश करें।",
    },
    "provider_down": {
        "english": "CliQ AI's AI service is having problems right now. Please try again in a few minutes.",
        "hinglish": "CliQ AI ki AI service mein abhi problem chal rahi hai. Please kuch minute baad try karo.",
        "hindi": "CliQ AI की AI सेवा में अभी समस्या है। कृपया कुछ मिनट बाद कोशिश करें।",
    },
    "bad_request": {
        "english": "I couldn't process that question. Please try rephrasing it.",
        "hinglish": "Main ye sawaal process nahi kar paaya. Please ise thoda alag tareeke se poochho.",
        "hindi": "मैं यह सवाल समझ नहीं पाया। कृपया इसे दूसरे शब्दों में पूछें।",
    },
    "internal": {
        "english": "Sorry, something went wrong on my side. Please try asking that again in a moment.",
        "hinglish": "Sorry, meri taraf kuch gadbad ho gayi. Please thodi der baad dobara poochho.",
        "hindi": "माफ़ कीजिए, मेरी तरफ़ कुछ गड़बड़ हो गई। कृपया थोड़ी देर बाद फिर पूछें।",
    },
}

UNKNOWN_WAIT = {"english": "a few hours", "hinglish": "kuch ghante", "hindi": "कुछ घंटे"}
BUSY_DEFAULT_SECONDS = 30
DAILY_DEFAULT_SECONDS = 60 * 60


def parse_wait_seconds(text):
    """'try again in 1h20m16.368s' -> 4816.4; None when the provider gave no time."""
    match = DURATION_RE.search(text or "")
    if not match or not match.group(1):
        return None
    factors = {"h": 3600, "m": 60, "s": 1}
    return sum(float(n) * factors[u] for n, u in DURATION_PART_RE.findall(match.group(1)))


def humanize_wait(seconds, language="english"):
    """Rounds up so users are never told to come back too early."""
    if seconds is None:
        return UNKNOWN_WAIT.get(language, UNKNOWN_WAIT["english"])
    seconds = max(1, math.ceil(seconds))
    hours, minutes = divmod(math.ceil(seconds / 60), 60)
    if seconds < 60:
        secs = max(10, math.ceil(seconds / 10) * 10)
        return {"english": f"about {secs} seconds", "hinglish": f"lagbhag {secs} second", "hindi": f"लगभग {secs} सेकंड"}.get(
            language, f"about {secs} seconds")
    if language == "hindi":
        parts = ([f"{hours} घंटे"] if hours else []) + ([f"{minutes} मिनट"] if minutes else [])
        return "लगभग " + " ".join(parts)
    if language == "hinglish":
        parts = ([f"{hours} ghante"] if hours else []) + ([f"{minutes} minute"] if minutes else [])
        return "lagbhag " + " ".join(parts)
    parts = ([f"{hours} hour{'s' if hours != 1 else ''}"] if hours else []) + (
        [f"{minutes} minute{'s' if minutes != 1 else ''}"] if minutes else [])
    return "about " + " ".join(parts)


def classify_error(exc):
    """Returns {"kind", "model", "retry_after"} for a provider or pipeline exception."""
    text = str(exc)
    model_match = MODEL_RE.search(text)
    info = {"kind": "internal", "model": model_match.group(1) if model_match else None, "retry_after": None}

    if isinstance(exc, groq.RateLimitError):
        scope = SCOPE_RE.search(text)
        daily = bool(scope and scope.group(1) == "day")
        info["kind"] = "daily_limit" if daily else "busy"
        info["retry_after"] = parse_wait_seconds(text) or (DAILY_DEFAULT_SECONDS if daily else BUSY_DEFAULT_SECONDS)
    elif isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError, groq.NotFoundError)):
        info["kind"] = "setup"
    elif isinstance(exc, groq.APITimeoutError):
        info["kind"] = "timeout"
    elif isinstance(exc, groq.APIConnectionError):
        info["kind"] = "unreachable"
    elif isinstance(exc, groq.InternalServerError):
        info["kind"] = "provider_down"
    elif isinstance(exc, groq.BadRequestError):
        info["kind"] = "bad_request"
    return info


def resolve_language(language, message):
    """Uses the router's language when known; otherwise guesses from the text (no LLM available)."""
    if language in ("english", "hinglish", "hindi"):
        return language
    if DEVANAGARI_RE.search(message or ""):
        return "hindi"
    words = set(re.findall(r"[a-z]+", (message or "").lower()))
    return "hinglish" if len(words & HINGLISH_WORDS) >= 2 else "english"


def user_message(kind, language="english", retry_after=None):
    templates = MESSAGES.get(kind, MESSAGES["internal"])
    template = templates.get(language, templates["english"])
    return template.format(wait=humanize_wait(retry_after, language))


class LimitBreaker:
    """Remembers that a model hit a rate limit until Groq says it resets.

    While open, questions are answered immediately with the limit message instead of
    calling the model (each blocked call would otherwise wait on retries and fail anyway).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._open = {}  # model -> (kind, reopen_at monotonic)

    def trip(self, model, kind, retry_after):
        if not model or kind not in ("daily_limit", "busy") or not retry_after:
            return
        with self._lock:
            self._open[model] = (kind, time.monotonic() + retry_after)

    def check(self, model):
        """Returns (kind, seconds_left) while the model is blocked, else None."""
        with self._lock:
            entry = self._open.get(model)
            if not entry:
                return None
            kind, reopen_at = entry
            left = reopen_at - time.monotonic()
            if left <= 0:
                del self._open[model]
                return None
            return kind, left
