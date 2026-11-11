"""What each language option asks of the writer.

`hinglish` is not Hindi with English sprinkled on top, and it is not English with Hindi
words: it is how many Indian engineers actually talk. Hindi carries the sentence, English
carries the technical vocabulary. Written the way it is said, a speech engine with an
Indian voice reads it as one natural stream, which a translated brief never sounds like.
"""

from __future__ import annotations

HINGLISH = (
    "Write it in Hinglish, the way two Indian engineers talk: Hindi grammar and connecting "
    "words in Devanagari script, with every technical term, product name, model name, library "
    "name and number kept in English in Latin letters. For example: 'ये library अब streaming "
    "support करती है, मतलब latency काफ़ी कम हो जाएगी।' Not a translation: say it as a person "
    "would say it. About a third of the words may be English."
)

_INSTRUCTIONS = {
    "en": "",
    "hi": "Write in Hindi.",
    "hinglish": HINGLISH,
}


def writing_instruction(language: str) -> str:
    """The sentence added to a writing prompt; empty for English."""
    return _INSTRUCTIONS.get(language, "")


def is_indian(language: str) -> bool:
    return language in {"hi", "hinglish"}
