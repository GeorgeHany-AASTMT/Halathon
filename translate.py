"""
translate.py — lightweight language detection + translation
using the same google-genai SDK as generate.py.
"""
import re
import json
import config

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

_client = genai.Client(api_key=config.GEMINI_API_KEY) if genai else None


def _clean(raw: str) -> str:
    """Strip markdown code fences the model sometimes adds despite instructions."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def detect_and_translate_to_english(text: str) -> tuple[str, str]:
    """Returns (english_text, detected_language_code)."""
    if _client is None:
        return text, "en"

    prompt = f"""You are a precise medical translation utility.
Detect the language of the text below and translate it to English.

IMPORTANT: Use standard, common medical English terminology (e.g. translate
"ارتفاع ضغط الدم" as "hypertension" or "high blood pressure", not an
overly literal or unusual phrasing). Preserve medical meaning exactly.
Keep the translation as a natural, direct clinical question — do not add
explanations, disclaimers, or extra words.

Respond ONLY in this exact format, nothing else, no markdown, no code fences:
LANG: <ISO 639-1 code, e.g. en, ar, fr>
TEXT: <English translation>

Text:
\"\"\"{text}\"\"\""""

    response = _client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    raw = _clean(response.text)

    lang_match = re.search(r"LANG:\s*([a-zA-Z-]+)", raw)
    text_match = re.search(r"TEXT:\s*(.+)", raw, re.DOTALL)

    lang = lang_match.group(1).strip().lower() if lang_match else "en"
    english_text = text_match.group(1).strip() if text_match else text

    print(f"[translate] detected lang={lang!r} | english_query={english_text!r}")  # debug

    return english_text, lang


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate English text back into target_lang (e.g. 'ar')."""
    if target_lang == "en" or _client is None:
        return text

    prompt = f"""Translate the following English text into language code "{target_lang}".
Preserve tone, medical accuracy, and formatting (markdown, lists, etc).
Respond ONLY with the translated text, nothing else, no markdown code fences.

Text:
\"\"\"{text}\"\"\""""

    response = _client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    translated = _clean(response.text)

    print(f"[translate] translated back to {target_lang!r}: {translated!r}")  # debug

    return translated


def is_arabic(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))