"""
Translation Service - Urdu/Roman Urdu ↔ English

Uses Groq LLM for translation. When user writes in Urdu/Roman Urdu:
  1. User message → English (for crisis/emotion/LLM)
  2. Bot replies in English
  3. Response → Urdu before sending to user

TO CUSTOMIZE URDU OUTPUT:
- Edit the Urdu translation prompt in _translate_with_llm() (search "Translate to natural Roman Urdu")
- Add or change the few-shot examples to match your preferred style
- Set URDU_OUTPUT_SCRIPT=urdu for Urdu script; default is Roman Urdu (Latin letters)
"""

import os
import re
import asyncio
import logging
import httpx
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TRANSLATION_TIMEOUT = httpx.Timeout(12.0, connect=8.0)

# Output style: "roman" = Roman Urdu (Latin script, like "main samajh sakta hoon") - often more natural for chat
# "urdu" = Urdu script (Arabic/Nastaliq). Set URDU_OUTPUT_SCRIPT=urdu for script.
URDU_OUTPUT_SCRIPT = os.getenv("URDU_OUTPUT_SCRIPT", "roman").lower()

# Unicode range for Urdu/Arabic script (Urdu uses Arabic script)
_URDU_SCRIPT_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')

# STRONG indicators - these words rarely appear in normal English (very likely Roman Urdu)
_ROMAN_URDU_STRONG = [
    r'\bmujhe\b', r'\btujhe\b', r'\bunhe\b', r'\bhumein\b', r'\btumhe\b',
    r'\budaas\b', r'\budass\b', r'\bpareshaan\b', r'\bparishan\b',
    r'\bhoon\b', r'\bhun\b', r'\bko\b', r'\bkya\b', r'\bkyun\b',
    r'\bkaise\b', r'\bkab\b', r'\bkahan\b', r'\bbahut\b', r'\bbohat\b',
    r'\bnahi\b', r'\bnahiin\b', r'\bnaheen\b', r'\btha\b', r'\bthi\b',
    r'\bchahta\b', r'\bchahti\b', r'\bchahte\b', r'\bsamajh\b', r'\blagta\b', r'\blagti\b',
    r'\bmera\b', r'\bmeri\b', r'\bmere\b', r'\btum\b', r'\baap\b', r'\bhum\b', r'\bham\b',
    r'\bthora\b', r'\bthori\b', r'\byeh\b', r'\bwoh\b', r'\bhain\b', r'\bhein\b',
    r'\bki\b', r'\bke\b', r'\bka\b',  # standalone ki/ke/ka (Urdu) - careful: "ki" can be English
]
# WEAK indicators - can appear in English too (main=primary, problem, tension)
_ROMAN_URDU_WEAK = [r'\bmain\b', r'\bmein\b', r'\bhai\b', r'\bho\b', r'\btheek\b', r'\bdil\b']

# English indicators - if text has these, it's likely English (not Roman Urdu)
_ENGLISH_INDICATORS = [
    r'\bthe\b', r'\band\b', r'\bbut\b', r'\bbecause\b', r'\bwith\b', r'\bthat\b',
    r'\bthis\b', r'\bwhat\b', r'\bwhen\b', r'\bwhere\b', r'\bhow\b', r'\bwhy\b',
    r'\bthink\b', r'\bthought\b', r'\bfeel\b', r'\bfeeling\b', r'\bfeelings\b',
    r'\bgoing\b', r'\bwant\b', r'\bneed\b', r'\btry\b', r'\btrying\b',
    r'\bsomething\b', r'\bnothing\b', r'\beverything\b', r'\breally\b',
    r"\bi'm\b", r"\bi've\b", r"\bi'll\b", r"\bdon't\b", r"\bcan't\b", r"\bwon't\b",
    r"\bit's\b", r"\bthat's\b", r"\bthere's\b", r"\bwhat's\b",
]

_STRONG_PATTERN = re.compile('|'.join(_ROMAN_URDU_STRONG), re.IGNORECASE)
_WEAK_PATTERN = re.compile('|'.join(_ROMAN_URDU_WEAK), re.IGNORECASE)
_ENGLISH_PATTERN = re.compile('|'.join(_ENGLISH_INDICATORS), re.IGNORECASE)


def _contains_urdu_script(text: str) -> bool:
    """Check if text contains Urdu/Arabic script characters."""
    return bool(_URDU_SCRIPT_PATTERN.search(text))


def _looks_like_roman_urdu(text: str) -> bool:
    """
    Distinguish Roman Urdu from English.
    - Urdu script = always Urdu
    - Strong Roman Urdu words (mujhe, udaas, hoon, etc.) = Roman Urdu
    - Many English words (the, feel, think, etc.) = English, skip translation
    - Weak indicators only (main, problem) with no strong = ambiguous, lean English
    """
    if _contains_urdu_script(text):
        return True

    text_lower = text.lower()
    strong_matches = len(_STRONG_PATTERN.findall(text_lower))
    weak_matches = len(_WEAK_PATTERN.findall(text_lower))
    english_matches = len(_ENGLISH_PATTERN.findall(text_lower))

    # Clear English: has typical English words
    if english_matches >= 2:
        return False
    if english_matches >= 1 and strong_matches == 0:
        return False

    # Clear Roman Urdu: has strong indicators (mujhe, udaas, hoon, etc.)
    if strong_matches >= 1:
        return True

    # Ambiguous: only weak indicators (main, hai, problem) - likely English
    # "I have a main problem" vs "main udaas hoon" - without strong words, assume English
    if weak_matches >= 2 and strong_matches == 0 and english_matches == 0:
        return True  # e.g. "main hai" - very short, could be Urdu

    return False


def _is_likely_urdu(text: str) -> bool:
    """Detect if the message is likely in Urdu or Roman Urdu (not English)."""
    text = text.strip()
    if len(text) < 3:
        return False
    return _contains_urdu_script(text) or _looks_like_roman_urdu(text)


async def _translate_with_llm(text: str, direction: str) -> Optional[str]:
    """
    Use Groq LLM for translation with context-aware prompts.
    direction: "to_english" or "to_urdu"
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set - translation unavailable")
        return None

    if direction == "to_english":
        system_prompt = """You are an expert Urdu-English translator for a mental health support chatbot. Translate accurately - lives may depend on understanding the user correctly.

Translate the user's message from Urdu (Arabic script) or Roman Urdu (Urdu in Latin letters) to clear English.

ROMAN URDU: Has no standard spelling. Interpret these common variations correctly:
- main/mein/maen = I | hoon/hun/hu = am | hai/he/hay = is/are
- udaas/udass/udaas = sad | pareshaan/parishan = anxious/worried
- tension/tension = stress | dil = heart/feelings | bura = bad
- mujhe/mujhse = to me | kya = what | kyun = why | kaise = how
- bahut/bohat/zyada = very/much | nahi/nahiin = not | theek = okay/fine

RULES:
1. Preserve the EXACT emotional meaning. "Main bahut udaas hoon" = "I am very sad". Mental health context is critical.
2. Output ONLY the English translation. No prefix, no quotes, no explanation.
3. If already English, return unchanged.
4. Use natural, conversational English."""

        user_content = f"Translate to English:\n\n{text.strip()}"
    else:
        # Roman Urdu = natural chat style (main samajh sakta hoon). Urdu script = formal written style.
        use_roman = URDU_OUTPUT_SCRIPT != "urdu"

        if use_roman:
            system_prompt = """Translate English to natural Roman Urdu for a mental health chat app. Roman Urdu = Urdu in Latin letters, like how people text.

COPY THESE EXACT STYLES (how people actually talk):

Example 1:
English: I hear you. That sounds really difficult.
Roman Urdu: Main sun raha hoon. Yeh sun kar bura laga, yeh mushkil lag raha hai.

Example 2:
English: It makes sense that you'd feel that way.
Roman Urdu: Yeh samajh aata hai ke aap aisa feel kar rahe hain.

Example 3:
English: I'm here to listen whenever you want to talk.
Roman Urdu: Main yahan hoon, jab bhi aap baat karna chahein sunne ke liye.

Example 4:
English: Take your time. There's no rush.
Roman Urdu: Apna time lo. Koi jaldi nahi hai.

Example 5:
English: What you're going through sounds exhausting.
Roman Urdu: Jo aap face kar rahe hain, sun kar lagta hai bahut thakaan ho rahi hogi.

RULES:
- Write in Roman Urdu (Latin letters). Use "aap" for you, "main" for I.
- Match the EXACT meaning. Keep the warm, supportive tone.
- Use simple words. "feel" = feel (or mehsoos), "understand" = samajh, "listen" = sunna.
- Natural flow - like a friend texting back. NOT stiff or formal.
- Output ONLY the translation. No quotes, no "Translation:"."""

        else:
            system_prompt = """Translate English to simple Urdu (Urdu script) for a mental health chat app. Use everyday, natural Urdu.

COPY THESE EXACT STYLES:

Example 1:
English: I hear you. That sounds really difficult.
Urdu: میں سن رہا ہوں۔ یہ سن کر برا لگا، یہ مشکل لگ رہا ہے۔

Example 2:
English: It makes sense that you'd feel that way.
Urdu: یہ سمجھ آتا ہے کہ آپ ایسا محسوس کر رہے ہیں۔

Example 3:
English: I'm here to listen whenever you want to talk.
Urdu: میں یہاں ہوں، جب بھی آپ بات کرنا چاہیں سننے کے لیے۔

Example 4:
English: Take your time. There's no rush.
Urdu: اپنا وقت لو۔ کوئی جلدی نہیں ہے۔

RULES:
- Use Urdu script (right-to-left). Use آپ for you.
- Match the EXACT meaning. Warm, supportive tone.
- Simple words. Short sentences. Like a caring friend.
- Output ONLY the translation. No quotes, no prefix."""

        user_content = f"Translate to {'Roman Urdu' if use_roman else 'Urdu'}:\n\n{text.strip()}"

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2,  # Low temperature for consistent, accurate translation
        "max_tokens": 300,
    }

    try:
        async with httpx.AsyncClient(timeout=TRANSLATION_TIMEOUT) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    # Clean up: remove common LLM prefixes
                    translated = content.strip()
                    for prefix in ["Translation:", "Here is the translation:", "The translation is:"]:
                        if translated.lower().startswith(prefix.lower()):
                            translated = translated[len(prefix):].strip()
                    if translated.startswith('"') and translated.endswith('"'):
                        translated = translated[1:-1]
                    return translated.strip()
    except Exception as e:
        logger.warning(f"LLM translation failed ({direction}): {e}")
    return None


async def translate_to_english_async(text: str) -> Tuple[str, bool]:
    """
    Translate text from Urdu/Roman Urdu to English using LLM.
    
    Returns:
        Tuple of (translated_text, was_translated)
    """
    if not text or not text.strip():
        return (text, False)
    
    if not _is_likely_urdu(text):
        return (text, False)
    
    translated = await _translate_with_llm(text, "to_english")
    if translated and translated.strip():
        logger.debug(f"Translated to English")
        return (translated.strip(), True)
    
    logger.warning("Translation to English failed, using original")
    return (text, False)


async def translate_to_urdu_async(text: str) -> Tuple[str, bool]:
    """
    Translate text from English to Urdu using LLM.
    
    Returns:
        Tuple of (translated_text, was_translated)
    """
    if not text or not text.strip():
        return (text, False)
    
    translated = await _translate_with_llm(text, "to_urdu")
    if translated and translated.strip():
        logger.debug(f"Translated to Urdu")
        return (translated.strip(), True)
    
    logger.warning("Translation to Urdu failed, using original English")
    return (text, False)


def translate_to_english(text: str) -> Tuple[str, bool]:
    """Sync wrapper for translate_to_english_async."""
    return asyncio.run(translate_to_english_async(text))


def translate_to_urdu(text: str) -> Tuple[str, bool]:
    """Sync wrapper for translate_to_urdu_async."""
    return asyncio.run(translate_to_urdu_async(text))


async def process_user_message_for_pipeline_async(user_message: str) -> Tuple[str, bool]:
    """
    Prepare user message for the chatbot pipeline (async).
    If the message is in Urdu/Roman Urdu, translate to English using LLM.
    """
    return await translate_to_english_async(user_message)


async def process_bot_response_for_user_async(bot_response: str, user_wrote_urdu: bool) -> str:
    """
    Prepare bot response for the user (async).
    If the user wrote in Urdu, translate the response to Urdu using LLM.
    """
    if not user_wrote_urdu:
        return bot_response
    translated, _ = await translate_to_urdu_async(bot_response)
    return translated


# Sync wrappers for backward compatibility (used if caller can't await)
def process_user_message_for_pipeline(user_message: str) -> Tuple[str, bool]:
    """Sync wrapper - runs async in new event loop."""
    return asyncio.run(process_user_message_for_pipeline_async(user_message))


def process_bot_response_for_user(bot_response: str, user_wrote_urdu: bool) -> str:
    """Sync wrapper - runs async in new event loop."""
    return asyncio.run(process_bot_response_for_user_async(bot_response, user_wrote_urdu))
