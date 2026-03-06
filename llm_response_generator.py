"""
LLM Response Generator

This module generates empathetic responses using Groq API.
It integrates with emotion detection to create contextually appropriate responses.
"""

import os
import httpx
import asyncio
import json
from typing import Optional, List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# Request timeout settings
REQUEST_TIMEOUT = httpx.Timeout(15.0, connect=10.0)  # 15s total, 10s connection timeout

# Fallback response if LLM fails
FALLBACK_RESPONSE = "I'm here to listen. What's on your mind?"


class LLMResponseGenerator:
    """
    Generates empathetic responses using Groq API.
    """
    
    def __init__(self, api_key: Optional[str] = None, prompt_template_path: Optional[str] = None):
        """
        Initialize the LLM response generator.
        
        Args:
            api_key: Groq API key. If None, uses GROQ_API_KEY env variable.
            prompt_template_path: Optional path to custom prompt template file.
        """
        self.api_key = api_key or GROQ_API_KEY
        self.api_url = GROQ_API_URL
        
        # Load prompt template if provided
        self.prompt_template = None
        if prompt_template_path:
            self.prompt_template = self._load_prompt_template(prompt_template_path)
        else:
            # Try to load default prompt template
            default_prompt_path = Path(__file__).parent / "prompts" / "llm_prompt_template.txt"
            if default_prompt_path.exists():
                self.prompt_template = self._load_prompt_template(str(default_prompt_path))
        
        if not self.api_key:
            logger.warning(
                "No Groq API key provided. "
                "Set GROQ_API_KEY environment variable."
            )
    
    def _load_prompt_template(self, template_path: str) -> Optional[str]:
        """
        Load prompt template from file.
        
        Args:
            template_path: Path to template file
            
        Returns:
            Template string, or None if loading fails
        """
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read().strip()
                logger.info(f"Loaded prompt template from {template_path}")
                return template
        except Exception as e:
            logger.warning(f"Could not load prompt template from {template_path}: {e}")
            return None
    
    async def generate_response_async(
        self,
        user_message: str,
        detected_emotion: Optional[str] = None,
        emotion_confidence: float = 0.0,
        prompt_template: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        respond_in_urdu: bool = False,
        original_urdu_message: Optional[str] = None,
        crisis_level: Optional[str] = None
    ) -> str:
        """
        Async version of generate_response for better timeout handling.
        
        SINGLE-CALL BILINGUAL GENERATION:
        - Generates response directly in user's language (Urdu or English) in one API call
        - Uses full conversation history in the same language for context
        - No translation/rewrite step needed
        
        Args:
            user_message: User's message in their original language (Urdu or English)
            respond_in_urdu: If True, generate response directly in Urdu script (proper Urdu, not Roman Urdu)
            original_urdu_message: Original Urdu message (same as user_message if user wrote Urdu)
            conversation_history: Full conversation history in the same language as user_message
        """
        try:
            # Reload prompt template from file on each request (allows hot-reloading)
            # This allows prompt changes without server restart
            if prompt_template is None:
                # Try to reload from default path
                default_prompt_path = Path(__file__).parent / "prompts" / "llm_prompt_template.txt"
                if default_prompt_path.exists():
                    try:
                        reloaded_template = self._load_prompt_template(str(default_prompt_path))
                        if reloaded_template:
                            self.prompt_template = reloaded_template
                    except Exception as e:
                        logger.debug(f"Could not reload prompt template: {e}, using cached version")
            
            # Build the base system prompt
            template_to_use = prompt_template or self.prompt_template
            if template_to_use:
                try:
                    # Format the template with variables
                    system_prompt = template_to_use.format(
                        user_message="",
                        emotion=detected_emotion or "not specified",
                        emotion_confidence=emotion_confidence
                    ).split("User message:")[0].strip()
                except KeyError as e:
                    logger.warning(f"Prompt template missing key {e}, using default prompt")
                    system_prompt = self._build_default_system_prompt()
            else:
                system_prompt = self._build_default_system_prompt()
            
            # Note: The conversation history is pre-seeded with the frontend greeting
            # so the LLM sees "assistant: Hi [name], how is your mood today?" already exists.
            # This naturally prevents the LLM from generating another greeting.
            
            # Add Urdu-specific instructions if responding in Urdu
            if respond_in_urdu:
                urdu_instructions = """

CRITICAL: You MUST respond ONLY in proper Urdu script (not Roman Urdu).

URDU LANGUAGE REQUIREMENTS:
- Respond in grammatically correct, natural Pakistani Urdu script
- Always use "آپ" form (respectful you)
- Avoid gender assumptions
- User may have written in Roman Urdu with incorrect spelling - understand it and respond in proper Urdu script
- Do NOT copy the user's grammar or sentence structure
- Do NOT repeat their sentence structure
- Use correct grammar and natural flow

URDU STYLE REQUIREMENTS:
- Avoid heavy psychological jargon
- Avoid very clinical tone
- Avoid over-therapeutic tone
- Avoid sounding like a textbook
- Sound like a calm, emotionally intelligent Pakistani adult
- Feel relational and warm
- Use simple, natural spoken Urdu
- Keep sentences short

EMOTIONAL GUARDRAILS (When User Expresses Distress):
1. Always validate first
2. Do not challenge the user
3. Do not question the legitimacy of their feeling
4. Avoid abstract thinking prompts
5. Use short, emotionally safe sentences

RESPONSE STRUCTURE (When User Expresses Distress):
- One validation sentence (acknowledge their feeling)
- One empathy/normalization sentence (normalize the experience)
- One gentle open-ended question (simple, clear, emotionally grounded)

Do NOT ask abstract or philosophical questions.
Do NOT ask confusing meta-questions.
Ask only simple, clear, emotionally grounded questions.

EXAMPLES OF GOOD vs BAD:

❌ BAD (mirroring user's grammar):
User: mai pareshan hu
Bot: aap pareshan hain

✅ GOOD (natural, grammatically correct):
User: mai pareshan hu
Bot: مجھے افسوس ہے کہ آپ کافی دنوں سے پریشان ہیں۔ یہ واقعی آسان نہیں ہوتا جب دل اور دماغ دونوں بوجھ محسوس کریں۔ اگر آپ چاہیں تو مجھے بتا سکتے ہیں کہ یہ پریشانی کس وجہ سے شروع ہوئی؟

❌ BAD (broken grammar, robotic):
User: yahi baat mujhao overthink karwa rahi
Bot: apki feeling ko samajhnai k liyai mai yahan hoo...

✅ GOOD (natural, warm):
User: yahi baat mujhao overthink karwa rahi
Bot: لگتا ہے یہ بات آپ کے ذہن میں بار بار آ رہی ہے اور آپ کو بہت تھکا رہی ہے۔ اوور تھنک کرنا واقعی ذہنی دباؤ بڑھا دیتا ہے۔ کیا آپ بتانا چاہیں گے کہ وہ کون سی بات ہے جو آپ کو سب سے زیادہ پریشان کر رہی ہے؟

OUTPUT REQUIREMENTS:
- Output ONLY the Urdu response in proper Urdu script
- NO notes, NO explanations, NO meta-commentary
- NO Roman Urdu - use proper Urdu script only
- Never sound robotic
- Never repeat user sentence structure
- Always use correct grammar
- Do NOT cut off sentences - complete all thoughts fully
- Ensure complete, grammatically correct sentences
"""
                system_prompt = system_prompt + urdu_instructions
            
            # SINGLE-CALL GENERATION: Generate directly in target language (Urdu or English)
            # Use original user message (not translated) and full conversation history in same language
            logger.info(f"Generating response directly in {'Urdu' if respond_in_urdu else 'English'} (single-call bilingual generation)")
            response_text = await self._call_groq_api_async(
                system_prompt, 
                user_message,  # Use original message (Urdu if user wrote Urdu, English if English)
                detected_emotion, 
                emotion_confidence,
                conversation_history,  # History in same language as user_message
                respond_in_urdu=respond_in_urdu
            )
            
            if response_text:
                return response_text.strip()
            else:
                logger.warning("LLM returned empty response, using fallback")
                # Return Urdu script fallback if responding in Urdu
                if respond_in_urdu:
                    return "میں یہاں ہوں، سننے کے لیے۔ آپ کیا کہنا چاہتے ہیں؟"
                return FALLBACK_RESPONSE
                
        except asyncio.TimeoutError:
            logger.error("LLM request timed out")
            if respond_in_urdu:
                return "میں یہاں ہوں، سننے کے لیے۔ آپ کیا کہنا چاہتے ہیں؟"
            return FALLBACK_RESPONSE
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            if respond_in_urdu:
                return "میں یہاں ہوں، سننے کے لیے۔ آپ کیا کہنا چاہتے ہیں؟"
            return FALLBACK_RESPONSE
    
    def generate_response(
        self,
        user_message: str,
        detected_emotion: Optional[str] = None,
        emotion_confidence: float = 0.0,
        prompt_template: Optional[str] = None,
        respond_in_urdu: bool = False,
        original_urdu_message: Optional[str] = None,
        crisis_level: Optional[str] = None
    ) -> str:
        """
        Generate an empathetic response using LLM.
        
        Args:
            user_message: The user's input message
            detected_emotion: Detected emotion (if available)
            emotion_confidence: Confidence score for detected emotion (0.0 to 1.0)
            prompt_template: Optional custom prompt template. If None, uses default.
            respond_in_urdu: If True, generate response in Roman Urdu instead of English.
            original_urdu_message: Original Urdu message if user wrote in Urdu (for context)
            crisis_level: Crisis level ("none", "low", "medium", "high", "critical")
            
        Returns:
            Generated response text, or fallback response if generation fails
        """
        # Use async version
        return asyncio.run(self.generate_response_async(
            user_message, detected_emotion, emotion_confidence, prompt_template, None, 
            respond_in_urdu, original_urdu_message, crisis_level
        ))
    
    def _build_default_system_prompt(self) -> str:
        """
        Build the default system prompt for LLM.
        
        Returns:
            System prompt string
        """
        return (
            "You are an empathetic mental health support assistant. "
            "Your role is to provide active listening and emotional support through conversation. "
            "Be warm, understanding, and non-judgmental. "
            "Acknowledge and validate the user's feelings. "
            "Reflect or paraphrase their emotional experience naturally. "
            "Use supportive language that feels human and compassionate. "
            "Do not give medical advice, diagnoses, or solutions. "
            "Do not minimize their experience (avoid phrases like 'everything will be okay'). "
            "Keep responses concise (2-3 sentences). "
            "Sound like a compassionate human listener, not a therapist or clinical professional."
        )
    
    async def _call_groq_api_async(
        self,
        system_prompt: str,
        user_message: str,
        detected_emotion: Optional[str],
        emotion_confidence: float,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        respond_in_urdu: bool = False
    ) -> Optional[str]:
        """
        Call Groq API to generate response.
        
        Args:
            system_prompt: System prompt with instructions
            user_message: User's message
            detected_emotion: Detected emotion (if available)
            emotion_confidence: Emotion confidence score
            
        Returns:
            Generated response text, or None if request fails
        """
        # Check API key at runtime (in case it was set after server startup)
        api_key = self.api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("Groq API key not set. Set GROQ_API_KEY environment variable and restart the server.")
            logger.error("Current api_key value: None")
            return None
        
        logger.debug(f"Calling Groq API: llama-3.1-8b-instant")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        # Build user message with emotion context if available
        user_content = user_message
        if detected_emotion and emotion_confidence > 0.3:
            # Add emotion context for both English and Urdu responses
            user_content += f"\n\n[Detected emotion: {detected_emotion} (confidence: {emotion_confidence:.2f})]"
        
        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        # History is pre-seeded with the frontend greeting ("Hi [name], how is your mood today?")
        # so the LLM naturally sees the greeting already happened and won't repeat it
        if conversation_history:
            messages.extend(conversation_history)
            logger.debug(f"Including {len(conversation_history)} history messages in context")
        
        # Add current user message
        messages.append({"role": "user", "content": user_content})
        logger.debug(f"Sending {len(messages)} messages to LLM, user message: {user_content[:100]}")
        
        # Adjust parameters based on language
        if respond_in_urdu:
            temperature = 0.4  # Lower temperature (0.4-0.5) for consistent Urdu generation
            max_tokens = 450  # Increased to 400-500 for Urdu script (using 450 as middle ground)
        else:
            temperature = 0.7
            max_tokens = 150
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9
        }
        
        try:
            # Use ensure_ascii=False to preserve Urdu script and other Unicode characters
            # httpx.post(json=...) uses json.dumps internally with ensure_ascii=True by default
            # So we need to manually serialize with ensure_ascii=False
            payload_json = json.dumps(payload, ensure_ascii=False)
            
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    content=payload_json.encode('utf-8')  # Send as UTF-8 encoded content
                )
                
                if response.status_code == 200:
                    result = response.json()
                    choices = result.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        content = message.get("content", "")
                        if content:
                            # Clean up response - remove any notes, explanations, or meta-commentary
                            cleaned = content.strip()
                            
                            # Remove common prefixes/notes (both English and Urdu)
                            prefixes_to_remove = [
                                "Translation:",
                                "Note:",
                                "Response:",
                                "Urdu Response:",
                                "Here's the response:",
                                "Detected emotion:",
                                "I didn't use the detected emotion",
                                "Note that",
                                "Please note",
                                "ترجمہ:",
                                "نوٹ:",
                                "جواب:",
                                "یہ جواب ہے:"
                            ]
                            
                            for prefix in prefixes_to_remove:
                                if cleaned.lower().startswith(prefix.lower()):
                                    cleaned = cleaned[len(prefix):].strip()
                                # Also check if it starts with prefix (case-insensitive for English, exact for Urdu)
                                if cleaned.startswith(prefix):
                                    cleaned = cleaned[len(prefix):].strip()
                            
                            # Remove content after common separators that indicate notes
                            separators = ["\n\nNote:", "\nNote:", "\n---", "\n[Note", "\n(Note", "\n\nنوٹ:", "\nنوٹ:"]
                            for sep in separators:
                                if sep in cleaned:
                                    cleaned = cleaned.split(sep)[0].strip()
                            
                            # Remove quotes if entire response is quoted
                            if cleaned.startswith('"') and cleaned.endswith('"'):
                                cleaned = cleaned[1:-1].strip()
                            if cleaned.startswith("'") and cleaned.endswith("'"):
                                cleaned = cleaned[1:-1].strip()
                            
                            return cleaned
                        return None
                    else:
                        logger.warning("No choices in Groq API response")
                        return None
                else:
                    logger.error(f"Groq API error: {response.status_code} - {response.text}")
                    return None
                    
        except httpx.TimeoutException:
            logger.error("Groq API request timed out")
            return None
        except httpx.RequestError as e:
            logger.error(f"Groq API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Groq API: {e}")
            return None


def get_llm_response_generator(api_key: Optional[str] = None) -> LLMResponseGenerator:
    """
    Get or create a global LLM response generator instance.
    
    Args:
        api_key: Optional API key. If None, uses environment variable.
        
    Returns:
        LLMResponseGenerator instance
    """
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = LLMResponseGenerator(api_key=api_key)
    return _llm_generator


# Global instance (initialized on first use)
_llm_generator: Optional[LLMResponseGenerator] = None

