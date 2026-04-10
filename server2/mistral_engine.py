"""
StatsAI v3.0 Intelligence Core: Mistral Engine
Supports: Reasoning, OCR, Vision, and Audio.
Design: Zero-Hallucination Anchor & Multimodal Processing.
"""

import os, json, base64, logging
from pathlib import Path
from dotenv import load_dotenv
from mistralai.client import Mistral  # Confirmed Speakeasy structure for your env

# ── LOGGING ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(message)s')
logger = logging.getLogger("MistralEngine")

class MistralEngine:
    def __init__(self, env_path: str = None):
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv(Path(__file__).parent / ".env")
        
        self.api_key = os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment.")
        
        self.client = Mistral(api_key=self.api_key)
        self.default_model = "mistral-medium-latest"
        self.reasoning_model = "mistral-small-latest"
        self.ocr_model = "mistral-ocr-latest"

    def _get_hallucination_guard(self):
        """Standard System Prompt to minimize hallucinations."""
        return (
            "You are StatsAI, an expert statistical analyst. "
            "STRICT RULES TO PREVENT HALLUCINATION:\n"
            "1. Only use the data provided in the conversation history or uploaded images.\n"
            "2. If you are unsure or the data is missing, state 'DATA NOT AVAILABLE' clearly.\n"
            "3. Do not invent statistical results or citations.\n"
            "4. For complex reasoning, provide step-by-step logical validation.\n"
            "5. Always prioritize factuality over politeness."
        )

    async def chat(self, 
                   message: str, 
                   history: list = None, 
                   image_base64: str = None, 
                   reasoning: bool = False):
        """
        Unified Chat Interface supporting Text, Vision, and Reasoning.
        """
        messages = [{"role": "system", "content": self._get_hallucination_guard()}]
        
        # 1. Add History
        if history:
            for h in history[-10:]: # 10 message Context Window
                messages.append({"role": h['role'], "content": h['text']})
        
        # 2. Construct Current Message
        content = []
        if message:
            content.append({"type": "text", "text": message})
        
        if image_base64:
            # Mistral Vision format
            content.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image_base64}"
            })
        
        messages.append({"role": "user", "content": content})

        # 3. Parameters Selection
        model = self.reasoning_model if reasoning else self.default_model
        params = {
            "model": model,
            "messages": messages,
            "temperature": 0.0 if reasoning else 0.7,
        }

        if reasoning:
            # For mistral-small-latest, use reasoning_effort
            # For magistral models, reasoning is native/automatic
            if "magistral" in model:
                pass 
            else:
                params["reasoning_effort"] = "high"
                # params["prompt_mode"] = "reasoning" # Disabling as it caused 400 error

        try:
            logger.info(f"Hitting Mistral API ({model}, Reasoning={reasoning}, Visual={image_base64 is not None})")
            response = self.client.chat.complete(**params)
            return {
                "success": True,
                "reply": response.choices[0].message.content,
                "model": model
            }
        except Exception as e:
            logger.error(f"Mistral API Error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def extract_data_ocr(self, document_url: str = None, image_base64: str = None):
        """
        OCR capabilities to extract structured data from documents/images.
        """
        try:
            doc_obj = {}
            if document_url:
                doc_obj = {"type": "document_url", "document_url": document_url}
            elif image_base64:
                doc_obj = {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            
            # Using the new OCR method from the Spec
            result = self.client.ocr.process(
                model=self.ocr_model,
                document=doc_obj
            )
            
            # Concatenate all pages into markdown
            full_markdown = "\n\n".join([p.markdown for p in result.pages])
            return {"success": True, "markdown": full_markdown}
        except Exception as e:
            logger.error(f"OCR Error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def generate_audio(self, text: str, voice_id: str = "george"):
        """
        Generate audio summary (Speech API).
        """
        try:
            response = self.client.audio.speech(
                input=text,
                model="mistral-speech-latest",
                voice_id=voice_id,
                response_format="mp3"
            )
            # Binary audio data usually comes back in 'audio_data' as base64 in Spec
            return {"success": True, "audio_base64": response.audio_data}
        except Exception as e:
            logger.error(f"Speech Error: {str(e)}")
            return {"success": False, "error": str(e)}

# ── TEST HANDSHAKE ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    async def _test():
        engine = MistralEngine()
        print("Engine Initialized.")
        # Simple test
        res = await engine.chat("Who are you?", reasoning=True)
        print(f"Reasoning Test: {res.get('reply', 'No reply')}")
    
    asyncio.run(_test())
