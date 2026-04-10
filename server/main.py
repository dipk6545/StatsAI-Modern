"""
StatsAI Headless API: Single-Shot Endpoint
The Frontend now handles the rotation loop for real-time status updates.
"""

import asyncio, json, logging, os, re
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Form
import uvicorn

# Engines
from groq import Groq
from cerebras.cloud.sdk import Cerebras
try:
    from mistralai import Mistral # v2
    MIST_MODE = "V2"
except ImportError:
    try:
        from mistralai.client import MistralClient # v1
        MIST_MODE = "V1"
    except ImportError:
        MIST_MODE = None

# ── INITIALIZATION ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')
GROQ_KEY  = os.getenv("GROQ_API_KEY", "").strip()
CERE_KEY  = os.getenv("CEREBRAS_API_KEY", "").strip()
MIST_KEY  = os.getenv("MISTRAL_API_KEY", "").strip()
ROOT = BASE_DIR.parent # For vault

VAULT_DIR = ROOT / ".statsai_vault"
VAULT_DIR.mkdir(exist_ok=True)
LOG_FILE  = VAULT_DIR / "statsai_api.log"

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger("StatsAI_Core")
app = FastAPI()

# ── LOGIC HELPERS ─────────────────────────────────────────────────────────────
def _get_system_prompt(domain: str, categories: str = "") -> str:
    cat_hint = f"\nCreative categories: {categories}" if categories else ""
    return (f"You are a Doctoral Statistical Researcher in {domain.upper()}.\n"
            f"MATH: Wrap formulas in ($$...$$).\n"
            f"VISUALS: Include <chart_params>{{...}}</chart_params> ONLY if 'chart/graph/plot/table' is in prompt."
            f"{cat_hint}")

def _sanitize(text: str) -> str:
    text = re.sub(r'<chart_params>.*?</chart_params>', '', text, flags=re.DOTALL)
    for a in ['```json','```python','```html','```']: text = text.replace(a,'')
    return text.strip()

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/api/config")
async def api_config():
    available = []
    # Check specific loaded keys
    if GROQ_KEY:  available.append("Groq Llama 3.3")
    if MIST_KEY:  available.append("Mistral Large")
    if CERE_KEY:  available.append("Cerebras Llama")
    logger.info(f"Available Engines: {available}")
    return {"models": available}

@app.post("/api/chat")
async def api_chat(
    message:  str = Form(...),
    model_id: str = Form(...), # Explicit model requested by frontend
    domain:   str = Form("statistics"),
    history:  str = Form("[]")
):
    # 1. Casual Greeting Detection (Persona Suppression)
    is_casual = re.match(r'^(hi|hello|hey|greetings|hola)\s*[\!\?\. ]*$', message, re.I)
    
    # 2. Resolve Provider
    client = None; actual_model = ""
    if "Groq" in model_id:
        client = Groq(api_key=GROQ_KEY); actual_model = "llama-3.3-70b-versatile"
    elif "Mistral" in model_id:
        client = Mistral(api_key=MIST_KEY) if MIST_MODE == "V2" else MistralClient(api_key=MIST_KEY)
        actual_model = "mistral-large-latest"
    elif "Cerebras" in model_id:
        client = Cerebras(api_key=CERE_KEY); actual_model = "llama3.1-8b"
    
    if not client: return {"error": "Key Missing"}, 401

    # 3. Construction
    sys_prompt = _get_system_prompt(domain)
    if is_casual:
        sys_prompt = "You are StatsAI, a helpful statistical assistant. Respond to this greeting briefly and naturally. Do not include math or technical jargon unless invited."
    
    msgs = [{"role": "system", "content": sys_prompt}]
    try:
        hist = json.loads(history)
        for h in hist[-6:]: msgs.append({"role": "assistant" if h.get('role') == 'bot' else "user", "content": h.get('text', '')})
    except: pass
    msgs.append({"role": "user", "content": message})

    # 3. Execution
    try:
        logger.info(f"Targeting {model_id}...")
        if "Mistral" in model_id:
            if MIST_MODE == "V2":
                resp = await asyncio.to_thread(client.chat.complete, model=actual_model, messages=msgs)
            else:
                resp = await asyncio.to_thread(client.chat, model=actual_model, messages=msgs)
            full_reply = resp.choices[0].message.content
        else:
            resp = await asyncio.to_thread(client.chat.completions.create, model=actual_model, messages=msgs)
            full_reply = resp.choices[0].message.content
        
        match = re.search(r'<chart_params>.*?</chart_params>', full_reply, flags=re.DOTALL)
        tag = match.group(0) if match else ""
        return {"reply": f"{_sanitize(full_reply)}\n\n{tag}"}
    except Exception as e:
        logger.error(f"Provider Error: {e}")
        return {"error": str(e)}, 500

if __name__ in {"__main__", "__mp_main__"}:
    uvicorn.run(app, host="127.0.0.1", port=3001)
