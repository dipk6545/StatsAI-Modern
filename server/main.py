"""
StatsAI Headless API: Single-Shot Endpoint
The Frontend now handles the rotation loop for real-time status updates.
"""

import asyncio, json, logging, os, re
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException
import uvicorn
from multimodel_engine import MultiModelEngine

# Engines
from groq import Groq
from cerebras.cloud.sdk import Cerebras
try:
    # Optimized for user environment: Speakeasy structure
    from mistralai.client import Mistral
    MIST_MODE = "V2_SPEAKEASY"
except ImportError:
    try:
        from mistralai import Mistral # v2 Standard
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
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
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
    return (f"You are a Specialist Statistical Researcher in {domain.upper()}.\n"
            f"GOAL: Provide highly detailed, technically accurate content for your assigned task.\n"
            f"MATH: Use ($$...$$) and ($...$). Be precise with LaTeX (S_t, \\mu, \\sigma).\n"
            f"FINANCE: Always include an '## Assumptions' block (S0, \\mu, \\sigma, days).\n"
            f"VISUALS: Include <chart_params> with REAL simulated data (252 points). No fake trends.\n"
            f"REPRODUCIBILITY: Always use `np.random.seed(42)` in your code examples.\n"
            f"CODE: Use vectorized numpy operations. Professional docstrings and comments required."
            f"{cat_hint}")

def _extract_raw_chart_json(text: str) -> str:
    # Look for anything that looks like a chart JSON block
    match = re.search(r'\{\s*"(?:type|chart_type|dist)"\s*:', text)
    if not match: return ""
    start_idx = match.start()
    open_braces = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{': open_braces += 1
        elif text[i] == '}': open_braces -= 1
        if open_braces == 0:
            return text[start_idx:i+1]
    return ""

def _sanitize(text: str) -> str:
    text = re.sub(r'<chart_params>.*?</chart_params>', '', text, flags=re.DOTALL)
    return text.strip()

def _update_stat(model_id: str):
    import os
    path = "d:/StatsAi/model_stats.json"
    try:
        stats = {}
        if os.path.exists(path):
            with open(path, 'r') as f:
                stats = json.load(f)
        stats[model_id] = stats.get(model_id, 0) + 1
        with open(path, 'w') as f:
            json.dump(stats, f)
    except Exception as e:
        logger.error(f"Failed to update stats: {e}")

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/api/config")
async def api_config():
    available = []
    if GROQ_KEY:  available.extend(["llama-3.1-8b-instant", "qwen3-32b", "llama-3.3-70b-versatile", "gpt-oss-120b", "kimi-k2-instruct", "deepseek-r1-distill-70b"])
    if MIST_KEY:  available.append("codestral")
    if CERE_KEY:  available.extend(["llama3.1-8b", "qwen-3-235b-a22b-instruct-2507"])
    if GEMINI_KEY: available.append("gemini-2.5-flash-lite")
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
    groq_models = ["llama-3.1-8b-instant", "qwen-2.5-32b", "qwen3-32b", "llama-3.3-70b-versatile", "gpt-oss-120b", "kimi-k2-instruct", "deepseek-r1-distill-70b"]
    cerebras_models = ["llama3.1-8b", "qwen-3-235b-a22b-instruct-2507"]
    
    if model_id in groq_models:
        client = Groq(api_key=GROQ_KEY)
        if model_id == "qwen3-32b":
            actual_model = "llama-3.3-70b-versatile" # Qwen was decommissioned on Groq, using Llama-3.3 as high-performance substitute
        elif model_id in ["deepseek-r1-distill-70b", "gpt-oss-120b", "kimi-k2-instruct"]:
            actual_model = "llama-3.3-70b-versatile"
        else:
            actual_model = model_id
    elif model_id == "codestral":
        if MIST_MODE == "V2_SPEAKEASY":
            from mistralai.client import Mistral
            client = Mistral(api_key=MIST_KEY)
        elif MIST_MODE == "V2":
            client = Mistral(api_key=MIST_KEY)
        else:
            client = MistralClient(api_key=MIST_KEY)
        actual_model = "codestral-latest"
    elif model_id in cerebras_models:
        client = Cerebras(api_key=CERE_KEY)
        if model_id == "qwen-3-235b-a22b-instruct-2507":
            actual_model = "llama3.1-70b"
        else:
            actual_model = model_id
    elif model_id == "gemini-2.5-flash-lite":
        actual_model = model_id
        if not GEMINI_KEY: return {"error": "Gemini Key Missing"}, 401
    elif "Groq" in model_id:
        client = Groq(api_key=GROQ_KEY); actual_model = "llama-3.3-70b-versatile"
    elif "Mistral" in model_id:
        if MIST_MODE == "V2_SPEAKEASY":
            from mistralai.client import Mistral
            client = Mistral(api_key=MIST_KEY)
        elif MIST_MODE == "V2":
            client = Mistral(api_key=MIST_KEY)
        else:
            client = MistralClient(api_key=MIST_KEY)
        actual_model = "mistral-medium-latest"
    elif "Cerebras" in model_id:
        client = Cerebras(api_key=CERE_KEY); actual_model = "llama3.1-8b"
    elif "Gemini" in model_id:
        actual_model = "gemini-2.5-flash"
        if not GEMINI_KEY: return {"error": "Gemini Key Missing"}, 401
    
    if not client and actual_model == "": return {"error": "Key Missing or Unknown Model"}, 401

    # 3. Construction
    sys_prompt = _get_system_prompt(domain)
    if is_casual:
        sys_prompt = "You are StatsAI, a helpful statistical assistant. Respond to this greeting briefly and naturally. Do not include math or technical jargon unless invited."
    
    msgs = [{"role": "system", "content": sys_prompt}]
    try:
        hist = json.loads(history)
        # Deep Memory: Last 12 messages for better context
        for h in hist[-12:]: msgs.append({"role": "assistant" if h.get('role') == 'bot' else "user", "content": h.get('text', '')})
    except: pass
    msgs.append({"role": "user", "content": message})

    # 3. Execution
    try:
        logger.info(f"Targeting {model_id}...")
        if model_id == "codestral" or "Mistral" in model_id:
            if MIST_MODE in ["V2", "V2_SPEAKEASY"]:
                resp = await asyncio.to_thread(client.chat.complete, model=actual_model, messages=msgs, temperature=0.8)
            else:
                resp = await asyncio.to_thread(client.chat, model=actual_model, messages=msgs)
            full_reply = resp.choices[0].message.content
        elif "gemini" in model_id.lower():
            import httpx
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{actual_model}:generateContent?key={GEMINI_KEY}"
            # Transform history to Gemini format
            gemini_msgs = []
            for m in msgs:
                role = "user" if m['role'] in ['user', 'system'] else "model"
                gemini_msgs.append({"role": role, "parts": [{"text": m['content']}]})
            
            async with httpx.AsyncClient() as hclient:
                resp = await hclient.post(url, json={"contents": gemini_msgs}, timeout=60.0)
                if resp.status_code == 200:
                    full_reply = resp.json()['candidates'][0]['content']['parts'][0]['text']
                else:
                    raise Exception(f"Gemini API Error {resp.status_code}: {resp.text}")
        else:
            resp = await asyncio.to_thread(client.chat.completions.create, model=actual_model, messages=msgs, temperature=0.8)
            full_reply = resp.choices[0].message.content
        
        match = re.search(r'<chart_params>.*?</chart_params>', full_reply, flags=re.DOTALL)
        tag = match.group(0) if match else ""
        
        if not tag:
            raw_json = _extract_raw_chart_json(full_reply)
            if raw_json:
                tag = f"<chart_params>{raw_json}</chart_params>"
                full_reply = full_reply.replace(raw_json, "")
                
        _update_stat(actual_model)
        return {"reply": f"{_sanitize(full_reply)}\n\n{tag}"}
    except Exception as e:
        logger.error(f"Provider Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/multi_chat")
async def api_multi_chat(message: str = Form(...), history: str = Form("[]"), domain: str = Form("general")):
    try:
        engine = MultiModelEngine()
        reply = await engine.run_pipeline(message, history, domain)
        return {"reply": reply}
    except Exception as e:
        logger.error(f"MultiModel Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ in {"__main__", "__mp_main__"}:
    uvicorn.run(app, host="127.0.0.1", port=3001)
