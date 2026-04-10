"""
StatsAI Headless API: Refined Smart Sync Edition
Added 'Casual Bypass' for greetings and Insight suppression.
"""

import asyncio, json, logging, os, re, random
from typing import Optional, Tuple, List
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
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')
GROQ_KEY  = os.getenv("GROQ_API_KEY", "").strip()
CERE_KEY  = os.getenv("CEREBRAS_API_KEY", "").strip()
MIST_KEY  = os.getenv("MISTRAL_API_KEY", "").strip()

VAULT_DIR = ROOT / ".statsai_vault"
VAULT_DIR.mkdir(exist_ok=True)
LOG_FILE  = VAULT_DIR / "statsai_api.log"

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger("StatsAI_Core")
app = FastAPI()

# ── SMART LOGIC HELPERS ───────────────────────────────────────────────────────
GREETINGS = ['hi', 'hello', 'hey', 'sup', 'yo', 'greetings']

def _is_casual(msg: str) -> bool:
    m = msg.lower().strip().strip('?!.')
    return m in GREETINGS or len(m) < 4

def _get_temp(msg: str) -> float:
    if any(k in msg.lower() for k in ['solve','calculate','derive','formula','mean','median']):
        return 0.1
    return 0.5

def _should_visualize(msg: str) -> bool:
    return any(k in msg.lower() for k in ['plot','chart','graph','table','pie','bar','box','histogram','scatter'])

async def provider_inference(name, client, model, messages, temp=0.5):
    try:
        if "Mistral" in name:
            if MIST_MODE == "V2":
                resp = await asyncio.to_thread(client.chat.complete, model=model, messages=messages, temperature=temp)
            else:
                resp = await asyncio.to_thread(client.chat, model=model, messages=messages, temperature=temp)
            return resp.choices[0].message.content
        else:
            resp = await asyncio.to_thread(client.chat.completions.create, model=model, messages=messages, temperature=temp)
            return resp.choices[0].message.content
    except:
        return None

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def api_chat(
    message:  str = Form(...),
    mode:     str = Form("single"),
    domain:   str = Form("statistics"),
    history:  str = Form("[]"),
):
    # 1. Define Model Pool & Probabilistic Entry
    available = []
    if GROQ_KEY: available.append(("Groq Llama 3.3", Groq(api_key=GROQ_KEY), "llama-3.3-70b-versatile"))
    if MIST_KEY and MISTRAL_MODE:
        m_c = Mistral(api_key=MIST_KEY) if MIST_MODE == "V2" else MistralClient(api_key=MIST_KEY)
        available.append(("Mistral Large", m_c, "mistral-large-latest"))
    if CERE_KEY: available.append(("Cerebras Llama", Cerebras(api_key=CERE_KEY), "llama3.1-8b"))
    
    if not available: return {"reply": "ERROR: Keys missing.", "model_used": "NONE"}
    
    r = random.randint(0, 8); start_idx = (r // 3) % len(available)
    stack = available[start_idx:] + available[:start_idx]
    primary = stack[0]; secondary = stack[1] if len(stack) > 1 else stack[0]
    
    is_c = _is_casual(message)
    temp = _get_temp(message)
    vis  = _should_visualize(message)
    
    # 2. PHASE 1: Main Inference
    sys_p = (f"You are a Doctoral Statistical Researcher.\n"
             f"STYLE: {'Casual and friendly' if is_c else 'Highly technical and academic'}.\n"
             f"VISUALS: {'Include <chart_params>{...}</chart_params>' if vis else 'DO NOT include charts.'}")
    
    msgs = [{"role": "system", "content": sys_p}]
    try:
        hist = json.loads(history)
        for h in hist[-4:]: msgs.append({"role": "assistant" if h.get('role') == 'bot' else "user", "content": h.get('text', '')})
    except: pass
    msgs.append({"role": "user", "content": message})

    main_reply = await provider_inference(primary[0], primary[1], primary[2], msgs, temp=temp)
    if not main_reply: return {"reply": "ERROR: Inference Failed.", "model_used": "ERROR"}

    # 3. PHASE 2: (Bypass for Casual Chat)
    full_output = main_reply
    if not is_c:
        expl_msgs = [
            {"role": "system", "content": "Provide a 2-sentence intuitive explanation of the technical answer below. NO MATH."},
            {"role": "user", "content": f"Answer: {main_reply}\n\nExplain this simply:"}
        ]
        explanation = await provider_inference(secondary[0], secondary[1], secondary[2], expl_msgs, temp=0.4)
        if explanation:
            tag_match = re.search(r'<chart_params>.*?</chart_params>', main_reply, flags=re.DOTALL)
            tag = tag_match.group(0) if tag_match else ""
            clean_main = re.sub(r'<chart_params>.*?</chart_params>', '', main_reply, flags=re.DOTALL).strip()
            full_output = f"{clean_main}\n\n**Analyst Insight:**\n{explanation}\n\n{tag}"
    
    return {"reply": full_output, "model_used": f"{primary[0]}{' + Insight' if not is_c else ''}"}

if __name__ in {"__main__", "__mp_main__"}:
    uvicorn.run(app, host="127.0.0.1", port=3001)
