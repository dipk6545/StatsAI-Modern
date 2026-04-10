"""
StatsAI Headless API: Multi-Model Synthesis Edition
Dual-Model brainstorming for categorical visuals.
"""

import asyncio, json, logging, os, re
from typing import Optional, Tuple, List
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Form
import uvicorn

# Engines
from groq import Groq
from cerebras.cloud.sdk import Cerebras

# ── INITIALIZATION ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')
GROQ_KEY  = os.getenv("GROQ_API_KEY", "").strip()
CERE_KEY  = os.getenv("CEREBRAS_API_KEY", "").strip()

VAULT_DIR = ROOT / ".statsai_vault"
VAULT_DIR.mkdir(exist_ok=True)
LOG_FILE  = VAULT_DIR / "statsai_api.log"

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger("StatsAI_Core")
app = FastAPI()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def _get_system_prompt(domain: str, mode: str, suggested_categories: str = "") -> str:
    categories_hint = f"\nBRAINSTORMED CATEGORIES: Use these specific categories in your data: {suggested_categories}" if suggested_categories else ""
    return (f"You are a Doctoral Statistical Researcher in {domain.upper()}.\n"
            f"STYLE: Professional, emoji-supported, scannable.\n"
            f"LAYOUT: Use double newlines between sections.\n"
            f"MATHEMATICS: Use display mode ($$...$$). Explaining all terms in a table.\n"
            f"VISUALIZATION: If the user asks for a chart/graph (Pie, Bar, Box, etc.), you MUST include a <chart_params>{{...}}</chart_params> block with relevant data points."
            f"{categories_hint}")

def _sanitize(text: str) -> str:
    # Remove all chart tags from visible text
    text = re.sub(r'<chart_params>.*?</chart_params>', '', text, flags=re.DOTALL)
    for a in ['```json','```python','```html','```']: text = text.replace(a,'')
    return text.strip()

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def api_chat(
    message:  str = Form(...),
    mode:     str = Form("single"),
    domain:   str = Form("statistics"),
    history:  str = Form("[]"),
):
    # 1. Identify Models
    stack = []
    if GROQ_KEY: stack.append(("Groq Llama 3.3", Groq(api_key=GROQ_KEY), "llama-3.3-70b-versatile"))
    if CERE_KEY: stack.append(("Cerebras Llama", Cerebras(api_key=CERE_KEY), "llama3.1-8b"))
    
    if len(stack) < 1: return {"reply": "ERROR: No API keys configured.", "model_used": "NONE"}
    
    primary = stack[0]
    secondary = stack[1] if len(stack) > 1 else stack[0]
    
    suggested_cats = ""
    # 2. MULTI-MODEL BRAINSTORMING (If applicable)
    if mode == 'multi' and any(kw in message.lower() for kw in ['pie', 'bar', 'chart', 'plot', 'box', 'categories', 'segment']):
        try:
            logger.info(f"Multi-Model Phase: Brainstorming categories via {secondary[0]}")
            brainstorm_msgs = [
                {"role": "system", "content": "You are a creative data scientist. Brainstorm 5 unique, highly specific, and random categorical names for a dataset based on the user's prompt. Return ONLY a comma-separated list of names."},
                {"role": "user", "content": message}
            ]
            resp = await asyncio.to_thread(secondary[1].chat.completions.create, model=secondary[2], messages=brainstorm_msgs, max_tokens=50)
            suggested_cats = resp.choices[0].message.content.strip()
            logger.info(f"Brainstormed Categories: {suggested_cats}")
        except Exception as e:
            logger.warning(f"Brainstorming failed: {e}")

    # 3. CONSTRUCT MAIN PROMPT
    msgs = [{"role": "system", "content": _get_system_prompt(domain, mode, suggested_cats)}]
    try:
        hist = json.loads(history)
        for h in hist[-6:]:
            role = "assistant" if h.get('role') in ('bot', 'assistant') else "user"
            msgs.append({"role": role, "content": h.get('text', '')})
    except: pass
    msgs.append({"role": "user", "content": message})

    # 4. MAIN INFERENCE
    try:
        logger.info(f"Primary Inference via {primary[0]}")
        resp = await asyncio.to_thread(primary[1].chat.completions.create, model=primary[2], messages=msgs)
        full_reply = resp.choices[0].message.content
        
        # Ensure only one chart tag exists if multiple were generated
        match = re.search(r'<chart_params>.*?</chart_params>', full_reply, flags=re.DOTALL)
        chart_tag = match.group(0) if match else ""
        
        clean_text = _sanitize(full_reply)
        return {"reply": f"{clean_text}\n\n{chart_tag}", "model_used": primary[0]}
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return {"reply": f"SYSTEM-ERROR: {str(e)}", "model_used": "ERROR"}

if __name__ in {"__main__", "__mp_main__"}:
    uvicorn.run(app, host="127.0.0.1", port=3001)
