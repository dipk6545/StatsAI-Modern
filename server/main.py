"""
StatsAI Headless API
Based on the Unified Architecture Engine
"""

import asyncio
import json
import logging
import os
import re
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, Form
import uvicorn
from groq import Groq

# ── LOGGING ────────────────────────────────────────────────────────────────────
VAULT_DIR = Path(__file__).parent.parent / ".statsai_vault"
VAULT_DIR.mkdir(exist_ok=True)
LOG_FILE = VAULT_DIR / "statsai_api.log"

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger("StatsAI_API")

load_dotenv(Path(__file__).parent.parent / '.env')
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

SPEED_MODEL  = "llama-3.1-8b-instant"
REASON_MODEL = "llama-3.3-70b-versatile"

app = FastAPI()

# ── ROUTING TABLES ─────────────────────────────────────────────────────────────
REASON_KW = {"why","prove","derive","compare","explain","interpret",
             "diagnose","should i","which is better","difference between"}
REASON_DOMAINS = {"linear algebra","deep learning","ai","regression","inferential"}

# ── BACKEND PROMPTS ────────────────────────────────────────────────────────────
_DIST_HINT = (
    "CHART PARAMS REFERENCE:\n"
    "  normal/gaussian     → {\"dist\":\"normal\"}\n"
    "  t-distribution      → {\"dist\":\"t\"}\n"
    "  f-distribution      → {\"dist\":\"f\"}\n"
    "  chi-square          → {\"dist\":\"chi2\"}\n"
    "  exponential         → {\"dist\":\"exponential\"}\n"
    "  log-normal          → {\"dist\":\"lognormal\"}\n"
    "  poisson             → {\"dist\":\"poisson\"}\n"
    "  binomial            → {\"dist\":\"binomial\"}\n"
    "  z-curve             → {\"dist\":\"z\"}\n"
    "  scatter             → {\"dist\":\"scatter\"}\n"
    "  box plot            → {\"dist\":\"box\"}\n"
    "  histogram           → {\"dist\":\"histogram\"}\n"
    "  regression          → {\"dist\":\"regression\"}\n"
    "  heatmap             → {\"dist\":\"heatmap\"}\n"
    "  violin              → {\"dist\":\"violin\"}\n"
    "  anova               → {\"dist\":\"anova\"}\n"
    "  pareto              → {\"dist\":\"pareto\"}\n"
    "  waterfall           → {\"dist\":\"waterfall\"}\n"
    "  pie                 → {\"dist\":\"pie\"}\n"
    "  trend/line          → {\"dist\":\"trend\"}\n"
)

_FORMAT = (
    "\nRESPONSE FORMAT — follow exactly, no deviation:\n"
    "  <explanation>1-3 sentence explanation here.</explanation>\n"
    "  <chart_params>{\"dist\":\"...\", ...params}</chart_params>\n\n"
    "RULES:\n"
    "  • Pure JSON in <chart_params>. No code. No markdown. No Plotly JSON.\n"
    "  • If no chart needed, omit <chart_params>.\n"
    "  • Multi mode: explanation is multi-paragraph with derivation.\n"
)

def _build_prompt(domain: str, reason: bool) -> str:
    if reason:
        return (f"You are a Doctoral Statistical Researcher in {domain.upper()}. "
                f"Show derivations, flag assumptions, give deep insight.\n\n"
                f"{_DIST_HINT}{_FORMAT}")
    return (f"You are a High-Speed Statistical API in {domain.upper()}. "
            f"Be concise. One sentence, then chart params.\n\n"
            f"{_DIST_HINT}{_FORMAT}")

def _resolve(message: str, mode: str, domain: str):
    msg = message.lower()
    dom = domain.lower()
    reason = (mode == 'multi'
              or dom in REASON_DOMAINS
              or any(k in msg for k in REASON_KW))
    model  = REASON_MODEL if reason else SPEED_MODEL
    return model, _build_prompt(domain, reason)

def _sanitize(text: str) -> str:
    for a in ['```json','```python','```html','```','**Summary:**','Summary:']:
        text = text.replace(a,'')
    return text.strip()

# ── ENDPOINT ───────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def api_chat(
    message:  str           = Form(...),
    mode:     str           = Form("single"),
    domain:   str           = Form("statistics"),
    history:  str           = Form("[]"),
    api_key:  Optional[str] = Form(None),
):
    try:
        key = (api_key or '').strip() or GROQ_API_KEY
        if not key:
            return {"reply": "ERROR: No GROQ_API_KEY set in .env", "model_used": None}

        hist  = json.loads(history)
        model, prompt = _resolve(message, mode, domain)
        logger.info(f"[{model.split('-')[1]}] {domain} | {message[:50]}")

        msgs = [{"role": "system", "content": prompt}]
        for h in hist[-6:]:
            role = "assistant" if h.get('role') in ('bot', 'assistant', 'model') else "user"
            # Strip chart tags to save conversational context tokens
            txt = re.sub(r'<chart_params>.*?</chart_params>', '', h.get('text', ''), flags=re.DOTALL).strip()
            if txt:
                msgs.append({"role": role, "content": txt})
        
        msgs.append({"role": "user", "content": message})

        client = Groq(api_key=key)
        resp   = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.chat.completions.create(model=model, messages=msgs, max_tokens=1024)
        )

        reply = _sanitize(resp.choices[0].message.content.strip())
        return {"reply": reply, "model_used": model}

    except Exception as e:
        logger.exception("API error")
        return {"reply": f"SYSTEM ERROR: {e}", "model_used": None}

@app.get("/health")
async def health():
    return {"status":"ok","speed":SPEED_MODEL,"reason":REASON_MODEL,"key_set":bool(GROQ_API_KEY)}

if __name__ in {"__main__", "__mp_main__"}:
    logger.info("Starting StatsAI API Headless Server on port 3001")
    uvicorn.run(app, host="0.0.0.0", port=3001)
