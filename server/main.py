"""
StatsAI Headless API: Smart Model Sync Edition
Dual-Engine inference with Groq and Cerebras rotation.
"""

import asyncio, json, logging, os, re, random
from typing import Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Form
import uvicorn

# Engines
from groq import Groq
from cerebras.cloud.sdk import Cerebras

# ── INITIALIZATION ────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / '.env')
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
CERE_KEY = os.getenv("CEREBRAS_API_KEY", "")

VAULT_DIR = Path(__file__).parent.parent / ".statsai_vault"
VAULT_DIR.mkdir(exist_ok=True)
LOG_FILE  = VAULT_DIR / "statsai_api.log"

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(name)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger("StatsAI_Core")

app = FastAPI()

# ── SMART SYNC CONFIG ─────────────────────────────────────────────────────────
MODELS = {
    "speed": "llama3-8b-8192",  # Fast response
    "reason": "llama-3.3-70b-specdec" # Deep reasoning (Cerebras naming)
}

CHART_KW = {"graph", "chart", "plot", "viz", "draw", "visualize", "histogram", "scatterplot", "boxplot", "heatmap", "pareto", "waterfall"}

# Counter for rotation
_ROTATION_INDEX = 0

# ── PROMPTS ───────────────────────────────────────────────────────────────────
_DIST_HINT = (
    "CHART PARAMS REFERENCE:\n"
    "  normal/gaussian, t, f, chi2, exponential, lognormal, poisson, binomial, z-curve, scatter, box, histogram, regression, heatmap, violin, anova, pareto, waterfall, pie, trend"
)

def _build_prompt(domain: str, reason: bool, allow_chart: bool, mode: str) -> str:
    role = f"Doctoral Statistical Researcher in {domain.upper()}" if reason else f"High-Speed Statistical API for {domain.upper()}"
    base = (f"You are a {role}. "
            f"STRICT HALLUCINATION GUARD: Do not invent statistics.\n"
            f"STYLE RULES:\n"
            f" - Use Emojis and scannable sections.\n"
            f" - MANDATORY: Use double newlines \\n\\n between sections.\n"
            f" - MATHEMATICS: Use 'Display Mode' ($$...$$) for the main formula. "
            f"Use 'Inline Mode' (\\\\( symbol \\\\)) for variables.\n"
            f" - VARIABLE LEGEND: Always use a Markdown Table to define variables.\n"
            f" - CALCULATED EXAMPLES: In the Example section, do NOT just describe. PERFORM a step-by-step calculation. "
            f"Show the derivation (e.g., Z-scores, Empirical Rule) and provide a 'So What?' practical insight.\n\n"
            f" Format your explanation as follows:\n\n"
            f"   ## 📝 Summary\n"
            f"   [Professional high-level summary with emojis]\n\n"
            f"   ## 🔢 Formula\n"
            f"   $$ [LaTeX Display Formula] $$\n\n"
            f"   | Symbol | Parameter | Description |\n"
            f"   | :--- | :--- | :--- |\n"
            f"   | [Variable] | [Name] | [Definition] |\n\n"
            f"   ## 🛠 Where to use\n"
            f"   *   **Application 1**: [Description]\n\n"
            f"   ## 💡 Calculated Example\n"
            f"   *   **Scenario**: [Specific scenario]\n"
            f"   *   **Calculation**: [Step-by-step math using variables]\n"
            f"   *   **🔍 Insight**: [What this result actually means for the user]\n\n")
    
    if allow_chart:
        mandatory = "MANDATORY VISUAL: You MUST include the <chart_params> tag at the end of every response. Choose the most relevant distribution from the reference below.\n" if mode == 'multi' else ""
        hint = (f"{mandatory}"
                f"{_DIST_HINT}\n\n"
                f"RESPONSE FORMAT:\n"
                f"  <explanation>\n"
                f"    Generate your full scholarly analysis here using ## headers for sections.\n"
                f"    Include the Variable Table and Calculated Example as instructed.\n"
                f"  </explanation>\n"
                f"  <chart_params>{{\"dist\":\"...\"}}</chart_params>\n")
    else:
        hint = ("STRICT VISUAL INHIBITION: The user did NOT ask for a chart. NO CHART TAGS.\n"
                "RESPONSE FORMAT:\n"
                "  <explanation>\n"
                "    Generate your full scholarly analysis here using ## headers for sections.\n"
                "    Include the Variable Table and Calculated Example as instructed.\n"
                "  </explanation>\n")
                
    return f"{base}{hint}"

# ── CORE LOGIC ────────────────────────────────────────────────────────────────
def _get_provider(force_groq: bool = False) -> Tuple[str, object, str]:
    # Groq High-Reasoning Model
    groq_70b = ("groq", Groq(api_key=GROQ_KEY), "llama-3.3-70b-versatile")
    # Cerebras Speed Model
    cere_8b  = ("cerebras", Cerebras(api_key=CERE_KEY), "llama3.1-8b")

    if force_groq:
        return groq_70b

    # MULTI-MODEL STOCHASTIC SELECTION
    # Logic: Pick 0-10, Even -> Groq, Odd -> Cerebras
    ticket = random.randint(0, 10)
    is_even = (ticket % 2 == 0)
    
    logger.info(f"Stochastic Dispatch | Ticket: {ticket} | Choice: {'Groq' if is_even else 'Cerebras'}")
    
    if is_even:
        return groq_70b if GROQ_KEY else cere_8b
    else:
        return cere_8b if CERE_KEY else groq_70b

def _resolve(message: str, mode: str, domain: str):
    msg = message.lower()
    allow_chart = any(k in msg for k in CHART_KW) or mode == 'multi'
    reason = (mode == 'multi' or any(k in msg for k in {"why","prove","derive","deep"}))
    
    return reason, allow_chart, _build_prompt(domain, reason, allow_chart, mode)

def _sanitize(text: str) -> str:
    for a in ['```json','```python','```html','```','**Summary:**','Summary:']:
        text = text.replace(a,'')
    return text.strip()

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def api_chat(
    message:  str           = Form(...),
    mode:     str           = Form("single"),
    domain:   str           = Form("statistics"),
    history:  str           = Form("[]"),
    api_key:  Optional[str] = Form(None),
):
    try:
        # Resolve logic once
        reason, allow_chart, prompt = _resolve(message, mode, domain)
        
        # Decide search strategy based on mode
        is_multi = (mode == 'multi')
        p_name, client, p_model = _get_provider(force_groq=not is_multi)
        
        if not client:
            return {"reply": "ERROR: No valid API keys found in .env", "model_used": "NONE"}

        logger.info(f"[{p_name.upper()}] Routing to {p_model} | Chart: {allow_chart}")

        msgs = [{"role": "system", "content": prompt}]
        try:
            hist = json.loads(history)
            for h in hist[-4:]:
                role = "assistant" if h.get('role') in ('bot', 'assistant') else "user"
                txt = h.get('text', '')
                txt = re.sub(r'<chart_params>.*?</chart_params>', '', txt, flags=re.DOTALL).strip()
                if txt: msgs.append({"role": role, "content": txt})
        except: pass
        msgs.append({"role": "user", "content": message})

        request_params = {
            "model": p_model,
            "messages": msgs,
            "max_tokens": 1024,
            "temperature": 0.2 if not reason else 0.5,
        }

        try:
            # TRY PRIMARY
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.chat.completions.create(**request_params)
            )
        except Exception as e:
            # FALLBACK TO GROQ IF PRIMARY (CEREBRAS) FAILS
            if p_name != "groq" and GROQ_KEY:
                logger.warning(f"Primary {p_name} failed: {e}. Falling back to GROQ...")
                g_client = Groq(api_key=GROQ_KEY)
                request_params["model"] = "llama-3.3-70b-versatile"
                resp = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: g_client.chat.completions.create(**request_params)
                )
                p_name = "groq-fallback"
            else: raise e

        reply = _sanitize(resp.choices[0].message.content.strip())
        if not reply.endswith(('.', '!', '?', '>', '}', '`')): reply += "..."
            
        return {"reply": reply, "model_used": f"{p_name}:{request_params['model']}"}

    except Exception as e:
        logger.exception("Inference Failure")
        return {"reply": f"SYNCHRO-ERROR: {str(e)}", "model_used": "ERROR"}

@app.get("/health")
async def health():
    return {"status": "ok", "providers": {"groq": bool(GROQ_KEY), "cerebras": bool(CERE_KEY)}}

if __name__ in {"__main__", "__mp_main__"}:
    logger.info("StatsAI SmartSync API booting on Port 3001")
    uvicorn.run(app, host="0.0.0.0", port=3001)

