import os
import re
import json
import subprocess
import sys
import time
import asyncio
import plotly.graph_objects as go
from dotenv import load_dotenv
from groq import Groq
from cerebras.cloud.sdk import Cerebras

# 1. --- CONFIG & KEYS ---
load_dotenv('server/.env')
G_KEY = os.getenv("GROQ_API_KEY")
C_KEY = os.getenv("CEREBRAS_API_KEY")

# Optimization: We use 70B for the teacher and 8B for the engineer
G_MODEL = "llama-3.3-70b-versatile" 
C_MODEL = "llama3.1-8b"

async def get_cerebras_math(client, query):
    """PHASE 1: Cerebras acts as the Research Engineer (designs the math)."""
    prompt = (
        f"Generate a single Python script (wrapped in ```python...```) that calculates precisely the data for: '{query}'. "
        "The script MUST use numpy/scipy and PRINT exactly one JSON manifest via print(json.dumps(manifest)). "
        "CRITICAL: Do NOT use print() for any other text or debugging. ONLY the JSON manifest should be printed. No matplotlib. Use scipy.special.erf if needed."
    )
    print("📡 Cerebras (8B) is designing the mathematical logic...")
    resp = await asyncio.get_event_loop().run_in_executor(
        None, lambda: client.chat.completions.create(model=C_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=1024)
    )
    return resp.choices[0].message.content.strip()

async def get_groq_insight(client, query):
    """PHASE 2: Groq takes on the role of an expert instructor, leveraging deep domain knowledge in data science and mathematics—particularly in linear algebra, statistics, probability, and calculus—to deliver clear explanations, including intuitive interpretations of graphs."""
    prompt = (
        f"Explain this statistical concept to a Grade 10 student: '{query}'. "
        "Include a 2-sentence summary and a deep <explanation> block with derivations if needed. "
        "Do NOT generate any chart tags or JSON data."
    )
    print("🧠 Groq (70B) is preparing the teacher-level analysis...")
    resp = await asyncio.get_event_loop().run_in_executor(
        None, lambda: client.chat.completions.create(model=G_MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=2048)
    )
    return resp.choices[0].message.content.strip()

from pathlib import Path
import webbrowser

def execute_and_plot(ai_script):
    """PHASE 3: Local CPU executes the math and Plotly visualizes."""
    m_code = re.search(r'```python\s*(.*?)\s*```', ai_script, re.DOTALL)
    if not m_code: 
        print("❌ ERROR: No Python script found in AI response block.")
        return

    # Execution Sandbox - Hardened for Greek characters and Numpy types
    print("⚙️ PHASE 3: Executing AI-generated logic locally...")
    code_content = m_code.group(1)
    
    code = (
        "import numpy as np\n"
        "import json\n"
        "import sys\n"
        "import scipy.stats as stats\n"
        "class NpEncoder(json.JSONEncoder):\n"
        "    def default(self, obj):\n"
        "        if isinstance(obj, np.integer): return int(obj)\n"
        "        if isinstance(obj, np.floating): return float(obj)\n"
        "        if isinstance(obj, np.ndarray): return obj.tolist()\n"
        "        return super(NpEncoder, self).default(obj)\n"
        "sys.stdout.reconfigure(encoding='utf-8')\n"
        "original_dumps = json.dumps\n"
        "json.dumps = lambda obj, **kwargs: original_dumps(obj, cls=NpEncoder, **kwargs)\n"
        f"{code_content}"
    )
    
    tmp_file = "_tmp_final_integrated.py"
    with open(tmp_file, "w", encoding='utf-8') as f: f.write(code)
    
    proc = subprocess.run([sys.executable, tmp_file], capture_output=True, text=True, encoding='utf-8')
    
    if proc.returncode != 0:
        print(f"❌ EXECUTION FAILED!")
        print(f"🔴 ERRORS (STDERR):\n{proc.stderr}")
        return

    raw_stdout = proc.stdout.strip()
    
    # 🛠️ UNIVERSAL REGEX: Capture Objects {} or Arrays []
    m_match = re.search(r'([\[\{].*[\]\}])', raw_stdout, re.DOTALL)
    if not m_match:
        print("❌ ERROR: No JSON manifest found in script output.")
        return
        
    try:
        m = json.loads(m_match.group(1).replace("'", '"'))
        labels, values, title, ctype = [], [], "Integrated Analysis", "line"

        # 🛠️ COORDINATE DETECTIVE: List-of-Dicts or List-of-Lists
        if isinstance(m, list):
            print("💡 DETECTED: List format. Normalizing...")
            if len(m) > 0 and isinstance(m[0], dict):
                first = m[0]
                label_key = next((k for k in ["x", "name", "label", "category"] if k in first), None)
                value_key = next((k for k in ["y", "sales", "value", "total", "count"] if k in first), None)
                if label_key and value_key:
                    labels = [item.get(label_key) for item in m]
                    values = [item.get(value_key) for item in m]
            elif len(m) > 0 and isinstance(m[0], list) and len(m[0]) >= 2:
                labels = [item[0] for item in m]
                values = [item[1] for item in m]
        
        # 🛠️ STANDARD FORMAT
        elif isinstance(m, dict):
            # Support various key synonyms including scientific x/y and business labels/values
            labels = m.get("labels") or m.get("categories") or m.get("x") or []
            values = m.get("values") or m.get("y") or m.get("data") or [s.get("value") for s in m.get("slices", [])] or []
            title = m.get("title", "Analysis")
            ctype = str(m.get("type", "line")).lower()

        # 🛠️ SCIENCE-GRADE CLEANER: Robust float conversion
        def to_f(val):
            try: return float(val)
            except: return val

        labels = [to_f(x) for x in labels]
        values = [to_f(y) for y in values]

        print(f"📦 DATA INTEGRITY CHECK:")
        print(f"   - Points: {len(values)}")
        if len(values) > 0:
            print(f"   - X Type: {type(labels[0])} | Sample: {labels[:2]}")
            print(f"   - Y Type: {type(values[0])} | Sample: {values[:2]}")
        print(f"   - Chart Type: {ctype}\n")
        
        fig = go.Figure()
        if ctype == "bar":
            fig.add_trace(go.Bar(x=labels, y=values, marker_color="#7c3aed"))
        elif ctype == "pie":
             fig.add_trace(go.Pie(labels=labels, values=values))
        else: # Standard Line/Scatter
            fig.add_trace(go.Scatter(
                x=labels, y=values, 
                mode='lines+markers', 
                fill='tozeroy', 
                line=dict(color="#7c3aed", width=3),
                marker=dict(size=4)
            ))
            
        fig.update_layout(
            title=f"PRO-STATS: {title}", 
            template="plotly_white",
            xaxis_title="Input (X)",
            yaxis_title="Probability Density (Y)",
            xaxis=dict(autorange=True),
            yaxis=dict(autorange=True)
        )
        
        # 🚀 OFFLINE-RESILIENT FORCE-LAUNCH
        output_path = Path("statsai_integrated_plot.html").resolve()
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        
        print(f"📊 VISUALIZING: Pipeline streaming {len(values)} points...")
        webbrowser.open(output_path.as_uri())
        print("💎 SUCCESS: Statistical visualization should now be fully rendered.")
    except Exception as e:
        print(f"❌ PROCESSING ERROR: {str(e)}")

async def run_integrated_test():
    if not G_KEY or not C_KEY:
        print("❌ ERROR: Keys missing in server/.env")
        return

    g_client = Groq(api_key=G_KEY)
    c_client = Cerebras(api_key=C_KEY)
    query = "Normal Probability Distribution"

    print(f"🚀 Launching Parallel Integrated Compute: Groq(Reasoning) + Cerebras(Math)...")
    start = time.time()
    
    # TRIGGER BOTH AT ONCE (This is what the StatsAI main server does)
    ai_script, ai_insight = await asyncio.gather(
        get_cerebras_math(c_client, query),
        get_groq_insight(g_client, query)
    )
    
    elapsed = time.time() - start
    print(f"\n✅ Total Infrastructure Handshake completed in {elapsed:.2f}s.")
    
    print("\n" + "="*60)
    print("🎓 TEACHER-LEVEL ANALYTICAL INSIGHT (Groq 70B)")
    print("="*60)
    print(ai_insight)
    print("="*60 + "\n")
    
    print("🎨 DESIGNER DATA MANIFEST (Cerebras 8B Code -> Local CPU)")
    execute_and_plot(ai_script)

if __name__ == "__main__":
    asyncio.run(run_integrated_test())
