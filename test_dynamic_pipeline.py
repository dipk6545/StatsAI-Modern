import os
import re
import json
import subprocess
import sys
import time
import plotly.graph_objects as go
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

# 1. --- CONFIG & KEYS ---
load_dotenv('server/.env')
CEREBRAS_KEY = os.getenv("CEREBRAS_API_KEY")
MODEL_ID = "llama3.1-8b"

def execute_llm_code(python_code: str):
    """Safely runs the LLM's data generation script and captures the manifest."""
    print("⚙️ PHASE 2: Executing AI-generated logic locally...")
    tmp_file = "_test_gen_data.py"
    # 🧠 SUPER-SAFE INJECTION: Global Numpy-to-JSON Transformer
    safe_code = (
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
        f"{python_code}"
    )
    with open(tmp_file, "w", encoding='utf-8') as f:
        f.write(safe_code)
    
    proc = subprocess.run([sys.executable, tmp_file], capture_output=True, text=True, timeout=10, encoding='utf-8')
    
    if proc.stderr:
        print(f"🔴 SCRIPT ERRORS (STDERR):\n{proc.stderr}")
    
    out = proc.stdout.strip()
    print(f"📝 SCRIPT OUTPUT (STDOUT):\n{out}\n")
    
    # 🛠️ UNIVERSAL REGEX: Capture Objects {} or Arrays []
    m = re.search(r'([\[\{].*[\]\}])', out, re.DOTALL) 
    return m.group(1) if m else out

from pathlib import Path
import webbrowser

def show_plotly(manifest_json: str):
    """Takes the raw manifest and shows it in your default browser using Plotly."""
    try:
        # Pre-process: AI sometimes prints a dict string instead of JSON
        clean_json = manifest_json.replace("'", '"')
        m = json.loads(clean_json)
        
        labels, values, title, ctype = [], [], "StatsAI Analysis", "pie"

        # 🛠️ AUTO-FLATTENING: Handle List of Dicts (e.g., [{"name": "Apple", "sales": 44}, ...])
        if isinstance(m, list):
            print("💡 DETECTED: List-of-Dictionaries format. Normalizing...")
            first = m[0] if len(m) > 0 else {}
            # Smart key detection
            label_key = next((k for k in ["name", "label", "category", "fruit"] if k in first), None)
            value_key = next((k for k in ["sales", "value", "total", "count", "cluster"] if k in first), None)
            
            if label_key and value_key:
                labels = [str(item.get(label_key)) for item in m]
                values = [item.get(value_key) for item in m]
        
        # 🛠️ STANDARD FORMAT: Handle {"labels": [], "values": []}
        elif isinstance(m, dict):
            # Support various key synonyms
            labels = m.get("labels") or m.get("categories") or []
            values = m.get("values") or [s.get("value") for s in m.get("slices", [])] or []
            title = m.get("title", "Analysis")
            ctype = str(m.get("type", "pie")).lower()
        
        print(f"📦 DATA INTEGRITY CHECK:")
        print(f"   - Labels Trace: {len(labels)}")
        print(f"   - Values Trace: {len(values)}")
        print(f"   - Chart Type: {ctype}\n")
        
        fig = go.Figure()
        if ctype == "bar":
            fig.add_trace(go.Bar(x=labels, y=values, marker_color="#7c3aed"))
        elif ctype == "pie":
             fig.add_trace(go.Pie(labels=labels, values=values))
        else: # Default: Scatter/Line
            fig.add_trace(go.Scatter(x=labels, y=values, mode='lines+markers', line=dict(color="#7c3aed", width=3)))
        
        fig.update_layout(
            title=f"PRO-STATS: {title}", 
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)"
        )
        
        # 🚀 OFFLINE-RESILIENT FORCE-LAUNCH
        output_path = Path("statsai_test_plot.html").resolve()
        fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
        
        print(f"📊 VISUALIZING: Graphing {len(values)} points...")
        webbrowser.open(output_path.as_uri())
        
    except Exception as e:
        print(f"❌ PLOTLY ERROR: {str(e)}")
        print(f"📦 RAW DATA RECEIVED: {manifest_json}")

def run_test():
    if not CEREBRAS_KEY:
        print("❌ ERROR: CEREBRAS_API_KEY not found in server/.env")
        return

    client = Cerebras(api_key=CEREBRAS_KEY)
    
    query = "generate a pie chart of fruit sales"
    prompt = (
        f"Generate a single Python script (wrapped in ```python...```) that calculates precisely the data for: '{query}'. "
        "The script MUST use numpy and PRINT exactly one JSON manifest via print(json.dumps(manifest)). "
        "CRITICAL: Output a SINGLE DICTIONARY with keys [type, labels, values, title]. Labels must be strings, values must be numbers."
    )

    print(f"🚀 PHASE 1: Cerebras LPU (235B) is designing the statistical logic...")
    start_time = time.time()
    try:
        resp = client.chat.completions.create(model=MODEL_ID, messages=[{"role": "user", "content": prompt}], max_tokens=8192)
        raw_ai = resp.choices[0].message.content.strip()
        print(f"✅ AI Script received in {time.time() - start_time:.2f}s.")
    except Exception as e:
        print(f"❌ CEREBRAS ERROR: {str(e)}")
        return

    # Extract the code block
    m_code = re.search(r'```python\s*(.*?)\s*```', raw_ai, re.DOTALL)
    if not m_code:
        print("❌ ERROR: No Python code block found in AI response.")
        print("-" * 60)
        print("RAW RESPONSE:")
        print(raw_ai)
        print("-" * 60)
        return

    # Print the code we found
    print("💎 LOGIC DESIGN:")
    print("-" * 30)
    print(m_code.group(1))
    print("-" * 30)

    # Phase 2 & 3: Run code then visualize
    try:
        manifest_raw = execute_llm_code(m_code.group(1))
        show_plotly(manifest_raw)
        print("💎 SUCCESS: Full Dynamic Pipeline Check Complete.")
    except Exception as e:
        print(f"❌ EXECUTION ERROR: {str(e)}")

if __name__ == "__main__":
    run_test()
