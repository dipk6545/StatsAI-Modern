import streamlit as st
import requests
import json
import plotly.io as pio
import re

st.set_page_config(page_title="Stats AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
:root {
    --primary: #6d28d9;
    --radius: 12px;
    --shadow: 0 4px 6px rgba(0,0,0,0.05);
}
footer {visibility: hidden;}
#stDecoration {display: none;}
[data-testid="stSidebar"] { padding-top: 1rem; }

/* Right Sidebar / Activity Panel Structure */
[data-testid="column"]:nth-of-type(2) {
    background-color: var(--secondary-background-color);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.02);
}

/* Hide Streamlit Avatars completely */
[data-testid="stChatMessage"] > div:first-child { display: none !important; }

/* Base Chat Box Wrap */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 24px;
    display: flex;
    width: 100% !important;
}

/* Chat text cards limits */
[data-testid="stChatMessage"] > div:last-child {
    max-width: 75% !important;
    padding: 16px 20px;
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    font-size: 0.95rem;
    line-height: 1.5;
}

/* User -> Right alignment */
[data-testid="stChatMessage"]:has(.user-msg) {
    justify-content: flex-end;
}
[data-testid="stChatMessage"]:has(.user-msg) > div:last-child {
    background-color: var(--primary);
    color: white;
    border-bottom-right-radius: 4px;
    margin-left: auto;
}

/* AI -> Left alignment */
[data-testid="stChatMessage"]:has(.model-msg) {
    justify-content: flex-start;
}
[data-testid="stChatMessage"]:has(.model-msg) > div:last-child {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    border-bottom-left-radius: 4px;
    margin-right: auto;
    border: 1px solid rgba(128,128,128,0.05);
}

/* Suggestion Cards Hover Effects */
div.stButton > button {
    border-radius: var(--radius) !important;
    padding: 15px !important;
    border: 1px solid rgba(128,128,128,0.1) !important;
    background-color: var(--secondary-background-color) !important;
    transition: all 0.2s ease;
    width: 100%;
}
div.stButton > button:hover {
    border-color: var(--primary) !important;
    box-shadow: 0 4px 12px rgba(109,40,217,0.1) !important;
    background-color: transparent !important;
}
div.stButton > button p {
    font-size: 0.95rem;
}

/* AI Activity Timeline Typography */
.pipeline-step {
    position: relative;
    padding-left: 20px;
    margin-bottom: 20px;
    border-left: 3px solid rgba(109,40,217,0.1);
}
.pipeline-step.active {
    border-left-color: var(--primary);
}
.pipeline-step::before {
    content: "●";
    position: absolute;
    left: -7.5px;
    top: -2px;
    font-size: 11px;
    color: rgba(109,40,217,0.1);
    background: var(--background-color);
}
.pipeline-step.active::before {
    color: var(--primary);
    text-shadow: 0 0 8px var(--primary);
}
.pipeline-title { font-weight: 700; font-size: 0.9rem; color: var(--text-color); }
.pipeline-desc { font-size: 0.8rem; color: gray; margin-top: 4px;}

</style>
""", unsafe_allow_html=True)

# State init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "specialization" not in st.session_state:
    st.session_state.specialization = "statistics"
if "workflow_steps" not in st.session_state:
    st.session_state.workflow_steps = []
if "engine_mode" not in st.session_state:
    st.session_state.engine_mode = "multi"
if "trigger_prompt" not in st.session_state:
    st.session_state.trigger_prompt = None

# Sidebar Content Helper
def sb_select(label, key):
    # Highlight logic using native st.button types
    b_type = "primary" if st.session_state.specialization == key else "secondary"
    if st.button(label, type=b_type, use_container_width=True):
        st.session_state.specialization = key
        st.session_state.messages = [] # Clean UI transition
        st.rerun()

# 1. SIDEBAR (Left 20%)
with st.sidebar:
    st.markdown("<h2 style='font-size: 1.6rem; font-weight: 800; margin-bottom: 0px;'>⚙️ AI Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.85rem; color: gray;'>Intelligent Data Workspace</p>", unsafe_allow_html=True)
    st.write("")
    
    st.markdown("##### Subject Context")
    sb_select("📊 Statistics", "statistics")
    sb_select("🎲 Probability", "probability")
    sb_select("✨ Data Science", "data_science")
    sb_select("📄 Research", "research")

    st.write("<br><br>", unsafe_allow_html=True)
    
    # Hide complex configuration into a popover modal!
    with st.popover("⚙️ Preferences", use_container_width=True):
        st.markdown("**Configuration**")
        multi_mode = st.toggle("Multi-Model Pipeline", value=(st.session_state.engine_mode=="multi"))
        st.session_state.engine_mode = "multi" if multi_mode else "single"
        st.text_area("Optimization Instruction", placeholder="e.g. Always format with markdown tables...")

# 2. MAIN WORKSPACE (60%) & 3. AI ACTIVITY (20%)
cols = st.columns([3, 1], gap="large")
main_col = cols[0]
activity_col = cols[1]

with main_col:
    # 2A. Smart Welcome State
    if not st.session_state.messages:
        st.markdown(f"<h1 style='font-weight: 800; font-size: 2.2rem;'>{st.session_state.specialization.replace('_', ' ').title()} Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1rem; color: gray; margin-bottom: 30px;'>Context-aware intelligence for your workflows. Select a starting point below.</p>", unsafe_allow_html=True)
        
        # 2B. Suggestion Grid Cards
        c1, c2 = st.columns(2)
        if c1.button("📊 **Hypothesis Testing**\n\nRun A/B testing frameworks against data"):
            st.session_state.trigger_prompt = "Can you walk me through the steps for a Two-Sample T-Test Hypothesis Testing?"
            st.rerun()
        if c1.button("📈 **Regression Analysis**\n\nExplore variable relationships naturally"):
            st.session_state.trigger_prompt = "Explain linear regression and generate a dummy scatter plot with a best fit line."
            st.rerun()
        if c2.button("📉 **Probability Distributions**\n\nAnalyze normal & binomial curves fully"):
            st.session_state.trigger_prompt = "Draw a normal distribution curve and explain the 68-95-99.7 rule."
            st.rerun()
        if c2.button("📦 **Data Summary**\n\nDescribe central tendencies accurately"):
            st.session_state.trigger_prompt = "Explain mean, median, and mode visually using a theoretical dataset."
            st.rerun()
            
    else:
        # Chat Component Rendering
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                marker = "<span class='user-msg'></span>" if msg["role"] == "user" else "<span class='model-msg'></span>"
                st.markdown(marker + msg["text"], unsafe_allow_html=True)
                if "chart" in msg:
                    st.plotly_chart(msg["chart"], use_container_width=True, key=str(hash(msg["text"])))
                if "explanation" in msg:
                    with st.expander("🔍 **Deep Analysis Document**"):
                        st.markdown(msg["explanation"])

# 3. RIGHT SIDEBAR (AI Activity Timeline)
with activity_col:
    st.markdown("<h4 style='font-size:1.1rem; color:var(--text-color); margin-top:0;'>AI Activity</h4>", unsafe_allow_html=True)
    st.write("")
    flow_placeholder = st.empty()
    
    def render_logs():
        if not st.session_state.workflow_steps:
             flow_placeholder.markdown("<div style='color: gray; font-size: 0.9rem; font-style: italic;'>Awaiting query...</div>", unsafe_allow_html=True)
        else:
             html = "<div>"
             for idx, step in enumerate(st.session_state.workflow_steps):
                 is_last = (idx == len(st.session_state.workflow_steps) - 1)
                 active_cls = "active" if is_last else ""
                 html += f"""
                 <div class='pipeline-step {active_cls}'>
                     <div class='pipeline-title'>{step['title']}</div>
                     <div class='pipeline-desc'>{step['details']}</div>
                 </div>
                 """
             html += "</div>"
             flow_placeholder.markdown(html, unsafe_allow_html=True)

    render_logs()
    
# Bottom Sticky Chat Input executed strictly constrained to Main Workspace
with main_col:
    prompt = st.chat_input("Ask a question...")
    
    # Auto-execute Grid Suggestions structurally identically to pure chat input
    if st.session_state.trigger_prompt:
        prompt = st.session_state.trigger_prompt
        st.session_state.trigger_prompt = None
    
    if prompt:
        st.session_state.messages.append({"role": "user", "text": prompt})
        
        # Timeline Overhauled Data
        st.session_state.workflow_steps = [
            {"title": "User Query", "details": "Evaluating request..."}
        ]
        render_logs()
        
        with st.chat_message("user"):
            st.markdown("<span class='user-msg'></span>" + prompt, unsafe_allow_html=True)
                
        with st.chat_message("model"):
            with st.spinner("Processing..."):
                try:
                    st.session_state.workflow_steps.append({"title": "Model Execution", "details": "Transmitting into network payload..."})
                    render_logs()
                    
                    history_json = json.dumps([{"role": m["role"], "text": m["text"]} for m in st.session_state.messages[:-1]])
                    payload = {
                        "message": prompt,
                        "mode": st.session_state.engine_mode,
                        "specialization": st.session_state.specialization,
                        "history": history_json
                    }
                    
                    res = requests.post("http://127.0.0.1:3001/api/chat", data=payload)
                    if res.status_code == 200:
                        raw_reply = res.json().get("reply", "")
                        
                        st.session_state.workflow_steps.append({"title": "Response Generated", "details": "Generating visualization layers..."})
                        render_logs()
                        
                        # Parse explanations
                        exp_match = re.findall(r'<explanation>([\s\S]*?)<\/explanation>', raw_reply, flags=re.IGNORECASE)
                        explanation = "\n\n".join(exp_match) if exp_match else None
                        raw_reply = re.sub(r'<explanation>[\s\S]*?<\/explanation>', '', raw_reply, flags=re.IGNORECASE).strip()
                        
                        # Parse charts
                        charts = []
                        chart_matches = re.finditer(r'<plotly_chart>([\s\S]*?)<\/plotly_chart>', raw_reply, flags=re.IGNORECASE)
                        for match in chart_matches:
                            try:
                                fig_data = match.group(1).strip()
                                fig = pio.from_json(fig_data)
                                charts.append(fig)
                            except Exception as e:
                                st.error(f"Failed to parse chart: {e}")
                        
                        text_reply = re.sub(r'<plotly_chart>[\s\S]*?<\/plotly_chart>', '', raw_reply, flags=re.IGNORECASE).strip()
                        
                        st.markdown("<span class='model-msg'></span>" + text_reply, unsafe_allow_html=True)
                        for fig in charts:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if explanation:
                            with st.expander("🔍 **Deep Analysis Document**"):
                                st.markdown(explanation)
                                
                        msg_obj = {"role": "model", "text": text_reply}
                        if charts: msg_obj["chart"] = charts[0]
                        if explanation: msg_obj["explanation"] = explanation
                        st.session_state.messages.append(msg_obj)
                        
                        st.session_state.workflow_steps[-1]["title"] = "Completed"
                        st.session_state.workflow_steps[-1]["details"] = "Waiting for next user prompt."
                        render_logs()
                        st.rerun()
                    else:
                        st.error(f"Backend error: {res.status_code}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
