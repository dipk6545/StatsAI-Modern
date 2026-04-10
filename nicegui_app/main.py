"""
StatsAI Frontend UI Layer
Based on the Unified Architecture, but specifically running purely as the NiceGUI Client.
"""

import asyncio, json, re, logging, random
from datetime import datetime
from pathlib import Path
from nicegui import ui, app

import numpy as np
from scipy import stats as sp
import plotly.graph_objects as go

# ── LOGGING ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', filename='nicegui.log')
logger = logging.getLogger("StatsAI_UI")

# ── CHIP → CHART PARAMS MAP (guaranteed correct type, no LLM guessing) ────────
CHIP_PARAMS = {
    "Normal Plot":     {"dist":"normal"},
    "T-Dist Plot":     {"dist":"t"},
    "F-Dist Plot":     {"dist":"f"},
    "Chi-Square":      {"dist":"chi2"},
    "Poisson":         {"dist":"poisson"},
    "Binomial":        {"dist":"binomial"},
    "Exponential":     {"dist":"exponential"},
    "Log-Normal":      {"dist":"lognormal"},
    "Scatter Plot":    {"dist":"scatter"},
    "Box Plot":        {"dist":"box"},
    "Regression Plot": {"dist":"regression"},
    "Z-Curve":         {"dist":"z"},
    "ANOVA Plot":      {"dist":"anova"},
    "Heatmap Grid":    {"dist":"heatmap"},
    "Histogram":       {"dist":"histogram"},
    "Trend Chart":     {"dist":"trend"},
    "Waterfall Plot":  {"dist":"waterfall"},
    "Violin Plot":     {"dist":"violin"},
    "Pareto Chart":    {"dist":"pareto"},
    "Pie Breakout":    {"dist":"pie"},
}

# ── IN-HOUSE CHART ENGINE ──────────────────────────────────────────────────────
BRAND      = '#7c3aed'
BRAND_FILL = 'rgba(124,58,237,0.13)'

def _base_layout(**extra):
    import time
    stamp = str(time.time()).replace('.', '')[-4:]
    return dict(
        template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='ui-sans-serif,system-ui,sans-serif', size=12, color='#374151'),
        margin=dict(t=38, r=22, b=46, l=50),
        xaxis=dict(gridcolor='#f3f4f6', zerolinecolor='#e5e7eb'),
        yaxis=dict(gridcolor='#f3f4f6'),
        annotations=[dict(text=f'Data Render: #{stamp}', x=1, y=-0.12, xref='paper', yref='paper', showarrow=False, font=dict(size=9, color='#9ca3af'))],
        **extra
    )

def _line(x, y, name=''):
    return go.Scatter(x=x, y=y, mode='lines', name=name,
                      line=dict(color=BRAND, width=2.5),
                      fill='tozeroy', fillcolor=BRAND_FILL)

def build_figure(params: dict) -> go.Figure:
    N = 300
    dist = str(params.get('dist', 'normal')).lower().replace('-','').replace(' ','')
    if dist in ('normal','gaussian','bell','normalplot','normaldist'):
        mu, sig = params.get('mu', np.random.randint(-5, 6)), params.get('sigma', round(np.random.uniform(0.5, 3.0), 1))
        x = np.linspace(mu-4*sig, mu+4*sig, N)
        fig = go.Figure(_line(x, sp.norm.pdf(x,mu,sig), f'N({mu},{sig})'))
        rug_x = np.random.normal(mu, sig, 80)
        rug_y = np.random.uniform(-0.02, 0.02, 80)
        fig.add_trace(go.Scatter(x=rug_x, y=rug_y, mode='markers', name='Sample Data', marker=dict(color='#8b5cf6', size=4, opacity=0.7), showlegend=True))
        for s, alpha in ((1,0.25),(2,0.12)):
            xs = x[(x>=mu-s*sig)&(x<=mu+s*sig)]
            fig.add_trace(go.Scatter(x=np.concatenate([[xs[0]],xs,[xs[-1]]]), y=np.concatenate([[0],sp.norm.pdf(xs,mu,sig),[0]]), fill='toself', mode='none', fillcolor=f'rgba(124,58,237,{alpha})', showlegend=False))
        fig.update_layout(**_base_layout(title=dict(text=f'Normal Distribution  μ={mu}  σ={sig}', font_size=13), xaxis_title='z', yaxis_title='Density'))
    elif dist in ('t','tdist','tdistplot','studentt'):
        df = params.get('df', np.random.randint(2, 30))
        x  = np.linspace(-5,5,N)
        fig = go.Figure([_line(x, sp.t.pdf(x,df), f't (df={df})'), go.Scatter(x=x, y=sp.norm.pdf(x), mode='lines', name='Normal', line=dict(color='#9ca3af', width=1.5, dash='dot'), showlegend=True)])
        fig.update_layout(**_base_layout(showlegend=True, title=dict(text=f't-Distribution  df={df}', font_size=13), xaxis_title='t', yaxis_title='Density'))
    elif dist in ('f','fdist','fdistplot'):
        d1,d2 = params.get('df1', np.random.randint(2, 15)), params.get('df2', np.random.randint(10, 40))
        x = np.linspace(0.01,6,N)
        fig = go.Figure(_line(x, sp.f.pdf(x,d1,d2), f'F({d1},{d2})'))
        fig.update_layout(**_base_layout(title=dict(text=f'F-Distribution  df1={d1}  df2={d2}', font_size=13), xaxis_title='F', yaxis_title='Density'))
    elif dist in ('chi2','chisquare','chisq','chi'):
        df = params.get('df', np.random.randint(2, 15))
        x  = np.linspace(0.01, max(df*3,10), N)
        fig = go.Figure(_line(x, sp.chi2.pdf(x,df), f'χ²(df={df})'))
        fig.update_layout(**_base_layout(title=dict(text=f'Chi-Square  df={df}', font_size=13), xaxis_title='χ²', yaxis_title='Density'))
    elif dist in ('exponential','exp'):
        lam = params.get('lambda', round(np.random.uniform(0.5, 2.5), 1))
        x   = np.linspace(0, 6/lam, N)
        fig = go.Figure(_line(x, sp.expon.pdf(x,scale=1/lam), f'Exp(λ={lam})'))
        fig.update_layout(**_base_layout(title=dict(text=f'Exponential  λ={lam}', font_size=13), xaxis_title='x', yaxis_title='Density'))
    elif dist in ('lognormal','lognorm'):
        mu,sig = params.get('mu', round(np.random.uniform(-1, 1), 1)), params.get('sigma', round(np.random.uniform(0.2, 0.8), 2))
        x = np.linspace(0.001, np.exp(mu+4*sig), N)
        fig = go.Figure(_line(x, sp.lognorm.pdf(x,s=sig,scale=np.exp(mu))))
        fig.update_layout(**_base_layout(title=dict(text=f'Log-Normal  μ={mu}  σ={sig}', font_size=13), xaxis_title='x', yaxis_title='Density'))
    elif dist in ('scatter','scatterplot'):
        n,corr = params.get('n', np.random.randint(60, 120)), params.get('corr', round(np.random.uniform(-0.9, 0.9), 2))
        x = np.random.randn(n)
        y = corr*x + np.sqrt(1-corr**2)*np.random.randn(n)
        fig = go.Figure(go.Scatter(x=x, y=y, mode='markers', marker=dict(color=BRAND,size=7,opacity=0.7, line=dict(color='white',width=1))))
        fig.update_layout(**_base_layout(title=dict(text=f'Scatter  ρ≈{corr}', font_size=13), xaxis_title='X', yaxis_title='Y'))
    elif dist in ('box','boxplot'):
        groups = params.get('groups', [f'Group {i+1}' for i in range(np.random.randint(2, 6))])
        traces = [go.Box(y=np.random.normal(i*2,1,50), name=g, marker_color=BRAND, line_color=BRAND, fillcolor=BRAND_FILL) for i,g in enumerate(groups)]
        fig = go.Figure(traces)
        fig.update_layout(**_base_layout(showlegend=False, title=dict(text='Box Plot', font_size=13), yaxis_title='Value'))
    elif dist in ('heatmap','heatmapgrid'):
        n    = params.get('n', np.random.randint(6, 12))
        data = np.random.rand(n,n)
        labs = [f'Axis-{i+1}' for i in range(n)]
        fig  = go.Figure(go.Heatmap(z=data, x=labs, y=labs, colorscale=[[0,'#f5f3ff'],[0.5,BRAND],[1,'#3b0764']]))
        fig.update_layout(**_base_layout(title=dict(text=f'Heatmap {n}x{n}', font_size=13)))
    elif dist in ('pie','piebreakout','piechart'):
        count = np.random.randint(4, 8)
        labels = params.get('labels', [f"Segment-{i+1}" for i in range(count)])
        values = np.random.randint(5, 100, len(labels)).tolist()
        colors = ['#7c3aed','#a78bfa','#6d28d9','#6366f1','#8b5cf6','#c4b5fd','#ddd6fe']
        fig    = go.Figure(go.Pie(labels=labels, values=values, hole=0.35, marker=dict(colors=colors[:len(labels)], line=dict(color='white',width=2))))
        fig.update_layout(**_base_layout(title=dict(text='Distribution Breakout', font_size=13)))
    elif dist in ('trend','trendchart','line'):
        n   = params.get('n',24)
        y   = np.cumsum(np.random.randn(n))+10
        fig = go.Figure(go.Scatter(x=list(range(n)), y=y, mode='lines+markers', fill='tozeroy', fillcolor=BRAND_FILL, line=dict(color=BRAND,width=2.5), marker=dict(color=BRAND,size=5)))
        fig.update_layout(**_base_layout(title=dict(text='Trend Chart', font_size=13), xaxis_title='Period', yaxis_title='Value'))
    else:
        x = np.linspace(-4,4,N)
        fig = go.Figure(_line(x, sp.norm.pdf(x), 'N(0,1)'))
        fig.update_layout(**_base_layout(title=dict(text='Standard Normal Distribution', font_size=13), xaxis_title='z', yaxis_title='Density'))
    return fig

# ── SESSION STORE ──────────────────────────────────────────────────────────────
class S:
    nav='Chat'; subject='Descriptive'; processing=False; multi=False; status_cls=''; model_name='Ready'; cur_sid=None; models=["Mistral Large"]

SESSIONS: dict = {}; SID_ORDER: list = []

# ── UI CONSTANTS ───────────────────────────────────────────────────────────────
MODE_LEFT   = 'Single'; MODE_RIGHT  = 'Multi'; APP_TITLE   = 'StatsAI'; APP_SUBTITLE= 'ANALYST'
BOT_LABEL   = 'STATSAI ANALYST'; USER_LETTER = 'S'; API_URL     = 'http://127.0.0.1:3001/api/chat'; API_TIMEOUT = 90
CHIPS = list(CHIP_PARAMS.keys()); NAV_ITEMS = ['Chat','Reports','Datasets','Gallery']
WELCOME = "Hello! I'm StatsAI — your statistical research assistant. Click a quick action chip or type a question to get started."

def _new_sid():
    import uuid; return str(uuid.uuid4())[:8]
def _today():
    return datetime.now().strftime('%b %d')

def icon_chart(color='#7c3aed'):
    return f'<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M2 12l4-4 3 3 5-7" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
def icon_send():
    return '<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M22 2L11 13" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/><path d="M22 2L15 22 11 13 2 9l20-7z" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'

_EXPL_RE   = re.compile(r'<explanation>(.*?)</explanation>', re.DOTALL|re.IGNORECASE)
_PARAMS_RE = re.compile(r'<chart_params>(.*?)</chart_params>', re.DOTALL|re.IGNORECASE)

def _parse_params(text: str) -> dict | None:
    m = _PARAMS_RE.search(text)
    if not m: return None
    raw = m.group(1).strip()
    try: return json.loads(raw)
    except: return None

def render_bot_block(raw_text: str, container):
    params = _parse_params(raw_text)
    expl_m = _EXPL_RE.search(raw_text)
    content = expl_m.group(1).strip() if expl_m else raw_text
    with container:
        with ui.element('div').classes('flex gap-3 items-start w-full mb-6'):
            with ui.element('div').classes('flex-shrink-0 mt-1'):
                ui.html(f'<div style="width:34px;height:34px;background:linear-gradient(135deg, #7c3aed, #4f46e5);border-radius:10px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(124,58,237,0.2);">{icon_chart("white")}</div>')
            with ui.element('div').classes('flex flex-col gap-1.5 flex-1 min-w-0'):
                with ui.element('div').classes('flex items-center gap-2'):
                    ui.label(BOT_LABEL).classes('text-[11px] text-purple-600 font-black uppercase tracking-[0.2em]')
                    ui.element('div').classes('h-[1px] flex-1 bg-gradient-to-r from-purple-100 to-transparent')
                parts = re.split(r'##\s*', content)
                for part in parts:
                    if not part.strip(): continue
                    lines = part.strip().split('\n', 1)
                    title = lines[0].strip(); body  = lines[1].strip() if len(lines) > 1 else ""
                    with ui.element('div').classes('bg-white border border-gray-100 rounded-2xl shadow-sm overflow-hidden mb-3'):
                        with ui.element('div').classes('bg-gray-50/50 px-4 py-2 border-b border-gray-50'):
                            ui.label(title).classes('text-[13px] font-bold text-gray-700')
                        with ui.element('div').classes('p-4'):
                            chunks = re.split(r'(\$\$.*?\$\$)', body, flags=re.DOTALL)
                            for chunk in chunks:
                                if not chunk.strip(): continue
                                if chunk.startswith('$$'):
                                    with ui.element('div').classes('py-4 flex justify-center bg-purple-50/30 rounded-xl my-2'):
                                        ui.markdown(chunk.strip()).classes('text-lg text-purple-900 math-target')
                                else:
                                    ui.markdown(chunk.strip()).classes('text-[13px] text-gray-600 math-target')
                ui.run_javascript('if(window.renderMathInElement) renderMathInElement(document.body, {delimiters: [{left: "$$", right: "$$", display: true}, {left: "$", right: "$", display: false}]});')
                if params:
                    try:
                        fig = build_figure(params)
                        with ui.element('div').classes('rounded-2xl overflow-hidden border border-purple-100 shadow-lg mt-2'):
                            with ui.element('div').classes('bg-purple-600 px-4 py-2 flex items-center justify-between'):
                                ui.label('LIVE ANALYTICAL PROJECTION').classes('text-[10px] text-white font-bold tracking-widest')
                            ui.plotly(fig.to_dict()).classes('w-full').style('height:350px;background:white;')
                    except Exception as e:
                        ui.label(f'DYNAMICS ERROR: {e}').classes('text-red-400 text-[10px] p-4 bg-red-50 rounded-xl')

def render_user_bubble(text: str, container):
    with container:
        with ui.element('div').style('display:flex;gap:10px;align-items:flex-start;flex-direction:row-reverse;'):
            ui.html(f'<div style="width:28px;height:28px;border-radius:9px;background:#7c3aed;display:flex;align-items:center;justify-content:center;flex-shrink:0;color:white;font-size:11px;font-weight:700;">{USER_LETTER}</div>')
            with ui.element('div').style('display:flex;flex-direction:column;align-items:flex-end;gap:4px;flex:1;min-width:0;'):
                ui.label(text).style('background:#7c3aed;color:white;border-radius:18px;border-top-right-radius:4px;padding:10px 16px;font-size:13px;line-height:1.5;display:block;max-width:88%;word-break:break-word;')

def render_typing(container):
    with container:
        el = ui.element('div').style('display:flex;gap:10px;align-items:flex-start;')
        with el:
            ui.html(f'<div style="width:28px;height:28px;background:#ede9fe;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{icon_chart()}</div>')
            with ui.element('div').style('background:#f3f4f6;border:1px solid #e5e7eb;border-radius:18px;border-top-left-radius:4px;padding:11px 15px;display:flex;gap:5px;align-items:center;'):
                for cls, bg in [('bounce1','#a78bfa'),('bounce2','#7c3aed'),('bounce3','#6d28d9')]:
                    ui.element('div').classes(cls).style(f'width:7px;height:7px;border-radius:50%;background:{bg};')
    return el

@ui.page('/')
async def main_page():
    ui.add_head_html('<link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet"/>')
    ui.add_head_html('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css">')
    ui.add_head_html('<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js"></script>')
    ui.add_head_html('<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js"></script>')
    ui.add_head_html("""<script>document.addEventListener('DOMContentLoaded', () => { const r = () => window.renderMathInElement && renderMathInElement(document.body, {delimiters: [{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError: false}); new MutationObserver(r).observe(document.body, {childList:true, subtree:true}); });</script>""")
    ui.add_head_html('''<style>
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;padding:0;width:100vw;height:100vh;background:#fff;font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;overflow:hidden}
.q-page-container,.q-page,.nicegui-content{padding:0!important;margin:0!important;width:100%!important;height:100%!important;max-width:100%!important}
.bounce1{animation:bdot 1.2s ease-in-out 0s infinite}
.bounce2{animation:bdot 1.2s ease-in-out .2s infinite}
.bounce3{animation:bdot 1.2s ease-in-out .4s infinite}
@keyframes bdot{0%,60%,100%{transform:translateY(0);opacity:.35}30%{transform:translateY(-5px);opacity:1}}
*{-webkit-tap-highlight-color:transparent!important}
.no-select,.q-btn,.nav-item,.send-btn{user-select:none!important;outline:none!important;-webkit-user-select:none!important}
.pulse-green{animation:pgreen 0.4s infinite alternate}
@keyframes pgreen{0%{background:#22c55e;opacity:1}100%{background:#4ade80;opacity:0.3;box-shadow:0 0 8px #22c55e}}
.pulse-red{animation:pred 0.3s infinite alternate}
@keyframes pred{0%{background:#ef4444;opacity:1}100%{background:#fca5a5;opacity:0.4;box-shadow:0 0 10px #ef4444}}
.q-focus-helper,.q-ripple,.q-btn__overlay{display:none!important;visibility:hidden!important;opacity:0!important}
.q-btn:before{box-shadow:none!important}
.q-btn:active{background:transparent!important}
.q-field__control,.q-field__native{background:transparent!important}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#d1d5db;border-radius:99px}
.app-shell{display:flex;width:100vw;height:100vh;overflow:hidden;background:#fff;position:fixed;top:0;left:0}
.sidebar-left{width:clamp(175px,16vw,225px);flex-shrink:0;background:#f9fafb;display:flex;flex-direction:column;height:100%;overflow:hidden;border-right:1px solid #e5e7eb}
.sidebar-right{width:225px;flex-shrink:0;background:#fff;display:flex;flex-direction:column;height:100%;overflow:hidden;border-left:1px solid #e5e7eb}
.chat-center{flex:1;min-width:0;display:flex;flex-direction:column;background:#fff;height:100%;overflow:hidden}
.nav-item{display:flex;align-items:center;gap:8px;padding:7px 10px;border-radius:9px;font-size:12.5px;cursor:pointer;transition:background .15s;border:1px solid transparent;width:100%}
.nav-item.active{background:#fff;border-color:#e5e7eb;font-weight:600;color:#111827}
.nav-item:not(.active){color:#6b7280}
.nav-item:not(.active):hover{background:#f3f4f6}
.recent-item{display:block;padding:8px 10px;border-radius:8px;cursor:pointer;transition:background .15s;width:100%;border:none;background:transparent;text-align:left}
.recent-item.cur{background:#ede9fe}
.slabel{font-size:10px;font-weight:700;color:#9ca3af;letter-spacing:.08em;text-transform:uppercase;display:block}
.send-btn{width:42px!important;height:42px!important;border-radius:12px!important;background:#7c3aed!important;border:none!important;display:flex!important;align-items:center!important;justify-content:center!important;cursor:pointer!important;flex-shrink:0!important;min-height:unset!important;padding:0!important;transition:all .15s cubic-bezier(.4,0,.2,1)!important}
.send-btn:hover{background:#6d28d9!important;transform:translateY(-1px)}
.send-btn:active{transform:scale(.91)!important}
.send-btn .q-focus-helper{display:none!important}
@media(max-width:820px){.sidebar-right{display:none}}
@media(max-width:560px){.sidebar-left{display:none}}
</style>''')

    s = S(); refs = {}

    with ui.element('div').classes('app-shell'):
        with ui.element('div').classes('sidebar-left'):
            with ui.element('div').style('display:flex;align-items:center;gap:10px;padding:15px 12px;border-bottom:1px solid #e5e7eb;flex-shrink:0;'):
                ui.html(f'<div style="width:30px;height:30px;background:#7c3aed;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{icon_chart("white")}</div>')
                with ui.element('div'):
                    ui.label(APP_TITLE).style('font-weight:700;font-size:13px;color:#111827;line-height:1.2;display:block;')
                    ui.label(APP_SUBTITLE).style('font-size:9px;color:#9ca3af;letter-spacing:.14em;font-weight:600;display:block;')
            nav_wrap = ui.element('div').style('padding:11px 8px;flex-shrink:0;')
            @ui.refreshable
            def render_nav():
                nav_wrap.clear()
                with nav_wrap:
                    ui.label('NAVIGATION').classes('slabel').style('padding:0 3px 7px;')
                    for item in NAV_ITEMS:
                        active = item == s.nav
                        with ui.element('div').classes(f'nav-item{"  active" if active else ""}').on('click',lambda it=item: (setattr(s,'nav',it), render_nav.refresh())):
                            ui.label(item).style(f'font-size:12.5px;{"font-weight:600;" if active else ""}')
            render_nav()
            ui.element('div').style('border-top:1px solid #e5e7eb;margin:7px 10px;flex-shrink:0;')
            with ui.element('div').style('padding:0px 12px;'):
                with ui.element('div').classes('w-full relative flex p-1 bg-gray-200/50 rounded-full no-select overflow-hidden').style('user-select:none;height:auto;'):
                    pill = ui.element('div').style('position:absolute;top:2px;bottom:2px;left:2px;width:calc(50% - 2px);background:white;border-radius:999px;box-shadow:0 1px 3px rgba(0,0,0,.1);transition:transform .3s cubic-bezier(.4,0,.2,1);transform:translateX(0);pointer-events:none;')
                    def _tog(val):
                        s.multi = val
                        pill.style(f'transform:translateX({"calc(100% + 1px)" if val else "0"});')
                        for l in [lbl1, lbl2]: l.classes(remove='text-purple-600 text-gray-500 font-bold font-medium text-[12px] text-[13px] scale-105 scale-95')
                        if val:
                            lbl1.classes('text-gray-500 font-medium text-[12px] scale-95 transition-transform duration-300')
                            lbl2.classes('text-purple-600 font-bold text-[13px] scale-105 transition-transform duration-300')
                        else:
                            lbl1.classes('text-purple-600 font-bold text-[13px] scale-105 transition-transform duration-300')
                            lbl2.classes('text-gray-500 font-medium text-[12px] scale-95 transition-transform duration-300')
                    
                    with ui.button(on_click=lambda:_tog(False)).classes('relative z-10 flex-1 h-full p-0 flex items-center justify-center').props('flat no-caps no-ripple'):
                        lbl1 = ui.label(MODE_LEFT).classes('w-full text-center leading-none no-select text-purple-600 font-bold text-[13px] mt-[1.5px] scale-105 transition-transform duration-300')
                    with ui.button(on_click=lambda:_tog(True)).classes('relative z-10 flex-1 h-full p-0 flex items-center justify-center').props('flat no-caps no-ripple'):
                        lbl2 = ui.label(MODE_RIGHT).classes('w-full text-center leading-none no-select text-gray-500 font-medium text-[12px] mt-[1px] scale-95 transition-transform duration-300')
            ui.element('div').style('border-top:1px solid #e5e7eb;margin:7px 10px;flex-shrink:0;')
            ui.label('RECENT').classes('slabel').style('padding:0 12px 6px;')
            recents_wrap = ui.element('div').style('flex:1;min-height:0;overflow-y:auto;padding:0 6px;')
            @ui.refreshable
            def render_recents():
                recents_wrap.clear()
                with recents_wrap:
                    for sid in SID_ORDER:
                        sess = SESSIONS.get(sid)
                        with ui.element('div').classes(f'recent-item{"  cur" if sid==s.cur_sid else ""}').on('click',lambda sv=sid: _load_session(sv)):
                            ui.label(sess['title']).style('font-size:12px;font-weight:500;color:#374151;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;')
            render_recents(); refs['render_recents']=render_recents

        with ui.element('div').classes('chat-center'):
            with ui.element('div').style('display:flex;align-items:center;gap:12px;padding:12px 18px;border-bottom:1px solid #e5e7eb;flex-shrink:0;background:#fff;'):
                ui.html(f'<div style="width:30px;height:30px;background:#ede9fe;border-radius:8px;display:flex;align-items:center;justify-center;flex-shrink:0;">{icon_chart()}</div>')
                ui.label('StatsAI Analyst').style('font-size:14px;font-weight:700;color:#111827;')
            scroll = ui.scroll_area().style('flex:1;background:#fff;')
            refs['scroll'] = scroll
            with scroll:
                mc = ui.element('div').style('display:flex;flex-direction:column;gap:18px;padding:18px;width:100%;')
                refs['mc'] = mc
                render_bot_block(WELCOME, mc)
            with ui.element('div').style('padding:10px 14px 13px;border-top:1px solid #f3f4f6;flex-shrink:0;background:#fff;'):
                with ui.element('div').style('display:flex;align-items:center;gap:10px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:999px;padding:4px 5px 4px 16px;'):
                    inp = ui.input(placeholder='Message StatsAI...').props('borderless dense').style('flex:1;font-size:13px;color:#111827;')
                    refs['inp'] = inp
                    async def send_message():
                        val = inp.value.strip()
                        if not val or s.processing: return
                        inp.value = ''; s.processing = True
                        if not s.cur_sid:
                            sid = _new_sid(); s.cur_sid=sid; SESSIONS[sid] = {'title': val[:40], 'date': _today(), 'messages': []}
                            SID_ORDER.insert(0, sid); render_recents.refresh()
                        mc_ref=refs['mc']; sa_ref=refs['scroll']; render_user_bubble(val, mc_ref); sa_ref.scroll_to(percent=1.0)
                        typing=render_typing(mc_ref); sa_ref.scroll_to(percent=1.0)
                        import requests as req
                        stack = S.models if S.models else ["Mistral Large"]
                        r = random.randint(0, 8); start = (r // len(stack)) % len(stack)
                        final_stack = stack[start:] + stack[:start]; success = False
                        for i, m_name in enumerate(final_stack):
                            try:
                                s.model_name = f'Hitting {m_name}...'
                                try:
                                    refs['status_led'].classes('pulse-green', remove='bg-gray-300 bg-green-500 pulse-red')
                                    refs['model_lbl'].text = s.model_name
                                except: pass
                                payload = {'message': val, 'model_id': m_name, 'domain': s.subject.lower(), 'history': json.dumps([{'role':m['role'],'text':m['text']} for m in SESSIONS[s.cur_sid]['messages']])}
                                res = await asyncio.get_event_loop().run_in_executor(None, lambda: req.post(API_URL, data=payload, timeout=API_TIMEOUT))
                                if res.status_code == 200:
                                    reply = res.json().get('reply',''); s.model_name = m_name; s.status_cls='success'
                                    try:
                                        refs['status_led'].classes('bg-green-500', remove='pulse-green pulse-red')
                                        refs['model_lbl'].text = s.model_name
                                        typing.delete()
                                    except: pass
                                    render_bot_block(reply, mc_ref); SESSIONS[s.cur_sid]['messages'].extend([{'role':'user','text':val},{'role':'bot','text':reply}])
                                    success = True; break
                                else: raise Exception(f"HTTP {res.status_code}")
                            except Exception as exc:
                                next_m = final_stack[i+1] if i+1 < len(final_stack) else None
                                s.model_name = f'Failover: {next_m}...' if next_m else 'Offline'
                                try:
                                    refs['status_led'].classes('pulse-red', remove='pulse-green')
                                    refs['model_lbl'].text = s.model_name
                                except: pass
                                if next_m: await asyncio.sleep(1.5)
                        if not success:
                            try: typing.delete()
                            except: pass
                            with mc_ref: ui.label('Critically Offline').style('color:#ef4444;font-size:12px;font-style:italic;')
                        s.processing = False; sa_ref.scroll_to(percent=1.0)
                    inp.on('keydown.enter', send_message)
                    with ui.button(on_click=send_message).classes('send-btn').props('flat'):
                        ui.html(icon_send())

        with ui.element('div').classes('sidebar-right'):
            with ui.scroll_area().style('flex:1;min-height:0;'):
                with ui.element('div').style('padding:20px 10px;'):
                    ui.label('MODEL STATUS').classes('slabel').style('margin-bottom:15px;')
                    with ui.element('div').classes('bg-gray-50 rounded-xl p-4 border border-gray-100 flex flex-col gap-3'):
                        with ui.element('div').classes('flex items-center gap-3'):
                            refs['status_led'] = ui.element('div').classes('w-2.5 h-2.5 rounded-full bg-gray-300')
                            ui.label('ENGINE STATUS').classes('text-[10px] font-black text-gray-400 tracking-widest')
                        refs['model_lbl'] = ui.label('Ready').classes('text-[12px] font-black text-purple-700 truncate')
            ui.label('StatsAI v2.5').classes('text-[10px] text-gray-300 font-bold uppercase text-center w-full block p-4')

    def _load_session(sid):
        sess = SESSIONS.get(sid); s.cur_sid = sid; mc_ref = refs['mc']; mc_ref.clear()
        for msg in sess['messages']: (render_user_bubble(msg['text'], mc_ref) if msg['role']=='user' else render_bot_block(msg['text'], mc_ref))
        render_recents.refresh(); refs['scroll'].scroll_to(percent=1.0)

async def _startup_sync():
    import requests as req
    try:
        await asyncio.sleep(2)
        res = await asyncio.get_event_loop().run_in_executor(None, lambda: req.get('http://127.0.0.1:3001/api/config', timeout=5))
        if res.status_code == 200: S.models = res.json().get('models', ["Mistral Large"])
    except: pass

if __name__ in {'__main__', '__mp_main__'}:
    app.on_startup(_startup_sync)
    ui.run(title='StatsAI Analyst', port=8080, show=False, reload=False)