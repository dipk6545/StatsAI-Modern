"""
StatsAI Frontend UI Layer
Based on the Unified Architecture, but specifically running purely as the NiceGUI Client.
"""

import asyncio, json, re, logging
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
    # No random seed - allows natural dynamic variance!
    logger.info(f"Generating data for {dist}. Random check: {np.random.rand()}")

    if dist in ('normal','gaussian','bell','normalplot','normaldist'):
        mu, sig = params.get('mu', np.random.randint(-5, 6)), params.get('sigma', round(np.random.uniform(0.5, 3.0), 1))
        x = np.linspace(mu-4*sig, mu+4*sig, N)
        fig = go.Figure(_line(x, sp.norm.pdf(x,mu,sig), f'N({mu},{sig})'))
        
        # Add a stochastic 'rug' noise scatter to visually prove dynamic generation
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

    elif dist == 'beta':
        a,b = params.get('alpha', np.random.randint(2, 8)), params.get('beta', np.random.randint(2, 8))
        x   = np.linspace(0,1,N)
        fig = go.Figure(_line(x, sp.beta.pdf(x,a,b)))
        fig.update_layout(**_base_layout(title=dict(text=f'Beta  α={a}  β={b}', font_size=13), xaxis_title='x', yaxis_title='Density'))

    elif dist == 'gamma':
        k,th = params.get('k', round(np.random.uniform(1.5, 4.0), 1)), params.get('theta', round(np.random.uniform(1.0, 3.0), 1))
        x    = np.linspace(0, k*th*4, N)
        fig  = go.Figure(_line(x, sp.gamma.pdf(x,a=k,scale=th)))
        fig.update_layout(**_base_layout(title=dict(text=f'Gamma  k={k}  θ={th}', font_size=13), xaxis_title='x', yaxis_title='Density'))

    elif dist == 'uniform':
        a = params.get('a', np.random.randint(-5, 0))
        b = params.get('b', np.random.randint(1, 5))
        x   = np.linspace(a-0.5, b+0.5, N)
        fig = go.Figure(go.Scatter(x=x, y=sp.uniform.pdf(x,loc=a,scale=b-a), mode='lines', line=dict(color=BRAND, width=2.5)))
        fig.update_layout(**_base_layout(title=dict(text=f'Uniform [{a},{b}]', font_size=13), xaxis_title='x', yaxis_title='Density'))

    elif dist in ('z','zcurve','zscore','zplot'):
        z   = params.get('z', round(np.random.uniform(1.0, 3.0), 2))
        x   = np.linspace(-4,4,N)
        y   = sp.norm.pdf(x)
        xs  = x[x>=z]
        p   = round(1-sp.norm.cdf(z),4)
        fig = go.Figure([go.Scatter(x=x, y=y, mode='lines', fill='tozeroy', fillcolor=BRAND_FILL, line=dict(color=BRAND,width=2.5), name='N(0,1)'), go.Scatter(x=np.r_[xs[0],xs,xs[-1]], y=np.r_[0,sp.norm.pdf(xs),0], fill='toself', mode='none', fillcolor='rgba(239,68,68,0.22)', name=f'p={p}')])
        fig.update_layout(**_base_layout(showlegend=True, title=dict(text=f'Z-Curve  z={z}  p={p}', font_size=13), xaxis_title='z', yaxis_title='Density'))

    elif dist == 'poisson':
        lam   = params.get('lambda', np.random.randint(2, 12))
        k     = np.arange(0, max(int(lam*3),15)+1)
        fig   = go.Figure(go.Bar(x=k, y=sp.poisson.pmf(k,lam), marker_color=BRAND, marker_line_color='white', marker_line_width=1.5))
        fig.update_layout(**_base_layout(title=dict(text=f'Poisson  λ={lam}', font_size=13), xaxis_title='k', yaxis_title='P(X=k)', bargap=0.2))

    elif dist in ('binomial','binom'):
        n,p = params.get('n', np.random.randint(10, 40)), params.get('p', round(np.random.uniform(0.2, 0.8), 2))
        k   = np.arange(0,n+1)
        fig = go.Figure(go.Bar(x=k, y=sp.binom.pmf(k,n,p), marker_color=BRAND, marker_line_color='white', marker_line_width=1.5))
        fig.update_layout(**_base_layout(title=dict(text=f'Binomial  n={n}  p={p}', font_size=13), xaxis_title='k', yaxis_title='P(X=k)', bargap=0.2))

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

    elif dist in ('violin','violinplot'):
        groups = params.get('groups', [f'Set {i+1}' for i in range(np.random.randint(2, 5))])
        traces = [go.Violin(y=np.random.normal(i,1,80), name=g, fillcolor=BRAND_FILL, line_color=BRAND, box_visible=True, meanline_visible=True) for i,g in enumerate(groups)]
        fig = go.Figure(traces)
        fig.update_layout(**_base_layout(showlegend=False, title=dict(text='Violin Plot', font_size=13), yaxis_title='Value'))

    elif dist in ('histogram','hist'):
        mu,sig = params.get('mu', np.random.randint(-10, 10)), params.get('sigma', round(np.random.uniform(0.5, 5.0), 1))
        data   = np.random.normal(mu,sig,params.get('n',300))
        xp     = np.linspace(data.min(),data.max(),200)
        fig    = go.Figure([go.Histogram(x=data, nbinsx=30, histnorm='probability density', marker=dict(color=BRAND,line=dict(color='white',width=0.5)), name='Histogram'), go.Scatter(x=xp, y=sp.norm.pdf(xp,mu,sig), mode='lines', line=dict(color='#374151',width=1.5,dash='dot'), name='PDF')])
        fig.update_layout(**_base_layout(showlegend=True, title=dict(text=f'Histogram  μ={mu}  σ={sig}', font_size=13), xaxis_title='x', yaxis_title='Density'))

    elif dist in ('regression','regplot','regressionplot'):
        n       = params.get('n',60)
        x       = np.linspace(0,10,n)
        y       = params.get('slope', round(np.random.uniform(-2, 3), 1))*x + params.get('intercept', np.random.randint(-5, 5)) + np.random.randn(n)*2
        m,b,r,*_= sp.linregress(x,y)
        fig     = go.Figure([go.Scatter(x=x, y=y, mode='markers', marker=dict(color=BRAND,size=7,opacity=0.65, line=dict(color='white',width=1))), go.Scatter(x=x, y=m*x+b, mode='lines', line=dict(color='#dc2626',width=2), name=f'y={m:.2f}x+{b:.2f}  r²={r**2:.3f}')])
        fig.update_layout(**_base_layout(showlegend=True, title=dict(text=f'Regression  r²={r**2:.3f}', font_size=13), xaxis_title='X', yaxis_title='Y'))

    elif dist in ('heatmap','heatmapgrid'):
        n    = params.get('n', np.random.randint(6, 12))
        data = np.random.rand(n,n)
        labs = [f'Axis-{i+1}' for i in range(n)]
        fig  = go.Figure(go.Heatmap(z=data, x=labs, y=labs, colorscale=[[0,'#f5f3ff'],[0.5,BRAND],[1,'#3b0764']]))
        fig.update_layout(**_base_layout(title=dict(text=f'Heatmap {n}x{n}', font_size=13)))

    elif dist in ('anova','anovaplot'):
        groups = params.get('groups', [f'G{i+1}' for i in range(np.random.randint(3, 7))])
        colors = ['#7c3aed','#a78bfa','#6d28d9','#c4b5fd']
        traces = [go.Box(y=np.random.normal(i*1.5,1,30), name=g, marker_color=colors[i%4], boxmean=True) for i,g in enumerate(groups)]
        fig = go.Figure(traces)
        fig.update_layout(**_base_layout(showlegend=False, title=dict(text='ANOVA Group Comparison', font_size=13), yaxis_title='Value'))

    elif dist in ('pareto','paretochart'):
        count = np.random.randint(5, 9)
        cats = params.get('categories', [f'Item-{chr(i+65)}' for i in range(count)])
        vals = np.sort(np.random.randint(10,100,len(cats)))[::-1].tolist()
        cum  = (np.cumsum(vals)/sum(vals)*100).tolist()
        fig  = go.Figure([go.Bar(x=cats, y=vals, marker_color=BRAND, name='Count'), go.Scatter(x=cats, y=cum, yaxis='y2', mode='lines+markers', line=dict(color='#dc2626',width=2), name='Cumulative %')])
        fig.update_layout(**_base_layout(showlegend=True, title=dict(text='Pareto Analysis', font_size=13), yaxis_title='Count', yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0,110], showgrid=False, ticksuffix='%')))

    elif dist in ('waterfall','waterfallplot'):
        count = np.random.randint(4, 8)
        cats = params.get('categories', ['Start'] + [f'Q{i+1}' for i in range(count-2)] + ['End'])
        vals = params.get('values', [np.random.randint(500, 1500)] + np.random.randint(-300, 400, count-2).tolist() + [0])
        measure = ['absolute']+['relative']*(len(vals)-2)+['total']
        fig = go.Figure(go.Waterfall(x=cats, y=vals, measure=measure, increasing_marker_color=BRAND, decreasing_marker_color='#dc2626', totals_marker_color='#374151', connector_line_color='#e5e7eb'))
        fig.update_layout(**_base_layout(title=dict(text='Waterfall Chart', font_size=13), yaxis_title='Value'))

    elif dist in ('pie','piebreakout','piechart'):
        count = np.random.randint(4, 8)
        labels = params.get('labels', [f"Segment-{i+1}" for i in range(count)])
        # FORCE randomization of values every time
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

# ── UI CONSTANTS ───────────────────────────────────────────────────────────────
MODE_LEFT   = 'Single'
MODE_RIGHT  = 'Multi'
APP_TITLE   = 'StatsAI'
APP_SUBTITLE= 'ANALYST'
BOT_LABEL   = 'STATSAI ANALYST'
USER_LETTER = 'S'
API_URL     = 'http://127.0.0.1:3001/api/chat'
API_TIMEOUT = 90

CHIPS = list(CHIP_PARAMS.keys())
NAV_ITEMS = ['Chat','Reports','Datasets','Gallery']

PIPELINE_STEPS = [
    'Agent Initialized','Mode Detection','Model Selected',
    'Groq LPU Request','Params Extracted','Chart Generated',
    'UI Rendered','Response Out',
]
WELCOME = "Hello! I'm StatsAI — your statistical research assistant. Click a quick action chip or type a question to get started."

# ── SESSION STORE ──────────────────────────────────────────────────────────────
SESSIONS: dict = {}
SID_ORDER: list = []

def _new_sid():
    import uuid; return str(uuid.uuid4())[:8]

def _today():
    return datetime.now().strftime('%b %d')

# ── SVG ICONS ──────────────────────────────────────────────────────────────────
def icon_chart(color='#7c3aed'):
    return f'<svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M2 12l4-4 3 3 5-7" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
def icon_check():
    return '<svg width="7" height="7" viewBox="0 0 7 7" fill="none"><path d="M1.5 3.5l1.5 1.5 2.5-3" stroke="white" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
def icon_send():
    return '<svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M22 2L11 13" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/><path d="M22 2L15 22 11 13 2 9l20-7z" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'

# ── RENDER HELPERS ─────────────────────────────────────────────────────────────
_EXPL_RE   = re.compile(r'<explanation>(.*?)</explanation>', re.DOTALL|re.IGNORECASE)
_PARAMS_RE = re.compile(r'<chart_params>(.*?)</chart_params>', re.DOTALL|re.IGNORECASE)

def _parse_params(text: str) -> dict | None:
    m = _PARAMS_RE.search(text)
    if not m: return None
    raw = m.group(1).strip()
    for attempt in (raw, re.sub(r',\s*([}\]])', r'\1', raw)):
        try: return json.loads(attempt)
        except: pass
    return None

def render_bot_block(raw_text: str, container):
    expl_m   = _EXPL_RE.search(raw_text)
    params   = _parse_params(raw_text)
    plain    = re.sub(r'<(?:explanation|chart_params|plotly_table)>.*?</(?:explanation|chart_params|plotly_table)>', '', raw_text, flags=re.DOTALL).strip()
    plain    = re.sub(r'<[^>]+>', '', plain).strip()

    with container:
        with ui.element('div').classes('flex gap-2.5 items-start w-full'):
            ui.html(f'<div style="width:28px;height:28px;background:#ede9fe;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{icon_chart()}</div>')
            with ui.element('div').classes('flex flex-col gap-2 flex-1 min-w-0'):
                with ui.element('div').classes('flex items-center gap-1.5 mb-0.5'):
                    ui.html(f'<span style="display:inline-flex;align-items:center;justify-content:center;width:12px;height:12px;border-radius:50%;background:#7c3aed;flex-shrink:0;">{icon_check()}</span>')
                    ui.label(BOT_LABEL).classes('text-[10px] text-gray-400 font-bold uppercase tracking-wider')

                if expl_m:
                    ui.markdown(expl_m.group(1).strip()).classes('bg-gray-100 border border-gray-200 rounded-2xl rounded-tl-sm p-4 text-[13px] text-gray-800 leading-relaxed w-full')
                elif plain:
                    ui.markdown(plain).classes('bg-gray-100 border border-gray-200 rounded-2xl rounded-tl-sm p-4 text-[13px] text-gray-800 leading-relaxed w-full')

                if params:
                    try:
                        fig = build_figure(params)
                        with ui.element('div').style('width:100%;border-radius:12px;overflow:hidden;border:1px solid #f3f4f6;margin-top:2px;'):
                            ui.plotly(fig.to_dict()).classes('w-full').style('height:340px;')
                    except Exception as e:
                        ui.label(f'Chart error: {e}').classes('text-red-400 text-xs italic mt-1')

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

# ── MAIN PAGE ──────────────────────────────────────────────────────────────────
@ui.page('/')
async def main_page():
    ui.add_head_html('<link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet"/>')
    ui.add_head_html('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css">')
    ui.add_head_html('<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.js"></script>')
    ui.add_head_html('<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>')
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
.q-focus-helper,.q-ripple,.q-btn__overlay{display:none!important;visibility:hidden!important;opacity:0!important}
.q-btn:before{box-shadow:none!important}
.q-btn:active{background:transparent!important}
@keyframes pipe-pulse{0%,100%{transform:scale(1);opacity:1;box-shadow:0 0 0 rgba(124,58,237,0)}50%{transform:scale(1.3);opacity:.7;box-shadow:0 0 12px rgba(124,58,237,.5)}}
.pipe-active-pulse{animation:pipe-pulse 1.2s ease-in-out infinite}
.q-field__control,.q-field__native{background:transparent!important}
.q-field--outlined .q-field__control:before{border:none!important}
.q-field--outlined .q-field__control:after{display:none!important}
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#d1d5db;border-radius:99px}
::-webkit-scrollbar-thumb:hover{background:#9ca3af}
.app-shell{display:flex;width:100vw;height:100vh;overflow:hidden;background:#fff;position:fixed;top:0;left:0}
.sidebar-left{width:clamp(175px,16vw,225px);flex-shrink:0;background:#f9fafb;display:flex;flex-direction:column;height:100%;overflow:hidden;border-right:1px solid #e5e7eb}
.sidebar-right{width:450px;flex-shrink:0;background:#fff;display:flex;flex-direction:column;height:100%;overflow:hidden;border-left:1px solid #e5e7eb}
.chat-center{flex:1;min-width:0;display:flex;flex-direction:column;background:#fff;height:100%;overflow:hidden}
.nav-item{display:flex;align-items:center;gap:8px;padding:7px 10px;border-radius:9px;font-size:12.5px;cursor:pointer;transition:background .15s;border:1px solid transparent;width:100%}
.nav-item.active{background:#fff;border-color:#e5e7eb;font-weight:600;color:#111827}
.nav-item:not(.active){color:#6b7280}
.nav-item:not(.active):hover{background:#f3f4f6}
.recent-item{display:block;padding:8px 10px;border-radius:8px;cursor:pointer;transition:background .15s;width:100%;border:none;background:transparent;text-align:left}
.recent-item:hover{background:#f3f4f6}
.recent-item.cur{background:#ede9fe}
.slabel{font-size:10px;font-weight:700;color:#9ca3af;letter-spacing:.08em;text-transform:uppercase;display:block}
.action-bubble{display:inline-block;background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:3px 9px;margin-right:2px;margin-bottom:4px;font-size:10px;color:#4b5563;font-weight:600;cursor:pointer;transition:all .15s cubic-bezier(.4,0,.2,1);box-shadow:0 1px 2px rgba(0,0,0,.03)}
.action-bubble:hover{border-color:#7c3aed;background:#f5f3ff;color:#7c3aed;transform:translateY(-1px);box-shadow:0 3px 5px -1px rgba(124,58,237,.08)}
.action-bubble:active{transform:scale(.96)}
.send-btn{width:42px!important;height:42px!important;border-radius:12px!important;background:#7c3aed!important;border:none!important;display:flex!important;align-items:center!important;justify-content:center!important;cursor:pointer!important;flex-shrink:0!important;min-height:unset!important;padding:0!important;transition:all .15s cubic-bezier(.4,0,.2,1)!important}
.send-btn:hover{background:#6d28d9!important;transform:translateY(-1px)}
.send-btn:active{transform:scale(.91)!important}
.send-btn .q-focus-helper{display:none!important}
@media(max-width:820px){.sidebar-right{display:none}}
@media(max-width:560px){.sidebar-left{display:none}}
    </style>''')

    class S:
        nav        = 'Chat'
        subject    = 'Descriptive'
        processing = False
        multi      = False
        cur_sid    = None
        pipe_step  = -1
        pipe_vis   = False
    s = S()
    refs = {}

    with ui.element('div').classes('app-shell'):
        # LEFT SIDEBAR
        with ui.element('div').classes('sidebar-left'):
            with ui.element('div').style('display:flex;align-items:center;gap:10px;padding:15px 12px 11px;border-bottom:1px solid #e5e7eb;flex-shrink:0;'):
                ui.html(f'<div style="width:30px;height:30px;background:#7c3aed;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{icon_chart("white")}</div>')
                with ui.element('div'):
                    ui.label(APP_TITLE).style('font-weight:700;font-size:13px;color:#111827;line-height:1.2;display:block;')
                    ui.label(APP_SUBTITLE).style('font-size:9px;color:#9ca3af;letter-spacing:.14em;font-weight:600;display:block;')

            nav_wrap = ui.element('div').style('padding:11px 8px 3px;flex-shrink:0;')
            @ui.refreshable
            def render_nav():
                nav_wrap.clear()
                with nav_wrap:
                    ui.label('NAVIGATION').classes('slabel').style('padding:0 3px 7px;')
                    for item in NAV_ITEMS:
                        active = item == s.nav
                        def _click(it=item):
                            s.nav = it; render_nav.refresh()
                        with ui.element('div').classes(f'nav-item{"  active" if active else ""}').on('click',_click):
                            ui.element('span').style(f'width:6px;height:6px;border-radius:50%;flex-shrink:0;background:{"#7c3aed" if active else "#d1d5db"};')
                            ui.label(item).style(f'font-size:12.5px;{"font-weight:600;" if active else ""}')
            render_nav()

            ui.element('div').style('border-top:1px solid #e5e7eb;margin:7px 10px;flex-shrink:0;')

            # Toggle
            with ui.element('div').style('padding:0px 12px;'):
                with ui.element('div').classes('w-full relative flex p-1 bg-gray-200/50 rounded-full no-select overflow-hidden').style('user-select:none;height:auto;'):
                    pill = ui.element('div').style('position:absolute;top:2px;bottom:2px;left:2px;width:calc(50% - 2px);background:white;border-radius:999px;box-shadow:0 1px 3px rgba(0,0,0,.1);transition:transform .3s cubic-bezier(.4,0,.2,1);transform:translateX(0);pointer-events:none;')
                    def _tog(val):
                        s.multi = val
                        pill.style(f'transform:translateX({"calc(100% + 1px)" if val else "0"});')
                        lbl1.classes(remove='text-purple-600 text-gray-500 font-bold font-medium text-[12px] text-[13px]')
                        lbl1.classes(f'{"text-gray-500 font-medium text-[12px]" if val else "text-purple-600 font-bold text-[13px]"}')
                        lbl2.classes(remove='text-purple-600 text-gray-500 font-bold font-medium text-[12px] text-[13px]')
                        lbl2.classes(f'{"text-purple-600 font-bold text-[13px]" if val else "text-gray-500 font-medium text-[12px]"}')
                    
                    with ui.button(on_click=lambda:_tog(False)).classes('relative z-10 flex-1 h-full p-0 flex items-center justify-center').props('flat no-caps no-ripple'):
                        lbl1 = ui.label(MODE_LEFT).classes('w-full text-center leading-none no-select text-purple-600 font-bold text-[13px] mt-[1.5px]')
                    with ui.button(on_click=lambda:_tog(True)).classes('relative z-10 flex-1 h-full p-0 flex items-center justify-center').props('flat no-caps no-ripple'):
                        lbl2 = ui.label(MODE_RIGHT).classes('w-full text-center leading-none no-select text-gray-500 font-medium text-[12px] mt-[1px]')

            ui.element('div').style('border-top:1px solid #e5e7eb;margin:7px 10px;flex-shrink:0;')
            ui.label('RECENT').classes('slabel').style('padding:0 12px 6px;flex-shrink:0;')

            recents_wrap = ui.element('div').style('flex:1;min-height:0;overflow-y:auto;padding:0 6px;')
            @ui.refreshable
            def render_recents():
                recents_wrap.clear()
                with recents_wrap:
                    if not SID_ORDER:
                        ui.label('No recent chats').style('font-size:11px;color:#d1d5db;padding:8px 10px;display:block;')
                    for sid in SID_ORDER:
                        sess = SESSIONS.get(sid)
                        if not sess: continue
                        def _load(sv=sid): _load_session(sv)
                        with ui.element('div').classes(f'recent-item{"  cur" if sid==s.cur_sid else ""}').on('click',_load).style('width:100%;'):
                            ui.label(sess['title']).style('font-size:12px;font-weight:500;color:#374151;display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;')
                            ui.label(sess['date']).style('font-size:10px;color:#9ca3af;display:block;margin-top:1px;')
            render_recents()
            refs['render_recents'] = render_recents

            with ui.element('div').style('padding:8px 8px 11px;border-top:1px solid #e5e7eb;flex-shrink:0;'):
                def _new_chat():
                    _save_session()
                    s.cur_sid = None; s.processing = False
                    mc = refs['mc']; mc.clear()
                    render_bot_block(WELCOME, mc)
                    render_recents.refresh()
                ui.button('+ New chat', on_click=_new_chat).style('width:100%;background:#f9fafb;border:1px solid #e5e7eb;border-radius:9px;color:#374151;font-size:12px;font-weight:600;padding:8px 0;cursor:pointer;').props('flat no-caps')

        # CENTER CHAT
        with ui.element('div').classes('chat-center'):
            with ui.element('div').style('display:flex;align-items:center;gap:12px;padding:12px 18px;border-bottom:1px solid #e5e7eb;flex-shrink:0;background:#fff;'):
                ui.html(f'<div style="width:30px;height:30px;background:#ede9fe;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">{icon_chart()}</div>')
                with ui.element('div').style('flex:1;min-width:0;'):
                    ui.label('StatsAI Analyst').style('font-size:14px;font-weight:700;color:#111827;display:block;line-height:1.2;')
                    with ui.element('div').style('display:flex;align-items:center;gap:5px;margin-top:2px;'):
                        ui.element('span').style('width:6px;height:6px;border-radius:50%;background:#22c55e;flex-shrink:0;')
                        ui.label('Online').style('font-size:11px;color:#6b7280;')
                ui.label('Verified').style('font-size:10px;font-weight:700;color:#7c3aed;background:#ede9fe;padding:2px 8px;border-radius:999px;flex-shrink:0;')

            scroll = ui.scroll_area().style('flex:1;min-height:0;background:#fff;')
            refs['scroll'] = scroll
            with scroll:
                mc = ui.element('div').style('display:flex;flex-direction:column;gap:18px;padding:18px;width:100%;')
                refs['mc'] = mc
                render_bot_block(WELCOME, mc)

            with ui.element('div').style('padding:10px 14px 13px;border-top:1px solid #f3f4f6;flex-shrink:0;background:#fff;'):
                with ui.element('div').style('display:flex;align-items:center;gap:10px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:999px;padding:4px 5px 4px 16px;'):
                    inp = ui.input(placeholder='Message StatsAI...').props('borderless dense').style('flex:1;font-size:13px;color:#111827;background:transparent;min-width:0;')
                    refs['inp'] = inp

                    async def send_message(forced=None):
                        val = (forced or inp.value).strip()
                        if not val or s.processing: return
                        inp.value = ''
                        s.processing = True
                        s.pipe_vis   = True
                        s.pipe_step  = -1
                        refs['render_pipeline'].refresh()

                        if s.cur_sid is None:
                            sid = _new_sid()
                            s.cur_sid = sid
                            SESSIONS[sid] = {'title': val[:40]+('…' if len(val)>40 else ''), 'date': _today(), 'messages': []}
                            SID_ORDER.insert(0, sid)
                            render_recents.refresh()

                        mc_ref = refs['mc']; sa_ref = refs['scroll']
                        render_user_bubble(val, mc_ref)
                        sa_ref.scroll_to(percent=1.0)
                        typing = render_typing(mc_ref)
                        sa_ref.scroll_to(percent=1.0)

                        async def _anim():
                            for i in range(len(PIPELINE_STEPS)):
                                s.pipe_step = i
                                refs['render_pipeline'].refresh()
                                await asyncio.sleep(0.45)
                        asyncio.create_task(_anim())

                        chip_params = CHIP_PARAMS.get(val)
                        if chip_params:
                            await asyncio.sleep(0.5)
                            try: typing.delete()
                            except: pass
                            expl_map = {
                                "Normal Plot": "Standard normal distribution N(0,1) — the classic bell curve centred at zero.",
                                "T-Dist Plot": "Student's t-distribution with 10 degrees of freedom, heavier tails than normal.",
                                "F-Dist Plot": "F-distribution with df1=5, df2=20 — used in ANOVA and regression tests.",
                                "Chi-Square":  "Chi-square distribution with 4 degrees of freedom — used in goodness-of-fit tests.",
                                "Poisson":     "Poisson distribution (λ=4) — models counts of rare events in a fixed interval.",
                                "Binomial":    "Binomial distribution (n=20, p=0.5) — models number of successes in 20 trials.",
                                "Exponential": "Exponential distribution (λ=1) — models time between Poisson events.",
                                "Log-Normal":  "Log-normal distribution — positively skewed, values are exponents of a normal variable.",
                                "Scatter Plot":"Scatter plot with correlation ρ≈0.7 showing a positive linear relationship.",
                                "Box Plot":    "Box plots for three groups showing median, IQR, and outliers.",
                                "Regression Plot":"Simple linear regression with fitted line and r² statistic.",
                                "Z-Curve":     "Standard normal with critical region shaded for z=1.96 (p=0.025).",
                                "ANOVA Plot":  "Box plots comparing four groups — the basis of one-way ANOVA.",
                                "Heatmap Grid":"8×8 correlation heatmap — intensity encodes variable relationship strength.",
                                "Histogram":   "Histogram of 300 normal samples with overlaid theoretical PDF.",
                                "Trend Chart": "24-period cumulative trend line showing temporal pattern.",
                                "Waterfall Plot":"Waterfall chart showing incremental value changes across periods.",
                                "Violin Plot": "Violin plots combining density estimation with box plot summaries.",
                                "Pareto Chart":"Pareto chart with cumulative percentage — the 80/20 rule visualised.",
                                "Pie Breakout":"Donut chart breaking down categorical proportions.",
                            }
                            explanation = expl_map.get(val, f"Visualisation for: {val}")
                            fake_reply  = f"<explanation>{explanation}</explanation><chart_params>{json.dumps(chip_params)}</chart_params>"
                            render_bot_block(fake_reply, mc_ref)
                            SESSIONS[s.cur_sid]['messages'].extend([{'role':'user','text':val},{'role':'bot','text':fake_reply}])
                        else:
                            try:
                                import requests as req
                                payload = {
                                    'message': val, 'mode': 'multi' if s.multi else 'single', 'domain': s.subject.lower(),
                                    'history': json.dumps([{'role':m['role'],'text':m['text']} for m in SESSIONS[s.cur_sid]['messages']])
                                }
                                res = await asyncio.get_event_loop().run_in_executor(None, lambda: req.post(API_URL, data=payload, timeout=API_TIMEOUT))
                                try: typing.delete()
                                except: pass
                                if res.status_code == 200:
                                    reply = res.json().get('reply','')
                                    render_bot_block(reply, mc_ref)
                                    SESSIONS[s.cur_sid]['messages'].extend([{'role':'user','text':val},{'role':'bot','text':reply}])
                                else:
                                    with mc_ref: ui.label(f'API error {res.status_code}').style('color:#ef4444;font-size:12px;font-style:italic;')
                            except Exception as exc:
                                try: typing.delete()
                                except: pass
                                with mc_ref: ui.label(f'Connection error: {exc}').style('color:#ef4444;font-size:12px;font-style:italic;')

                        s.pipe_step  = -1
                        s.processing = False
                        refs['render_pipeline'].refresh()
                        sa_ref.scroll_to(percent=1.0)

                    inp.on('keydown.enter', lambda: asyncio.create_task(send_message()))
                    with ui.button(on_click=lambda: asyncio.create_task(send_message())).classes('send-btn').props('flat'):
                        ui.html(icon_send())

        # RIGHT SIDEBAR
        with ui.element('div').classes('sidebar-right'):
            with ui.scroll_area().style('flex:1;min-height:0;'):
                with ui.element('div').style('padding:13px 12px 6px;'):
                    ui.label('QUICK ACTIONS').classes('slabel').style('margin-bottom:8px;padding:0 2px;')
                    with ui.element('div').classes('flex flex-wrap gap-1.5 w-full mb-4 no-select'):
                        for chip in CHIPS:
                            with ui.element('div').classes('action-bubble').on('click', lambda c=chip: asyncio.create_task(send_message(c))):
                                ui.label(chip)

            with ui.element('div').style('flex-shrink:0;border-top:1px solid #e5e7eb;padding:11px 12px 13px;background:#f9fafb;'):
                ui.label('PIPELINE').classes('slabel').style('margin-bottom:9px;')
                pipe_wrap = ui.element('div').style('display:flex;flex-direction:column;')
                @ui.refreshable
                def render_pipeline():
                    pipe_wrap.clear()
                    if not s.pipe_vis: return
                    with pipe_wrap:
                        for i, step in enumerate(PIPELINE_STEPS):
                            active  = 0 <= s.pipe_step and i <= s.pipe_step
                            current = i == s.pipe_step
                            lbl = step
                            if 'Mode' in lbl: lbl = f'Mode: {"Multi" if s.multi else "Single"}'
                            with ui.element('div').classes('flex items-start gap-4 w-full relative h-6'):
                                with ui.element('div').classes('w-3 flex flex-col items-center h-full relative'):
                                    if i < len(PIPELINE_STEPS)-1:
                                        ba = 0 <= s.pipe_step and i < s.pipe_step
                                        ui.element('div').style(f'position:absolute;left:50%;top:8px;width:1.5px;height:calc(100% + 1px);transform:translateX(-50%);z-index:0;transition:all .3s;background:{"#7c3aed" if ba else "#f3f4f6"};')
                                    ui.element('div').classes(f'w-1.5 h-1.5 rounded-full relative z-10 mt-1.5 {"pipe-active-pulse" if current else ""} {"shadow-[0_0_8px_rgba(124,58,237,0.3)]" if active else ""}'.strip()).style(f'background:{"#7c3aed" if active else "#d1d5db"};transition:all .3s;')
                                with ui.element('div').classes('flex-1 pt-0'):
                                    ui.label(lbl).classes('text-[8.5px] font-bold uppercase tracking-wider leading-none truncate').style(f'color:{"#7c3aed" if active else "#9ca3af"};transition:all .3s;padding-top:4px;')
                render_pipeline()
                refs['render_pipeline'] = render_pipeline

    def _save_session():
        sid = s.cur_sid
        if not sid or sid not in SESSIONS: return
        if not SESSIONS[sid].get('messages'):
            SESSIONS.pop(sid, None)
            if sid in SID_ORDER: SID_ORDER.remove(sid)

    def _load_session(sid: str):
        sess = SESSIONS.get(sid)
        if not sess: return
        s.cur_sid = sid
        mc_ref = refs['mc']; mc_ref.clear()
        for msg in sess['messages']:
            if msg['role'] == 'user': render_user_bubble(msg['text'], mc_ref)
            else: render_bot_block(msg['text'], mc_ref)
        render_recents.refresh()
        refs['scroll'].scroll_to(percent=1.0)


if __name__ in {'__main__', '__mp_main__'}:
    logger.info("Starting StatsAI Unified UI Frontend on port 8080...")
    ui.run(title='StatsAI Analyst', port=8080, show=False, reload=False)