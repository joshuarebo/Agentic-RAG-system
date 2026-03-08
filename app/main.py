from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api import router as api_router
from app.config import get_settings

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Policy Agent Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f0f2f5;color:#1a1a2e}
header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:#fff;padding:24px 32px}
header h1{font-size:1.5rem;font-weight:600}
header p{opacity:0.7;font-size:0.85rem;margin-top:4px}
.container{max-width:1100px;margin:0 auto;padding:24px 16px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:28px}
.card{background:#fff;border-radius:10px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}
.card .label{font-size:0.75rem;text-transform:uppercase;letter-spacing:0.5px;color:#666;margin-bottom:6px}
.card .value{font-size:1.8rem;font-weight:700;color:#1a1a2e}
.card .value.accent{color:#0f3460}
.section{background:#fff;border-radius:10px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:20px}
.section h2{font-size:1rem;font-weight:600;margin-bottom:14px;color:#1a1a2e}
table{width:100%;border-collapse:collapse;font-size:0.85rem}
th{text-align:left;padding:8px 12px;border-bottom:2px solid #eee;color:#666;font-weight:600}
td{padding:8px 12px;border-bottom:1px solid #f0f0f0}
.model-tag{display:inline-block;background:#e8eaf6;color:#1a1a2e;padding:2px 8px;border-radius:4px;font-size:0.8rem;font-family:monospace}
.refresh{font-size:0.75rem;color:#999;text-align:center;margin-top:16px}
</style>
</head>
<body>
<header>
  <h1>Policy-Aware AI Decision Agent</h1>
  <p>System Dashboard</p>
</header>
<div class="container">
  <div class="stats" id="stats"></div>
  <div class="section" id="models-section">
    <h2>Model Usage</h2>
    <table><thead><tr><th>Model</th><th>API Calls</th><th>Tokens</th></tr></thead>
    <tbody id="models-body"></tbody></table>
  </div>
  <div class="section">
    <h2>Recent Activity</h2>
    <table><thead><tr><th>Model</th><th>Tokens In</th><th>Tokens Out</th><th>Latency</th><th>Time</th></tr></thead>
    <tbody id="logs-body"></tbody></table>
  </div>
  <p class="refresh">Auto-refreshes every 30s</p>
</div>
<script>
function fmt(n){return n>=1000?(n/1000).toFixed(1)+'k':n.toString()}
function renderStats(d){
  const items=[
    {label:'Documents',value:d.documents},
    {label:'Chunks',value:d.chunks},
    {label:'Queries',value:d.total_queries},
    {label:'API Calls',value:d.api_calls},
    {label:'Total Tokens',value:fmt(d.total_tokens)},
    {label:'Avg Latency',value:d.avg_latency_ms+'ms'}
  ];
  document.getElementById('stats').innerHTML=items.map(i=>
    '<div class="card"><div class="label">'+i.label+'</div><div class="value accent">'+i.value+'</div></div>'
  ).join('');
}
function renderModels(models){
  const body=document.getElementById('models-body');
  const entries=Object.entries(models);
  if(!entries.length){body.innerHTML='<tr><td colspan="3" style="color:#999">No data yet</td></tr>';return;}
  body.innerHTML=entries.map(([m,v])=>
    '<tr><td><span class="model-tag">'+m+'</span></td><td>'+v.calls+'</td><td>'+fmt(v.tokens)+'</td></tr>'
  ).join('');
}
function renderLogs(logs){
  const body=document.getElementById('logs-body');
  if(!logs.length){body.innerHTML='<tr><td colspan="5" style="color:#999">No activity yet</td></tr>';return;}
  body.innerHTML=logs.map(l=>{
    const t=l.timestamp?new Date(l.timestamp).toLocaleTimeString():'--';
    return '<tr><td><span class="model-tag">'+l.model+'</span></td><td>'+l.tokens_in+'</td><td>'+l.tokens_out+'</td><td>'+l.latency_ms+'ms</td><td>'+t+'</td></tr>';
  }).join('');
}
async function load(){
  try{
    const r=await fetch('/api/dashboard/data');
    const d=await r.json();
    renderStats(d);renderModels(d.models);renderLogs(d.recent_logs);
  }catch(e){console.error(e);}
}
load();setInterval(load,30000);
</script>
</body>
</html>"""


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "A Policy-Aware AI Decision Agent with RAG, "
            "multi-model routing, and structured decisions."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/health",
            "dashboard": "/dashboard",
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    return app


app = create_app()
