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

APP_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Policy-Aware AI Decision Agent</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f0f2f5;color:#1a1a2e}
header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:#fff;padding:20px 32px;display:flex;align-items:center;justify-content:space-between}
header h1{font-size:1.4rem;font-weight:600}
header nav a{color:rgba(255,255,255,0.7);text-decoration:none;margin-left:20px;font-size:0.85rem;transition:color 0.2s}
header nav a:hover{color:#fff}
.layout{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 64px)}
.sidebar{background:#fff;border-right:1px solid #e0e0e0;padding:20px;overflow-y:auto}
.sidebar h2{font-size:0.9rem;text-transform:uppercase;letter-spacing:0.5px;color:#666;margin-bottom:14px}
.upload-area{border:2px dashed #ccc;border-radius:8px;padding:20px;text-align:center;cursor:pointer;transition:border-color 0.2s;margin-bottom:16px}
.upload-area:hover,.upload-area.dragover{border-color:#0f3460;background:#f8f9ff}
.upload-area input{display:none}
.upload-area p{font-size:0.85rem;color:#666}
.upload-area .icon{font-size:1.8rem;margin-bottom:6px}
.doc-list{list-style:none}
.doc-item{display:flex;align-items:center;justify-content:space-between;padding:10px 12px;border-radius:6px;margin-bottom:6px;background:#f8f9fa;font-size:0.85rem}
.doc-item .name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.doc-item .chunks{color:#666;font-size:0.75rem;margin:0 8px}
.doc-item .del{background:none;border:none;color:#999;cursor:pointer;font-size:1rem;padding:0 4px;transition:color 0.2s}
.doc-item .del:hover{color:#e53935}
.main{padding:24px 32px;overflow-y:auto}
.query-box{background:#fff;border-radius:10px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,0.08);margin-bottom:24px}
.query-box h2{font-size:1rem;font-weight:600;margin-bottom:12px}
.query-row{display:flex;gap:12px}
.query-row textarea{flex:1;border:1px solid #ddd;border-radius:8px;padding:12px;font-size:0.95rem;resize:vertical;min-height:56px;font-family:inherit;transition:border-color 0.2s}
.query-row textarea:focus{outline:none;border-color:#0f3460}
.btn{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;border:none;border-radius:8px;padding:12px 28px;font-size:0.9rem;cursor:pointer;font-weight:600;transition:opacity 0.2s;white-space:nowrap}
.btn:hover{opacity:0.9}
.btn:disabled{opacity:0.5;cursor:not-allowed}
.result{background:#fff;border-radius:10px;box-shadow:0 1px 3px rgba(0,0,0,0.08);overflow:hidden}
.result-header{padding:20px 24px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.badge{display:inline-block;padding:6px 18px;border-radius:20px;font-weight:700;font-size:0.95rem;letter-spacing:0.5px}
.badge.PASS{background:#e8f5e9;color:#2e7d32}
.badge.FAIL{background:#ffebee;color:#c62828}
.badge.NEEDS_INFO{background:#fff3e0;color:#e65100}
.confidence-bar{flex:1;min-width:120px;max-width:200px}
.confidence-bar .track{height:8px;background:#eee;border-radius:4px;overflow:hidden}
.confidence-bar .fill{height:100%;border-radius:4px;transition:width 0.5s}
.confidence-bar .lbl{font-size:0.75rem;color:#666;margin-top:4px}
.model-info{font-size:0.75rem;color:#999;margin-left:auto}
.answer-section{padding:0 24px 20px;border-bottom:1px solid #f0f0f0}
.answer-section p{font-size:0.95rem;line-height:1.6;color:#333}
.section-title{font-size:0.8rem;text-transform:uppercase;letter-spacing:0.5px;color:#666;margin:20px 24px 10px;font-weight:600}
.reasons{padding:0 24px 16px;list-style:none}
.reasons li{padding:6px 0;font-size:0.9rem;color:#444;display:flex;align-items:flex-start;gap:8px}
.reasons li::before{content:"";flex-shrink:0;width:6px;height:6px;border-radius:50%;background:#0f3460;margin-top:7px}
.evidence-list{padding:0 24px 16px}
.ev-card{background:#f8f9fa;border-radius:8px;padding:14px;margin-bottom:10px;border-left:3px solid #0f3460}
.ev-card .ev-source{font-size:0.75rem;color:#666;margin-bottom:4px;font-weight:600}
.ev-card .ev-content{font-size:0.85rem;color:#444;line-height:1.5}
.ev-card .ev-score{font-size:0.7rem;color:#999;margin-top:4px}
.steps{padding:0 24px 20px}
.step{display:flex;gap:12px;margin-bottom:10px;font-size:0.85rem}
.step-num{width:24px;height:24px;border-radius:50%;background:#1a1a2e;color:#fff;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0}
.step-body{flex:1}
.step-action{font-weight:600;color:#1a1a2e}
.step-detail{color:#666;font-size:0.8rem}
.step-result{color:#444;font-size:0.8rem;margin-top:2px}
.loading{text-align:center;padding:40px;color:#666}
.loading .spinner{display:inline-block;width:28px;height:28px;border:3px solid #eee;border-top-color:#1a1a2e;border-radius:50%;animation:spin 0.8s linear infinite;margin-bottom:12px}
@keyframes spin{to{transform:rotate(360deg)}}
.empty{text-align:center;padding:40px;color:#999;font-size:0.9rem}
.toast{position:fixed;bottom:24px;right:24px;background:#1a1a2e;color:#fff;padding:12px 20px;border-radius:8px;font-size:0.85rem;opacity:0;transition:opacity 0.3s;z-index:100;pointer-events:none}
.toast.show{opacity:1}
@media(max-width:768px){
  .layout{grid-template-columns:1fr}
  .sidebar{border-right:none;border-bottom:1px solid #e0e0e0;max-height:240px}
  .main{padding:16px}
  .query-row{flex-direction:column}
}
</style>
</head>
<body>
<header>
  <h1>Policy-Aware AI Decision Agent</h1>
  <nav>
    <a href="/dashboard">Dashboard</a>
    <a href="/docs">API Docs</a>
  </nav>
</header>
<div class="layout">
  <aside class="sidebar">
    <h2>Documents</h2>
    <div class="upload-area" id="dropZone">
      <input type="file" id="fileInput" accept=".pdf,.txt,.md,.markdown" multiple>
      <div class="icon">+</div>
      <p>Drop files here or click to upload<br><small>PDF, TXT, Markdown</small></p>
    </div>
    <ul class="doc-list" id="docList"></ul>
  </aside>
  <div class="main">
    <div class="query-box">
      <h2>Ask a Question</h2>
      <div class="query-row">
        <textarea id="question" placeholder="e.g. Is this invoice compliant with the payment policy?" rows="2"></textarea>
        <button class="btn" id="analyzeBtn" onclick="analyze()">Analyze</button>
      </div>
    </div>
    <div id="resultArea"></div>
  </div>
</div>
<div class="toast" id="toast"></div>
<script>
const API='/api';

// Toast
function toast(msg){
  const t=document.getElementById('toast');
  t.textContent=msg;t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),3000);
}

// Documents
async function loadDocs(){
  try{
    const r=await fetch(API+'/documents');
    const docs=await r.json();
    const list=document.getElementById('docList');
    if(!docs.length){list.innerHTML='<li style="color:#999;font-size:0.85rem;padding:8px">No documents uploaded yet</li>';return;}
    list.innerHTML=docs.map(d=>
      '<li class="doc-item"><span class="name" title="'+d.filename+'">'+d.filename+'</span><span class="chunks">'+d.chunk_count+' chunks</span><button class="del" onclick="deletDoc(\\''+d.doc_id+'\\',\\''+d.filename+'\\')">x</button></li>'
    ).join('');
  }catch(e){console.error(e);}
}

async function uploadFile(file){
  const form=new FormData();
  form.append('file',file);
  try{
    const r=await fetch(API+'/documents/upload',{method:'POST',body:form});
    if(!r.ok){const e=await r.json();toast('Error: '+(e.detail||'Upload failed'));return;}
    const d=await r.json();
    toast(d.filename+' uploaded ('+d.chunk_count+' chunks)');
    loadDocs();
  }catch(e){toast('Upload failed: '+e.message);}
}

async function deletDoc(id,name){
  try{
    await fetch(API+'/documents/'+id,{method:'DELETE'});
    toast(name+' deleted');
    loadDocs();
  }catch(e){toast('Delete failed');}
}

// Drop zone
const drop=document.getElementById('dropZone');
const fileInput=document.getElementById('fileInput');
drop.addEventListener('click',()=>fileInput.click());
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('dragover');});
drop.addEventListener('dragleave',()=>drop.classList.remove('dragover'));
drop.addEventListener('drop',e=>{
  e.preventDefault();drop.classList.remove('dragover');
  [...e.dataTransfer.files].forEach(uploadFile);
});
fileInput.addEventListener('change',()=>{
  [...fileInput.files].forEach(uploadFile);
  fileInput.value='';
});

// Analyze
async function analyze(){
  const q=document.getElementById('question').value.trim();
  if(!q){toast('Please enter a question');return;}
  const btn=document.getElementById('analyzeBtn');
  const area=document.getElementById('resultArea');
  btn.disabled=true;btn.textContent='Analyzing...';
  area.innerHTML='<div class="loading"><div class="spinner"></div><p>Retrieving evidence and generating decision...</p></div>';
  try{
    const r=await fetch(API+'/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    if(!r.ok){const e=await r.json();area.innerHTML='<div class="empty">Error: '+(e.detail||'Query failed')+'</div>';return;}
    const d=await r.json();
    renderResult(d);
  }catch(e){
    area.innerHTML='<div class="empty">Error: '+e.message+'</div>';
  }finally{
    btn.disabled=false;btn.textContent='Analyze';
  }
}

document.getElementById('question').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();analyze();}
});

function renderResult(d){
  const dec=d.decision;
  const u=d.model_usage;
  const confPct=Math.round(dec.confidence*100);
  const confColor=confPct>=80?'#2e7d32':confPct>=60?'#e65100':'#c62828';

  let html='<div class="result">';

  // Header
  html+='<div class="result-header">';
  html+='<span class="badge '+dec.decision+'">'+dec.decision+'</span>';
  html+='<div class="confidence-bar"><div class="track"><div class="fill" style="width:'+confPct+'%;background:'+confColor+'"></div></div><div class="lbl">Confidence: '+confPct+'%</div></div>';
  html+='<div class="model-info">'+u.model+' &middot; '+u.tokens_input+'in/'+u.tokens_output+'out &middot; '+Math.round(u.latency_ms)+'ms</div>';
  html+='</div>';

  // Answer
  html+='<div class="answer-section"><p>'+escHtml(d.answer)+'</p></div>';

  // Reasons
  if(dec.reasons&&dec.reasons.length){
    html+='<div class="section-title">Reasons</div><ul class="reasons">';
    dec.reasons.forEach(r=>{html+='<li>'+escHtml(r)+'</li>';});
    html+='</ul>';
  }

  // Evidence
  if(dec.evidence&&dec.evidence.length){
    html+='<div class="section-title">Evidence ('+dec.evidence.length+' chunks)</div><div class="evidence-list">';
    dec.evidence.forEach(e=>{
      html+='<div class="ev-card"><div class="ev-source">'+escHtml(e.document_source)+' &middot; Chunk '+e.chunk_index+'</div><div class="ev-content">'+escHtml(e.content)+'</div><div class="ev-score">Relevance: '+(e.relevance_score*100).toFixed(1)+'%</div></div>';
    });
    html+='</div>';
  }

  // Reasoning steps
  if(dec.reasoning_steps&&dec.reasoning_steps.length){
    html+='<div class="section-title">Reasoning Steps</div><div class="steps">';
    dec.reasoning_steps.forEach(s=>{
      html+='<div class="step"><div class="step-num">'+s.step_number+'</div><div class="step-body"><div class="step-action">'+escHtml(s.action)+'</div><div class="step-detail">'+escHtml(s.detail)+'</div>'+(s.result?'<div class="step-result">'+escHtml(s.result)+'</div>':'')+'</div></div>';
    });
    html+='</div>';
  }

  html+='</div>';
  document.getElementById('resultArea').innerHTML=html;
}

function escHtml(s){
  if(!s)return'';
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

loadDocs();
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
            "ui": "/ui",
        }

    @app.get("/ui", response_class=HTMLResponse)
    async def ui():
        return APP_UI_HTML

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return DASHBOARD_HTML

    return app


app = create_app()
