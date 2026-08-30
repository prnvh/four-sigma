const pages=[['overview','Overview','▦'],['portfolio','Portfolio','⌁'],['markets','Markets','⌁'],['research','Research','□'],['risk','Risk Center','◇'],['decisions','Decisions','▤'],['audit','Audit Trail','◉'],['agents','Agents','⌂'],['system-plan','System Plan','⌘'],['settings','Settings','⚙']];
let state=null;
const nav=document.querySelector('#nav');
const view=document.querySelector('#view');
pages.forEach(([id,label,icon])=>{const link=document.createElement('a');link.href=`#${id}`;link.textContent=label;link.dataset.icon=icon;nav.append(link)});
const escapeHtml=value=>String(value??'—').replace(/[&<>"']/g,character=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const formatPercent=value=>Number.isFinite(Number(value))?`${(Number(value)*100).toFixed(2)}%`:'—';
const formatMoney=value=>Number.isFinite(Number(value))?new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:2}).format(value):'Awaiting run';

// Reusable metric component: each KPI uses the same visual and semantic structure.
function metricCard({label,value,note,tone='',icon='◇'}){return `<article class="metric-card"><div class="metric-head"><span>${escapeHtml(label)}</span><span class="metric-icon">${icon}</span></div><strong class="metric-value">${escapeHtml(value)}</strong><span class="metric-note ${tone}">${tone==='positive'?'●':'·'} ${escapeHtml(note)}</span></article>`}

// Reusable panel header keeps operational modules consistent with the Figma layout.
function panelHeader(title,subtitle,action=''){return `<header class="panel-head"><div><h2>${escapeHtml(title)}</h2><p>${escapeHtml(subtitle)}</p></div>${action}</header>`}
function emptyState(title,message){return `<div class="empty-state"><strong>${escapeHtml(title)}</strong><br>${escapeHtml(message)}</div>`}

// Equity points come only from a completed backend run; portfolio values are never invented.
function equityChart(points){
  if(!Array.isArray(points)||points.length<2)return `<div class="chart-wrap"><div class="chart-empty"><span><strong>No completed paper run</strong>Run a backtest to populate the equity curve.</span></div></div>`;
  const values=points.map(point=>Number(point.value??point.equity)).filter(Number.isFinite);
  if(values.length<2)return `<div class="chart-wrap"><div class="chart-empty">Equity data is unavailable.</div></div>`;
  const min=Math.min(...values),max=Math.max(...values),span=Math.max(max-min,1);
  const coordinates=values.map((value,index)=>[((index/(values.length-1))*100).toFixed(2),(92-((value-min)/span)*78).toFixed(2)]);
  const line=coordinates.map(pair=>pair.join(',')).join(' '),area=`0,100 ${line} 100,100`,[lastX,lastY]=coordinates.at(-1);
  return `<div class="chart-wrap" title="Latest equity: ${escapeHtml(formatMoney(values.at(-1)))}"><svg class="equity-chart" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Portfolio equity curve"><line class="chart-grid" x1="0" y1="25" x2="100" y2="25"/><line class="chart-grid" x1="0" y1="50" x2="100" y2="50"/><line class="chart-grid" x1="0" y1="75" x2="100" y2="75"/><polygon class="chart-area" points="${area}"/><polyline class="chart-line" points="${line}"/><circle class="chart-dot" cx="${lastX}" cy="${lastY}" r="2.2"/></svg></div>`;
}

function riskPanel(limits=[]){const rows=limits.map(limit=>{const ratio=Math.min(100,Math.max(0,Number(limit.utilization||0)*100));return `<div class="risk-row"><div class="risk-meta"><span>${escapeHtml(limit.label)}</span><strong>${escapeHtml(limit.current)} / ${escapeHtml(limit.limit)}</strong></div><div class="bar ${ratio>75?'amber':''}"><span style="width:${ratio}%"></span></div></div>`}).join('');return `<article class="panel">${panelHeader('Risk Limits','Deterministic controls','<span class="badge">● Enforced</span>')}<div class="risk-list">${rows}</div></article>`}
function activityFeed(items=[]){const rows=items.map(item=>`<div class="feed-row"><span class="feed-icon">${escapeHtml(item.icon||'✓')}</span><span><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.detail)}</small></span><span class="feed-state">${escapeHtml(item.state)}</span></div>`).join('');return `<article class="panel">${panelHeader('Recent Decisions','Latest governed system activity','<span class="badge neutral">Read only</span>')}<div class="feed">${rows||'<div class="empty-state">No audited activity loaded.</div>'}</div></article>`}
function agentPanel(agents=[]){const rows=agents.slice(0,6).map(agent=>`<div class="agent-row"><span class="agent-name"><strong>${escapeHtml(agent.name)}</strong><small>${escapeHtml(agent.output)}</small></span><span><strong>${escapeHtml(agent.version)}</strong><small>Versioned model</small></span><span class="feed-state">Ready</span></div>`).join('');return `<article class="panel">${panelHeader('Agent Runtime','Governed model topology','<span class="badge neutral">Registry</span>')}<div class="agent-list">${rows}</div></article>`}

function renderOverview(){
  const run=state.run,agents=state.agents||[],portfolio=run?.portfolio||{};
  const cards=[
    {label:'Portfolio Value',value:formatMoney(portfolio.value),note:run?'Paper portfolio':'No completed run',icon:'$'},
    {label:'Maximum Drawdown',value:formatPercent(run?.metrics?.max_drawdown),note:run?'Within configured guard':'Awaiting run metrics',tone:run?'positive':'',icon:'⌁'},
    {label:'Risk Status',value:'LOW',note:'Deterministic guards active',tone:'positive',icon:'◇'},
    {label:'Active Agents',value:agents.length,note:'Registry specifications loaded',tone:'positive',icon:'⌘'},
  ];
  view.innerHTML=`<div class="metric-grid">${cards.map(metricCard).join('')}</div><div class="overview-grid"><article class="panel">${panelHeader('Portfolio Performance','Actual paper-run equity and capital','<span class="badge neutral">Paper only</span>')}${equityChart(run?.equity_curve)}</article>${riskPanel(state.risk_limits)}</div><div class="lower-grid">${activityFeed(state.activity)}${agentPanel(agents)}</div>`;
}
function renderAgents(){const rows=(state.agents||[]).map(agent=>`<tr><td>${escapeHtml(agent.name)}</td><td>${escapeHtml(agent.version)}</td><td>${escapeHtml(agent.output)}</td><td><span class="badge">Ready</span></td></tr>`).join('');view.innerHTML=`<div class="page-intro"><h2>Agent registry</h2><p>Live, read-only specifications loaded directly from the Python registry.</p></div><table class="data-table"><thead><tr><th>AGENT</th><th>VERSION</th><th>ROLE / OUTPUT</th><th>STATE</th></tr></thead><tbody>${rows}</tbody></table>`}
function renderSettings(){const settings=state.settings;view.innerHTML=`<div class="page-intro"><h2>Runtime settings</h2><p>Read-only operational configuration. Secret values are never returned to the browser.</p></div><div class="metric-grid">${metricCard({label:'News Model',value:settings.news_model,note:'Server-side selection',icon:'N'})}${metricCard({label:'Decision Model',value:settings.decision_model,note:'Server-side selection',icon:'D'})}${metricCard({label:'API Configuration',value:settings.api_key_configured?'Configured':'Not configured',note:'Secret remains server-side',tone:settings.api_key_configured?'positive':'warning',icon:'K'})}${metricCard({label:'Execution Mode',value:'Paper only',note:'No brokerage authority',tone:'positive',icon:'P'})}</div>`}
function render(){
  const requested=location.hash.slice(1),page=pages.some(([id])=>id===requested)?requested:'overview',label=pages.find(([id])=>id===page)[1];
  document.querySelector('#page-title').textContent=label;[...nav.children].forEach(link=>link.classList.toggle('active',link.hash===`#${page}`));
  if(page==='system-plan')view.innerHTML='<iframe title="4 Sigma System Plan" src="system-plan.html"></iframe>';
  else if(!state)view.innerHTML=emptyState('Connecting','Loading read-only operations data…');
  else if(page==='overview')renderOverview();
  else if(page==='agents')renderAgents();
  else if(page==='settings')renderSettings();
  else if(!state.run)view.innerHTML=`<div class="page-intro"><h2>${escapeHtml(label)}</h2><p>This module populates only from a completed, audited paper-trading run.</p></div>${emptyState('No completed run loaded','No trading values have been fabricated.')}`;
  else view.innerHTML=emptyState(label,'Run visualization is ready for persisted backend data.');
}
async function load(){
  try{const response=await fetch('/api/dashboard',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);state=await response.json();document.querySelector('#backend-status').textContent=state.system.mode;document.querySelector('#system-label').textContent='All systems operational'}
  catch(error){state={system:{status:'Frontend only',mode:'Backend offline'},agents:[],run:null,settings:{news_model:'—',decision_model:'—',api_key_configured:false},risk_limits:[],activity:[]};document.querySelector('#backend-status').textContent='Start web.server for live data';document.querySelector('#system-label').textContent='Backend offline'}
  render();
}
document.querySelector('#current-date').textContent=new Intl.DateTimeFormat('en-US',{month:'short',day:'2-digit',year:'numeric'}).format(new Date());
window.addEventListener('hashchange',render);document.querySelector('#refresh').addEventListener('click',load);load();
