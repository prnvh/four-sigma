const pages=[['overview','Overview','▦'],['portfolio','Portfolio','⌁'],['markets','Markets','⌁'],['research','Research','□'],['risk','Risk Center','◇'],['decisions','Decisions','▤'],['audit','Audit Trail','◉'],['agents','Agents','⌂'],['system-plan','System Plan','⌘'],['settings','Settings','⚙']];
let state=null,pollTimer=null;
const nav=document.querySelector('#nav');
const view=document.querySelector('#view');
pages.forEach(([id,label,icon])=>{const link=document.createElement('a');link.href=`#${id}`;link.textContent=label;link.dataset.icon=icon;nav.append(link)});

const escapeHtml=value=>String(value??'—').replace(/[&<>"']/g,character=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
const number=value=>Number.isFinite(Number(value))?new Intl.NumberFormat('en-US',{maximumFractionDigits:0}).format(value):'—';
const formatPercent=value=>Number.isFinite(Number(value))?`${(Number(value)*100).toFixed(2)}%`:'—';
const formatMoney=value=>Number.isFinite(Number(value))?new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:2}).format(value):'Awaiting mark';
const formatQuantity=value=>Number.isFinite(Number(value))?new Intl.NumberFormat('en-US',{maximumFractionDigits:4}).format(value):'—';
const formatDate=value=>{if(!value)return '—';const parsed=new Date(value);return Number.isNaN(parsed.valueOf())?String(value):new Intl.DateTimeFormat('en-US',{month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'}).format(parsed)};
const titleCase=value=>String(value??'—').replaceAll('_',' ').replace(/\b\w/g,letter=>letter.toUpperCase());
const toneFor=value=>['approved','allow','buy','completed','ready','live'].includes(String(value).toLowerCase())?'positive':['rejected','reject','sell','reduce'].includes(String(value).toLowerCase())?'negative':['resize','defer','paused'].includes(String(value).toLowerCase())?'warning':'';

function metricCard({label,value,note,tone='',icon='◇'}){return `<article class="metric-card"><div class="metric-head"><span>${escapeHtml(label)}</span><span class="metric-icon">${escapeHtml(icon)}</span></div><strong class="metric-value">${escapeHtml(value)}</strong><span class="metric-note ${tone}">${tone==='positive'?'●':'·'} ${escapeHtml(note)}</span></article>`}
function panelHeader(title,subtitle,action=''){return `<header class="panel-head"><div><h2>${escapeHtml(title)}</h2><p>${escapeHtml(subtitle)}</p></div>${action}</header>`}
function emptyState(title,message){return `<div class="empty-state"><strong>${escapeHtml(title)}</strong><br>${escapeHtml(message)}</div>`}
function badge(value,tone=toneFor(value)){return `<span class="badge ${tone}">${escapeHtml(value)}</span>`}
function pageIntro(title,copy,action=''){return `<div class="page-intro"><div><h2>${escapeHtml(title)}</h2><p>${escapeHtml(copy)}</p></div>${action}</div>`}

function equityChart(points){
  if(!Array.isArray(points)||points.length<2)return `<div class="chart-wrap"><div class="chart-empty"><span><strong>No equity history yet</strong>The chart will grow as the paper clock advances.</span></div></div>`;
  const usable=points.map(point=>({date:point.date,value:Number(point.value)})).filter(point=>Number.isFinite(point.value));
  if(usable.length<2)return `<div class="chart-wrap"><div class="chart-empty">Equity data is unavailable.</div></div>`;
  const values=usable.map(point=>point.value),min=Math.min(...values),max=Math.max(...values),span=Math.max(max-min,1);
  const coordinates=values.map((value,index)=>[((index/(values.length-1))*100).toFixed(2),(92-((value-min)/span)*78).toFixed(2)]);
  const line=coordinates.map(pair=>pair.join(',')).join(' '),area=`0,100 ${line} 100,100`,[lastX,lastY]=coordinates.at(-1);
  return `<div class="chart-wrap" title="Latest equity: ${escapeHtml(formatMoney(values.at(-1)))}"><svg class="equity-chart" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Paper portfolio equity curve"><line class="chart-grid" x1="0" y1="25" x2="100" y2="25"/><line class="chart-grid" x1="0" y1="50" x2="100" y2="50"/><line class="chart-grid" x1="0" y1="75" x2="100" y2="75"/><polygon class="chart-area" points="${area}"/><polyline class="chart-line" points="${line}"/><circle class="chart-dot" cx="${lastX}" cy="${lastY}" r="2.2"/></svg><div class="chart-caption"><span>${escapeHtml(usable[0].date)}</span><strong>${escapeHtml(formatMoney(values.at(-1)))}</strong><span>${escapeHtml(usable.at(-1).date)}</span></div></div>`;
}

function trajectoryChart(trajectory){
  const points=trajectory?.points||[];
  if(points.length<2)return equityChart(points);
  const benchmark=trajectory.benchmark?.points||[],all=[...points,...benchmark],values=all.map(point=>Number(point.growth)).filter(Number.isFinite),minimum=Math.min(10000,...values),maximum=Math.max(10000,...values),padding=Math.max((maximum-minimum)*.14,40),min=minimum-padding,max=maximum+padding,span=max-min;
  const start=new Date(`${trajectory.start}T00:00:00Z`).valueOf(),end=new Date(`${trajectory.end}T00:00:00Z`).valueOf();
  const coordinate=point=>[((((new Date(`${point.date}T00:00:00Z`).valueOf()-start)/(end-start))*100)).toFixed(2),(92-((Number(point.growth)-min)/span)*78).toFixed(2)];
  const strategyCoordinates=points.map(coordinate),benchmarkCoordinates=benchmark.map(coordinate),strategyLine=strategyCoordinates.map(pair=>pair.join(',')).join(' '),benchmarkLine=benchmarkCoordinates.map(pair=>pair.join(',')).join(' '),area=`0,100 ${strategyLine} 100,100`,startY=(92-((10000-min)/span)*78).toFixed(2);
  const dots=(trajectory.milestones||[]).map(item=>{const [x,y]=coordinate(item);return `<circle class="milestone-dot" cx="${x}" cy="${y}" r="2.2"/>`}).join('');
  const milestones=(trajectory.milestones||[]).map(item=>`<span><small>${escapeHtml(item.label)}</small><strong>${escapeHtml(formatMoney(item.growth))}</strong><b>${Number(item.return)>=0?'+':''}${escapeHtml(formatPercent(item.return))}</b><i>${escapeHtml(item.date)}</i></span>`).join('');
  const strategyEnd=points.at(-1),benchmarkEnd=benchmark.at(-1);
  return `<div class="chart-wrap trajectory-wrap"><div class="chart-legend"><span class="strategy-key">Strategy <strong>${escapeHtml(formatMoney(strategyEnd.growth))} · +${escapeHtml(formatPercent(strategyEnd.return))}</strong></span><span class="benchmark-key">S&amp;P 500 <strong>${escapeHtml(formatMoney(benchmarkEnd?.growth))} · ${Number(benchmarkEnd?.return)>=0?'+':''}${escapeHtml(formatPercent(benchmarkEnd?.return))}</strong></span></div><svg class="equity-chart" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Growth of ten thousand dollars in the paper strategy compared with the S and P 500 from January 1 through February 28"><line class="chart-grid zero-line" x1="0" y1="${startY}" x2="100" y2="${startY}"/><line class="chart-grid" x1="0" y1="25" x2="100" y2="25"/><line class="chart-grid" x1="0" y1="50" x2="100" y2="50"/><line class="chart-grid" x1="0" y1="75" x2="100" y2="75"/><polygon class="chart-area" points="${area}"/><polyline class="benchmark-line" points="${benchmarkLine}"/><polyline class="chart-line" points="${strategyLine}"/>${dots}</svg><span class="start-line-label" style="top:${startY}%">$10,000 start</span><div class="chart-caption"><span>Jan 1</span><strong>Growth of $10,000</strong><span>Feb 28</span></div></div><div class="milestone-row">${milestones}</div>`;
}

function runStrip(run){
  if(!run)return `<div class="run-strip muted-strip"><span class="pulse"></span><div><strong>No run log found</strong><small>Start a paper backtest or set QFIRM_RUN_LOG.</small></div></div>`;
  const progress=run.progress||{},status=run.status||'connected';
  return `<div class="run-strip"><span class="pulse ${status==='running'?'live':''}"></span><div><strong>${escapeHtml(titleCase(status))} · ${escapeHtml(run.id)}</strong><small>${escapeHtml(progress.simulation_date||'Paper clock initializing')} · updated ${escapeHtml(formatDate(run.updated_at))}</small></div><div class="run-progress"><strong>${escapeHtml(`${Number(progress.percent||0).toFixed(1)}%`)}</strong><span><i style="width:${Math.max(0,Math.min(100,Number(progress.percent||0)))}%"></i></span></div></div>`;
}

function periodStrip(trajectory){
  if(!trajectory)return runStrip(null);
  const end=trajectory.milestones?.at(-1),benchmark=trajectory.benchmark?.points?.at(-1);
  return `<div class="run-strip period-strip"><span class="pulse live"></span><div><strong>Featured Jan 1 – Feb 28 period</strong><small>Strategy and S&amp;P 500 normalized to the same $10,000 starting line</small></div><div class="period-result"><strong>${escapeHtml(formatMoney(end?.growth))}</strong><small>Strategy ${Number(end?.return)>=0?'+':''}${escapeHtml(formatPercent(end?.return))} · S&amp;P 500 ${Number(benchmark?.return)>=0?'+':''}${escapeHtml(formatPercent(benchmark?.return))}</small></div></div>`;
}

function riskPanel(limits=[]){
  const rows=limits.map(limit=>{const known=Number.isFinite(Number(limit.utilization)),ratio=known?Math.min(100,Math.max(0,Number(limit.utilization)*100)):0;return `<div class="risk-row"><div class="risk-meta"><span>${escapeHtml(limit.label)}</span><strong>${escapeHtml(limit.current)} / ${escapeHtml(limit.limit)}</strong></div><div class="bar ${known&&ratio>75?'amber':''} ${known?'':'pending'}"><span style="width:${ratio}%"></span></div></div>`}).join('');
  return `<article class="panel">${panelHeader('Risk Limits','Deterministic controls','<span class="badge positive">● Enforced</span>')}<div class="risk-list">${rows}</div></article>`;
}
function activityFeed(items=[]){const rows=items.slice(0,6).map(item=>`<div class="feed-row"><span class="feed-icon">${escapeHtml(item.icon||'✓')}</span><span><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.detail)}</small></span><span class="feed-state">${escapeHtml(item.state)}</span></div>`).join('');return `<article class="panel">${panelHeader('System Activity','Local connection and controls','<span class="badge neutral">Read only</span>')}<div class="feed">${rows||emptyState('No activity','The local API has not reported activity.')}</div></article>`}
function agentPanel(agents=[]){const rows=agents.slice(0,6).map(agent=>`<div class="agent-row"><span class="agent-name"><strong>${escapeHtml(titleCase(agent.name))}</strong><small>${escapeHtml(agent.output)}</small></span><span><strong>${escapeHtml(agent.version)}</strong><small>Versioned spec</small></span><span class="feed-state">Ready</span></div>`).join('');return `<article class="panel">${panelHeader('Agent Runtime','Governed model topology','<span class="badge neutral">Registry</span>')}<div class="agent-list">${rows}</div></article>`}

function renderOverview(){
  const run=state.run,agents=state.agents||[],featured=state.featured_trajectory,milestones=featured?.milestones||[],start=milestones[0],peak=milestones[2],end=milestones.at(-1),periodPnl=Number(end?.growth)-Number(featured?.starting_capital);
  const cards=[
    {label:'Paper Equity',value:formatMoney(end?.growth),note:'Jan–Feb period close',tone:'positive',icon:'$'},
    {label:'Paper P&L',value:formatMoney(periodPnl),note:'From the $10,000 starting line',tone:'positive',icon:'↗'},
    {label:'Peak Return',value:formatPercent(peak?.return),note:`Peak equity ${formatMoney(peak?.growth)}`,tone:'positive',icon:'⌁'},
    {label:'Period Return',value:formatPercent(end?.return),note:`Started ${formatMoney(start?.growth)}`,tone:'positive',icon:'⌘'},
  ];
  view.innerHTML=`${periodStrip(featured)}<div class="metric-grid">${cards.map(metricCard).join('')}</div><div class="overview-grid"><article class="panel">${panelHeader('Jan–Feb performance','Strategy and S&P 500 normalized to the same $10,000 start','<span class="badge neutral">Jan 1 – Feb 28</span>')}${trajectoryChart(featured)}</article>${riskPanel(state.risk_limits)}</div><div class="lower-grid">${activityFeed(state.activity)}${agentPanel(agents)}</div>`;
}

function positionTable(positions=[]){
  const rows=positions.map(position=>`<tr><td><strong>${escapeHtml(position.symbol)}</strong></td><td>${escapeHtml(formatQuantity(position.quantity))}</td><td>${badge(Number(position.quantity)>=0?'Long':'Short',Number(position.quantity)>=0?'positive':'warning')}</td></tr>`).join('');
  return rows?`<table class="data-table compact"><thead><tr><th>SYMBOL</th><th>QUANTITY</th><th>EXPOSURE</th></tr></thead><tbody>${rows}</tbody></table>`:emptyState('No open positions','The latest paper mark contains no positions.');
}
function fillTable(fills=[]){
  const rows=fills.slice(0,16).map(fill=>`<tr><td>${escapeHtml(formatDate(fill.timestamp))}</td><td><strong>${escapeHtml(fill.symbol)}</strong></td><td>${badge(fill.side)}</td><td>${escapeHtml(formatQuantity(fill.quantity))}</td><td>${escapeHtml(formatMoney(fill.price))}</td><td>${escapeHtml(formatMoney(fill.fee))}</td></tr>`).join('');
  return rows?`<table class="data-table"><thead><tr><th>TIME</th><th>SYMBOL</th><th>SIDE</th><th>QUANTITY</th><th>PRICE</th><th>FEE</th></tr></thead><tbody>${rows}</tbody></table>`:emptyState('No fills yet','Approved paper orders will appear here.');
}

function renderPortfolio(){
  const run=state.run,portfolio=run?.portfolio||{},featured=state.featured_trajectory,milestones=featured?.milestones||[],peak=milestones[2],end=milestones.at(-1),benchmarkEnd=featured?.benchmark?.points?.at(-1),periodPnl=Number(end?.growth)-Number(featured?.starting_capital);
  const cards=[
    {label:'Ending Equity',value:formatMoney(end?.growth),note:'Feb period close',tone:'positive',icon:'$'},
    {label:'Period P&L',value:formatMoney(periodPnl),note:'From $10,000 starting capital',tone:'positive',icon:'P'},
    {label:'Peak Return',value:formatPercent(peak?.return),note:formatMoney(peak?.growth),tone:'positive',icon:'↗'},
    {label:'S&P 500 Return',value:formatPercent(benchmarkEnd?.return),note:formatMoney(benchmarkEnd?.growth),tone:Number(benchmarkEnd?.return)>=0?'positive':'negative',icon:'S'},
  ];
  view.innerHTML=`${pageIntro('Paper portfolio','Featured Jan–Feb performance with current-run positions and fills below.',badge('Jan 1 – Feb 28','positive'))}${periodStrip(featured)}<div class="metric-grid">${cards.map(metricCard).join('')}</div><div class="overview-grid"><article class="panel">${panelHeader('Growth of $10,000','Strategy versus S&P 500 across the same Jan–Feb window',badge('Jan 1 – Feb 28','neutral'))}${trajectoryChart(featured)}</article><article class="panel">${panelHeader('Open positions','Quantities at the latest paper mark')}${positionTable(portfolio.positions)}</article></div><article class="panel section-panel">${panelHeader('Recent fills',`${number(run?.fills?.length)} most recent parsed executions`,badge('Simulated','neutral'))}${fillTable(run?.fills)}</article>`;
}

function renderMarkets(){
  const run=state.run,market=run?.market||{},progress=run?.progress||{},universe=market.universe||[];
  const cards=[
    {label:'Universe',value:number(universe.length),note:'Symbols in this run',tone:'positive',icon:'U'},
    {label:'Market Prints',value:number(market.prints),note:'Loaded Yahoo bars',tone:'positive',icon:'M'},
    {label:'Interval',value:market.interval||'—',note:'Paper-clock bar size',icon:'I'},
    {label:'Simulation Date',value:progress.simulation_date?.slice(0,10)||'—',note:'Latest processed tick',icon:'D'},
  ];
  const symbols=universe.map(symbol=>`<span class="symbol-chip"><i></i>${escapeHtml(symbol)}</span>`).join('');
  view.innerHTML=`${pageIntro('Market feed','Point-in-time market coverage for the connected paper run.',run?badge(run.status):'')}${runStrip(run)}<div class="metric-grid">${cards.map(metricCard).join('')}</div><div class="lower-grid"><article class="panel">${panelHeader('Run universe','Symbols loaded from the paper-run feed',badge(`${universe.length} symbols`,'neutral'))}<div class="chip-list">${symbols||emptyState('No symbols loaded','The run is still preparing its market feed.')}</div></article><article class="panel">${panelHeader('Data contract','What the dashboard is reading')}<div class="definition-list"><div><span>Price source</span><strong>Yahoo historical bars</strong></div><div><span>News source</span><strong>GDELT + original URLs</strong></div><div><span>Knowledge rule</span><strong>Point-in-time only</strong></div><div><span>Execution</span><strong>Paper fills only</strong></div></div></article></div>`;
}

function pipelinePanel(){const rows=(state.pipeline||[]).map(item=>`<div class="pipeline-row"><span>${escapeHtml(item.step)}</span><div><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(item.detail)}</small></div></div>`).join('');return `<article class="panel">${panelHeader('Research pipeline','From source event to governed paper action')}<div class="pipeline-list">${rows}</div></article>`}
function insightTable(insights=[]){const rows=insights.slice(0,16).map(item=>`<tr><td>${escapeHtml(formatDate(item.timestamp))}</td><td><strong>${escapeHtml(item.symbol)}</strong></td><td>${badge(item.direction,toneFor(item.direction))}</td><td>${escapeHtml(formatPercent(item.confidence))}</td><td>${badge(item.outcome)}</td></tr>`).join('');return rows?`<table class="data-table"><thead><tr><th>TIME</th><th>SYMBOL</th><th>DIRECTION</th><th>CONFIDENCE</th><th>GATE</th></tr></thead><tbody>${rows}</tbody></table>`:emptyState('No findings yet','Research findings will appear as the paper clock advances.')}
function renderResearch(){
  const research=state.run?.research||{},companies=research.companies||[],risks=research.risks||[];
  const cards=[
    {label:'Approved Insights',value:number(research.approved||0),note:'Passed governance',tone:'positive',icon:'✓'},
    {label:'Rejected Insights',value:number(research.rejected||0),note:'Failed promotion gate',icon:'×'},
    {label:'Company Analyses',value:number(companies.length),note:'Recent parsed analyses',tone:'positive',icon:'C'},
    {label:'Risk Analyses',value:number(risks.length),note:'Recent scenario reviews',tone:'positive',icon:'R'},
  ];
  const notes=companies.slice(0,4).map(item=>`<article class="note-card"><header><strong>${escapeHtml(item.symbol)}</strong>${badge(formatPercent(item.confidence),'neutral')}</header><p>${escapeHtml(item.summary)}</p><small>${escapeHtml(formatDate(item.timestamp))}</small></article>`).join('');
  view.innerHTML=`${pageIntro('Research operations','Evidence-bound findings and analyses produced inside the current paper run.')}<div class="metric-grid">${cards.map(metricCard).join('')}</div><div class="overview-grid"><article class="panel">${panelHeader('Latest insight gates','Recent promoted and rejected findings',badge('Evidence bound','neutral'))}${insightTable(research.insights)}</article>${pipelinePanel()}</div><div class="note-grid">${notes||emptyState('No company analyses yet','Company notes will appear after an insight is promoted.')}</div>`;
}

function reviewTable(reviews=[]){const rows=reviews.slice(0,18).map(item=>`<tr><td>${escapeHtml(formatDate(item.timestamp))}</td><td>${escapeHtml(titleCase(item.type))}</td><td><strong>${escapeHtml(item.symbol)}</strong></td><td>${badge(item.action)}</td><td>${item.size===null||item.size===undefined?'—':escapeHtml(formatPercent(item.size))}</td><td class="truncate" title="${escapeHtml(item.rationale)}">${escapeHtml(item.rationale||'Deterministic review recorded')}</td></tr>`).join('');return rows?`<table class="data-table"><thead><tr><th>TIME</th><th>LAYER</th><th>SYMBOL</th><th>ACTION</th><th>SIZE</th><th>RATIONALE</th></tr></thead><tbody>${rows}</tbody></table>`:emptyState('No risk reviews yet','Portfolio and timing decisions will appear after a trade candidate exists.')}
function renderRisk(){
  const controls=state.controls||[],reviews=state.run?.reviews||[];
  const rows=controls.map(item=>`<tr><td><strong>${escapeHtml(item.name)}</strong></td><td>${escapeHtml(item.value)}</td><td>${badge(item.effect,item.effect.includes('exit')||item.effect.includes('closed')?'warning':'neutral')}</td></tr>`).join('');
  view.innerHTML=`${pageIntro('Risk center','Hard controls stay deterministic; model reviews can advise but never override a veto.',badge('Fail closed','positive'))}<div class="overview-grid">${riskPanel(state.risk_limits)}<article class="panel">${panelHeader('Control policy','Configured defaults from the Python engine')}<table class="data-table flush"><thead><tr><th>CONTROL</th><th>LIMIT</th><th>EFFECT</th></tr></thead><tbody>${rows}</tbody></table></article></div><article class="panel section-panel">${panelHeader('Latest risk reviews','Portfolio sizing and timing decisions from the run',badge(`${reviews.length} recent`,'neutral'))}${reviewTable(reviews)}</article>`;
}

function renderDecisions(){
  const run=state.run,reviews=run?.reviews||[],fills=run?.fills||[];
  const actions=reviews.reduce((result,item)=>{result[item.action]=(result[item.action]||0)+1;return result},{});
  const cards=[
    {label:'Reviews',value:number(reviews.length),note:'Recent parsed decisions',tone:'positive',icon:'D'},
    {label:'Approvals / Allows',value:number((actions.approve||0)+(actions.allow||0)),note:'Advisory path clear',tone:'positive',icon:'✓'},
    {label:'Resizes',value:number(actions.resize||0),note:'Bounded before execution',tone:'warning',icon:'↘'},
    {label:'Recent Fills',value:number(fills.length),note:'Simulated executions',icon:'F'},
  ];
  view.innerHTML=`${pageIntro('Decision ledger','The most recent sizing, timing and execution outcomes from the paper run.')}<div class="metric-grid">${cards.map(metricCard).join('')}</div><article class="panel section-panel">${panelHeader('Governed decisions','Each action remains linked to its paper-run log',badge('Read only','neutral'))}${reviewTable(reviews)}</article><article class="panel section-panel">${panelHeader('Execution results','Paper fills after all required gates')}${fillTable(fills)}</article>`;
}

function renderAudit(){
  const events=state.run?.audit||[];
  const rows=events.map(item=>`<tr><td>${escapeHtml(formatDate(item.timestamp))}</td><td>${badge(item.type,'neutral')}</td><td><strong>${escapeHtml(item.symbol)}</strong></td><td>${badge(item.state)}</td></tr>`).join('');
  view.innerHTML=`${pageIntro('Audit trail','A compact view of immutable run events parsed from the latest local log.',badge('Read only','neutral'))}<div class="audit-summary"><span><strong>${escapeHtml(number(events.length))}</strong> recent events</span><span><strong>${escapeHtml(state.run?.source||'—')}</strong> source log</span><span><strong>${escapeHtml(formatDate(state.run?.updated_at))}</strong> last update</span></div>${rows?`<table class="data-table"><thead><tr><th>TIME</th><th>EVENT</th><th>SUBJECT</th><th>STATE</th></tr></thead><tbody>${rows}</tbody></table>`:emptyState('No audit events yet','The connected run has not emitted a governed event.')}`;
}

function renderAgents(){const rows=(state.agents||[]).map(agent=>`<tr><td><strong>${escapeHtml(titleCase(agent.name))}</strong></td><td>${escapeHtml(agent.version)}</td><td>${escapeHtml(agent.output)}</td><td>${badge('Ready')}</td></tr>`).join('');view.innerHTML=`${pageIntro('Agent registry','Live, read-only specifications loaded directly from the Python registry.',badge(`${state.agents?.length||0} specs`,'neutral'))}<table class="data-table"><thead><tr><th>AGENT</th><th>VERSION</th><th>ROLE / OUTPUT</th><th>STATE</th></tr></thead><tbody>${rows}</tbody></table>`}
function renderSettings(){const settings=state.settings,run=state.run;view.innerHTML=`${pageIntro('Runtime settings','Operational configuration and a deployment-ready snapshot workflow.')}<div class="metric-grid">${metricCard({label:'News Model',value:settings.news_model,note:'Server-side selection',icon:'N'})}${metricCard({label:'Decision Model',value:settings.decision_model,note:'Server-side selection',icon:'D'})}${metricCard({label:'API Configuration',value:settings.api_key_configured?'Configured':'Not configured',note:'Secret remains server-side',tone:settings.api_key_configured?'positive':'warning',icon:'K'})}${metricCard({label:'Execution Mode',value:'Paper only',note:'No brokerage authority',tone:'positive',icon:'P'})}</div><div class="lower-grid"><article class="panel">${panelHeader('Paper-run connection','Local server watches the newest run log')}<div class="definition-list"><div><span>Connection</span><strong>${run?'Connected':'Waiting'}</strong></div><div><span>Source</span><strong>${escapeHtml(run?.source||'Newest .qfirm-cache/runs/*.log')}</strong></div><div><span>Updated</span><strong>${escapeHtml(formatDate(run?.updated_at))}</strong></div><div><span>Mode</span><strong>${escapeHtml(run?.status||'No run')}</strong></div></div></article><article class="panel">${panelHeader('Vercel handoff','One snapshot command before upload')}<div class="deploy-card"><span>1</span><p><strong>Export the latest run</strong><code>npm run snapshot</code></p><span>2</span><p><strong>Preview or deploy</strong><code>npm run deploy</code></p><small>Vercel receives a read-only snapshot. API keys and local cache files are never uploaded.</small></div></article></div>`}

function render(){
  const requested=location.hash.slice(1),page=pages.some(([id])=>id===requested)?requested:'overview',label=pages.find(([id])=>id===page)[1];
  document.body.classList.toggle('system-plan-page',page==='system-plan');
  document.querySelector('#page-title').textContent=label;[...nav.children].forEach(link=>link.classList.toggle('active',link.hash===`#${page}`));
  if(page==='system-plan')view.innerHTML='<iframe title="4 Sigma System Plan" src="system-plan.html?embed=1"></iframe>';
  else if(!state)view.innerHTML=emptyState('Connecting','Loading paper-run operations data…');
  else ({overview:renderOverview,portfolio:renderPortfolio,markets:renderMarkets,research:renderResearch,risk:renderRisk,decisions:renderDecisions,audit:renderAudit,agents:renderAgents,settings:renderSettings}[page]||renderOverview)();
}

async function load({quiet=false}={}){
  const refresh=document.querySelector('#refresh');if(!quiet)refresh.classList.add('loading');
  try{
    const localHosts=new Set(['127.0.0.1','localhost','[::1]']),sources=localHosts.has(location.hostname)?['/api/dashboard','/data/dashboard.json']:['/data/dashboard.json'];
    let response=null;
    for(const source of sources){response=await fetch(source,{cache:'no-store'});if(response.ok)break}
    if(!response?.ok)throw new Error(`HTTP ${response?.status||'offline'}`);
    state=await response.json();
    document.querySelector('#backend-status').textContent=state.system.mode;
    document.querySelector('#system-label').textContent=state.run?`Paper run ${state.run.status}`:'Dashboard ready';
    document.querySelector('.status-dot').classList.toggle('paused',state.run?.status==='paused');
  }catch(error){
    state={system:{status:'Frontend only',mode:'Backend offline'},agents:[],run:null,settings:{news_model:'—',decision_model:'—',api_key_configured:false},risk_limits:[],activity:[],pipeline:[],controls:[]};
    document.querySelector('#backend-status').textContent='Start web.server for live data';document.querySelector('#system-label').textContent='Backend offline';
  }finally{refresh.classList.remove('loading')}
  render();schedulePoll();
}
function schedulePoll(){clearTimeout(pollTimer);if(state?.run?.status==='running')pollTimer=setTimeout(()=>{if(document.visibilityState==='visible')load({quiet:true});else schedulePoll()},10000)}

document.querySelector('#current-date').textContent=new Intl.DateTimeFormat('en-US',{month:'short',day:'2-digit',year:'numeric'}).format(new Date());
window.addEventListener('hashchange',render);
document.querySelector('#refresh').addEventListener('click',()=>load());
document.querySelector('.collapse').addEventListener('click',()=>{document.body.classList.toggle('nav-collapsed');document.querySelector('.collapse span').textContent=document.body.classList.contains('nav-collapsed')?'Expand':'Collapse'});
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible'&&state?.run?.status==='running')load({quiet:true})});
load();
