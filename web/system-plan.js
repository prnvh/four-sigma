const details = {
  website: ['ENTRY POINT', 'Dashboard / Website', 'The human-facing control surface for portfolio state, markets, agent health, risk and analytics.', 'Displays governed outputs; it never bypasses backend controls.'],
  query: ['USER INPUT', 'Chat & Query Form', 'Accepts a company, market or portfolio question and converts it into an explicit research request.', 'A user request initiates analysis, not an automatic trade.'],
  'news-feed': ['IMMUTABLE INPUT', 'News Events', 'Point-in-time articles are retained with timestamps and original evidence URLs.', 'Future or unsourced news cannot enter historical context.'],
  'market-feed': ['IMMUTABLE INPUT', 'Market Data', 'Price, volume and derived tape features are captured as visible market events.', 'Simulation decisions only see prints available at that clock time.'],
  'company-feed': ['IMMUTABLE INPUT', 'Company Evidence', 'Profiles, filings and fundamental facts provide sourced company context.', 'Evidence time and knowledge time remain explicit.'],
  'portfolio-feed': ['LIVE STATE', 'Portfolio State', 'Cash, positions, fills and P&L feed portfolio-level review and monitoring.', 'Research agents cannot read private portfolio fields.'],
  gateway: ['PERMISSION BOUNDARY', 'Context Gateway', 'Builds a purpose-specific snapshot from only authorized, point-in-time records.', 'Every context snapshot is hashed and auditable.'],
  'news-agent': ['AGENT 01', 'News Agent', 'Filters, deduplicates, classifies and links news to relevant companies without forming investment conclusions.', 'Output remains an evidence-linked observation.'],
  'news-analyst': ['AGENT 02', 'News Analyst', 'Scores relevance, direction, horizon and confidence from visible news evidence.', 'Proposals must pass governance before becoming shared insight.'],
  'company-analyst': ['AGENT 03', 'Company Analyst', 'Combines promoted insights and company evidence into a time-bounded thesis.', 'No portfolio access and no execution authority.'],
  'risk-analyst': ['AGENT 04', 'Risk Analyst', 'Challenges the thesis with failure modes, success scenarios and outcome probabilities.', 'Qualitative risk advice cannot override deterministic limits.'],
  'portfolio-agent': ['AGENT 05', 'Portfolio Risk Analyst', 'Reviews the proposal against existing exposure and recommends rejection, deferral or bounded sizing.', 'Recommended size must remain inside the deterministic approval.'],
  governance: ['GATE 01', 'Governance', 'Checks evidence, permissions, schema, expiry and duplication before promoting knowledge.', 'Malformed or unauthorized output fails closed.'],
  'risk-engine': ['FINAL VETO', 'Deterministic Risk Engine', 'Computes exposure, concentration, volatility, liquidity, drawdown and loss controls.', 'It may pass, resize or reject—and no model can overrule it.'],
  'portfolio-action': ['SIMULATED ACTION', 'Portfolio Decision', 'Creates a paper order only after every required gate approves the candidate.', 'Failure at any stage produces no trade.'],
  audit: ['FEEDBACK LOOP', 'Audit & Attribution', 'Connects sources, analysis, governance, risk, fills and P&L for replay and evaluation.', 'Performance feedback improves experiments without rewriting history.']
};

const canvas = document.querySelector('#flow-canvas');
const zoomLabel = document.querySelector('#zoom-level');
const inspector = document.querySelector('#inspector');
let zoom = 1;

function applyZoom(next) {
  zoom = Math.min(1.35, Math.max(.65, next));
  canvas.style.transform = `scale(${zoom})`;
  canvas.parentElement.style.setProperty('min-height', `${Math.round(850 * zoom)}px`);
  zoomLabel.value = `${Math.round(zoom * 100)}%`;
  zoomLabel.textContent = zoomLabel.value;
}

document.querySelector('#zoom-in').addEventListener('click', () => applyZoom(zoom + .1));
document.querySelector('#zoom-out').addEventListener('click', () => applyZoom(zoom - .1));
document.querySelector('#reset-view').addEventListener('click', () => {
  applyZoom(1);
  canvas.parentElement.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
});

document.querySelectorAll('[data-node]').forEach((node) => {
  node.addEventListener('click', () => {
    document.querySelectorAll('[data-node]').forEach(item => item.classList.remove('selected'));
    node.classList.add('selected');
    const [type, title, copy, rule] = details[node.dataset.node];
    document.querySelector('#detail-type').textContent = type;
    document.querySelector('#detail-title').textContent = title;
    document.querySelector('#detail-copy').textContent = copy;
    document.querySelector('#detail-rule').textContent = rule;
    inspector.classList.add('open');
  });
});

document.querySelector('#close-inspector').addEventListener('click', () => {
  inspector.classList.remove('open');
  document.querySelectorAll('[data-node]').forEach(item => item.classList.remove('selected'));
});

applyZoom(window.innerWidth < 900 ? .75 : 1);
