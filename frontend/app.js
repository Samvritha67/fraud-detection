const $ = (id) => document.getElementById(id);
const form = $('transaction-form');
const presets = {
  payment: { type: 'PAYMENT', step: 10, amount: 9839.64, oldbalanceOrg: 170136, oldbalanceDest: 0 },
  transfer: { type: 'TRANSFER', step: 215, amount: 50000, oldbalanceOrg: 60000, oldbalanceDest: 0 },
  fraud: { type: 'CASH_OUT', step: 12, amount: 181000, oldbalanceOrg: 181000, oldbalanceDest: 21182 },
};
const money = (amount) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 2 }).format(Number(amount));
const pct = (value, places = 1) => `${(Number(value) * 100).toFixed(places)}%`;
const safe = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

$('today').textContent = new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric' }).format(new Date());

document.querySelectorAll('[data-preset]').forEach((button) => {
  button.addEventListener('click', () => {
    Object.entries(presets[button.dataset.preset]).forEach(([key, value]) => { form.elements[key].value = value; });
    $('result-time').textContent = 'READY';
  });
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('submit-button');
  const payload = Object.fromEntries(new FormData(form).entries());
  for (const key of ['step', 'amount', 'oldbalanceOrg', 'oldbalanceDest']) payload[key] = Number(payload[key]);
  button.disabled = true;
  button.querySelector('span').textContent = 'Analyzing…';
  try {
    const response = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Could not analyze this transaction.');
    renderResult(result);
    $('result-time').textContent = new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(new Date());
    await loadHistory();
  } catch (error) {
    $('result-content').innerHTML = `<div class="error-message">${safe(error.message || 'The service could not be reached. Check that the backend is running.')}</div>`;
    $('result-time').textContent = 'ERROR';
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Analyze transaction';
  }
});

function renderResult(result) {
  const fraud = result.prediction === 1;
  const percent = Math.max(0, Math.min(100, result.fraud_probability * 100));
  $('result-content').innerHTML = `
    <div class="assessment">
      <div class="assessment-banner ${fraud ? 'fraud' : 'legitimate'}">
        <span class="assessment-symbol">${fraud ? '!' : '✓'}</span>
        <span><strong>${fraud ? 'Potential fraud detected' : 'Transaction appears legitimate'}</strong><small>${fraud ? 'This transaction crossed the screening threshold.' : 'This transaction is below the screening threshold.'}</small></span>
      </div>
      <div class="probability"><div class="probability-top"><span>Fraud probability</span><strong>${pct(result.fraud_probability)}</strong></div><div class="progress-track"><span style="width:${percent}%"></span></div><div class="risk-meta"><span>Risk level</span><strong>${safe(result.risk_level)}</strong></div></div>
      <div class="action-box"><small>RECOMMENDED ACTION</small><strong>${safe(result.action)}</strong></div>
    </div>`;
}

async function loadHealth() {
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error('Backend unavailable');
    await response.json();
    $('service-status').textContent = 'Model service online';
  } catch {
    $('service-status').textContent = 'Model service offline';
  }
}

async function loadMetrics() {
  try {
    const response = await fetch('/metrics');
    if (!response.ok) throw new Error('Metrics unavailable');
    const metrics = await response.json();
    $('rows-used').textContent = Number(metrics.rows_used || 0).toLocaleString();
    $('fraud-rate').textContent = pct(metrics.fraud_rate || 0, 3);
    $('threshold').textContent = pct(metrics.threshold ?? 0.5);
    const results = Array.isArray(metrics.results) ? metrics.results : Object.entries(metrics.results || {}).map(([model, values]) => ({ model, ...values }));
    const best = results.find((row) => row.model === metrics.best_model) || results[0];
    $('pr-auc').textContent = best ? Number(best['PR-AUC'] ?? best.pr_auc ?? 0).toFixed(3) : '—';
  } catch {
    $('rows-used').textContent = '—';
    $('fraud-rate').textContent = '—';
    $('threshold').textContent = '—';
    $('pr-auc').textContent = '—';
  }
}

async function loadHistory() {
  const body = $('history-table').querySelector('tbody');
  try {
    const response = await fetch('/history');
    if (!response.ok) throw new Error('History unavailable');
    const history = await response.json();
    $('activity-count').textContent = `${history.length} record${history.length === 1 ? '' : 's'}`;
    if (!history.length) {
      body.innerHTML = '<tr><td colspan="5" class="table-empty">Screened transactions will appear here.</td></tr>';
      return;
    }
    body.innerHTML = history.slice(0, 8).map((item) => `<tr><td>${safe(item.time)}</td><td>${safe(item.type)}</td><td>${money(item.amount)}</td><td><span class="status-pill ${item.prediction ? 'fraud' : 'legitimate'}">${safe(item.label)}</span></td><td class="prob-cell">${pct(item.fraud_probability)}</td></tr>`).join('');
  } catch {
    body.innerHTML = '<tr><td colspan="5" class="table-empty">Activity history is unavailable.</td></tr>';
  }
}

loadHealth();
loadMetrics();
loadHistory();
