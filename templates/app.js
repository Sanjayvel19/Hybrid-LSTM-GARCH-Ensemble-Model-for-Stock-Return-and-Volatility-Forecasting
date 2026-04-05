// ═══════════ MAIN APP ══════════════════════════════════════════════
let allPredictions = [];
let allAnalysis    = {};
let filterSector   = 'ALL';
let filterSignal   = 'ALL';
let searchTerm     = '';
let isLoading      = false;
let refreshTimer   = null;

async function loadData() {
  if (isLoading) return;
  isLoading = true;
  setLoadingUI(true);
  try {
    const r = await fetch('/api/predict_all');
    if (!r.ok) throw new Error('Server ' + r.status);
    const d = await r.json();
    allPredictions = d.predictions || [];
    allAnalysis    = d.analysis    || {};
    render();
  } catch (e) {
    document.getElementById('tableBody').innerHTML =
      `<tr><td colspan="11" style="text-align:center;padding:3rem"><div class="error">⚠ ${e.message}<br><small>Make sure Flask is running on port 5000</small></div></td></tr>`;
    document.getElementById('tableInfo').textContent = 'Error';
  } finally { isLoading = false; setLoadingUI(false); }
}

async function forceRefresh() {
  await fetch('/api/cache/clear',{method:'POST'}).catch(()=>{});
  loadData();
}

function setLoadingUI(l) {
  const b = document.getElementById('refreshBtn');
  b.disabled = l; b.textContent = l ? '…' : '↺ Refresh';
}

function startAutoRefresh() {
  if (refreshTimer) clearInterval(refreshTimer);
  refreshTimer = setInterval(()=>{ if(!document.hidden) loadData(); }, 60000);
}
document.addEventListener('visibilitychange', ()=>{ if(!document.hidden) loadData(); });

function switchMainTab(e, tab) {
  e.preventDefault();
  document.querySelectorAll('.tab-content').forEach(el=>el.classList.remove('active'));
  document.querySelectorAll('.main-tab').forEach(el=>el.classList.remove('active'));
  document.getElementById(tab+'-tab').classList.add('active');
  e.target.classList.add('active');
  if (tab==='analysis') setTimeout(renderAllCharts, 80);
  if (tab==='compare')  setTimeout(initCompareTab, 80);
}

function filterBySector(e, s) {
  e.preventDefault();
  filterSector = s;
  document.querySelectorAll('.controls .filter-btn').forEach((b,i)=>{ if(i<4) b.classList.remove('active'); });
  e.target.classList.add('active');
  render();
}

function filterBySignal(e, s) {
  e.preventDefault();
  filterSignal = s;
  ['sigAll','sigBuy','sigHold','sigSell'].forEach(id=>document.getElementById(id).classList.remove('active'));
  e.target.classList.add('active');
  render();
}

function filterStocks() { searchTerm = document.getElementById('searchBox').value.toLowerCase(); render(); }

function getFiltered() {
  return allPredictions.filter(p => {
    if (filterSector !== 'ALL' && p.Sector !== filterSector) return false;
    if (filterSignal !== 'ALL' && p.Signal !== filterSignal)   return false;
    if (searchTerm && !p.Stock.toLowerCase().includes(searchTerm)) return false;
    return true;
  });
}

function render() { renderStats(); renderPriceCards(); renderTable(); }

function renderStats() {
  const s = allAnalysis.statistics || {};
  const set = (id, v, sfx='') => { document.getElementById(id).textContent = v!=null ? v+sfx : '—'; };
  set('totalStocks',   s.total_stocks);
  set('buySignals',    s.buy_signals);
  set('holdSignals',   s.hold_signals);
  set('sellSignals',   s.sell_signals);
  set('avgVolatility', s.avg_volatility, '%');
  set('avgReturn',     s.avg_return,     '%');
  set('winRate',       s.win_rate,       '%');
}

function renderPriceCards() {
  const items = allAnalysis.price_comparison || [];
  const grid  = document.getElementById('priceGrid');
  if (!items.length) { grid.innerHTML=''; return; }
  grid.innerHTML = items.map(item => {
    const pos   = item.percentage_change > 0;
    const cls   = pos ? 'positive' : 'negative';
    const cSign = pos ? '+' : '';
    const sigCls = item.signal==='BUY'?'sb-buy':item.signal==='HOLD'?'sb-hold':'sb-sell';
    return `<div class="price-card">
      <div class="price-card-header">
        <span class="price-card-name">${item.stock} · ${item.sector}</span>
        <span class="sb ${sigCls}">${item.signal}</span>
      </div>
      <div class="price-row"><div class="price-label">Actual</div><div class="price-value">₹${item.actual_price.toFixed(2)}</div></div>
      <div class="price-row"><div class="price-label">LSTM Predicted</div><div class="price-value">₹${item.lstm_predicted.toFixed(2)}</div></div>
      <div class="price-row"><div class="price-label">Ensemble</div><div class="price-value">₹${item.ensemble.toFixed(2)}</div></div>
      <div class="price-row"><div class="price-label">Return %</div><div class="price-value ${cls}">${cSign}${item.percentage_change.toFixed(2)}%</div></div>
      <div class="price-row"><div class="price-label">Volatility</div><div class="price-value">${item.volatility.toFixed(2)}%</div></div>
    </div>`;
  }).join('');
}

function renderTable() {
  const filtered  = getFiltered();
  const tableBody = document.getElementById('tableBody');
  document.getElementById('tableInfo').textContent =
    filtered.length + ' stock' + (filtered.length!==1?'s':'') + ' found';

  if (!filtered.length) {
    tableBody.innerHTML = '<tr><td colspan="11" style="text-align:center;padding:2rem;color:var(--ink3)">No results</td></tr>';
    return;
  }

  const rows = filtered.map((stock, idx) => {
    const change   = parseFloat(stock['LSTM_Return_%']           ||0);
    const vol      = parseFloat(stock['garch_volatility_percent']||0);
    const sig      = stock['Signal']||'HOLD';
    const sigCls   = sig==='BUY'?'sb-buy':sig==='HOLD'?'sb-hold':'sb-sell';
    const chgCls   = change>0?'tc-pos':change<0?'tc-neg':'tc-hold';
    const secCls   = 'sp-'+(stock['Sector']||'').toLowerCase();
    const src      = stock['has_lstm_model']?'✓ LSTM':stock['enhanced_source']||'Ensemble';
    const actual   = parseFloat(stock['Current_Price']       ||0).toFixed(2);
    const lstm     = parseFloat(stock['LSTM_Predicted_Price']||0).toFixed(2);
    const ens      = parseFloat(stock['Ensemble_Prediction'] ||0).toFixed(2);
    const rank     = idx<3?`[${idx+1}]`:`#${idx+1}`;
    const sym      = stock['Stock'].replace('.NS','');
    const dId      = 'dr-'+sym.replace(/[.\s]/g,'_');
    const macro    = stock['macro_variables']||{};
    const kpi      = stock['kpi_variables']  ||{};

    const macroHtml = Object.keys(macro).length
      ? Object.entries(macro).map(([k,v])=>`<div class="kpi-chip"><div class="kpi-label">${k}</div><div class="kpi-val">${typeof v==='number'?v.toFixed(2):v}</div></div>`).join('')
      : '<span style="color:var(--ink3);font-size:.8rem">No macro data</span>';
    const kpiHtml = Object.keys(kpi).length
      ? Object.entries(kpi).map(([k,v])=>`<div class="kpi-chip kpi-chip--green"><div class="kpi-label">${k}</div><div class="kpi-val">${typeof v==='number'?v.toFixed(2):v}</div></div>`).join('')
      : '<span style="color:var(--ink3);font-size:.8rem">No KPI data</span>';

    return `
      <tr class="stock-row" id="row-${dId}" onclick="toggleDetail('${dId}','${sym}')">
        <td><span class="expand-arrow" id="arr-${dId}">▶</span></td>
        <td class="td-rank">${rank}</td>
        <td><span class="td-sym">${sym}</span></td>
        <td><span class="s-pill ${secCls}">${stock['Sector']}</span></td>
        <td class="td-price">₹${actual}</td>
        <td class="td-price">₹${lstm}</td>
        <td class="td-price">₹${ens}</td>
        <td class="td-chg ${chgCls}">${change>0?'+':''}${change.toFixed(2)}%</td>
        <td>${vol.toFixed(2)}%</td>
        <td><span class="sb ${sigCls}">${sig}</span></td>
        <td><span class="sb sb-src">${src}</span></td>
      </tr>
      <tr class="detail-row" id="${dId}" style="display:none">
        <td colspan="11">
          <div class="detail-panel">
            <div class="detail-grid">
              <div><div class="detail-heading">📌 Macro — ${stock['Sector']}</div><div class="kpi-chips">${macroHtml}</div></div>
              <div><div class="detail-heading">📐 KPI — ${stock['Sector']}</div><div class="kpi-chips">${kpiHtml}</div></div>
            </div>
            <div style="margin-top:1rem;text-align:right">
              <button class="btn" style="font-size:.72rem;padding:.4rem 1rem" onclick="openModal('${sym}');event.stopPropagation()">🔍 Full Detail</button>
            </div>
          </div>
        </td>
      </tr>`;
  });
  tableBody.innerHTML = rows.join('');
}

function toggleDetail(dId, sym) {
  const row = document.getElementById(dId);
  const arr = document.getElementById('arr-'+dId);
  if (!row) return;
  const isOpen = row.style.display !== 'none';
  document.querySelectorAll('.detail-row').forEach(r=>r.style.display='none');
  document.querySelectorAll('.expand-arrow').forEach(a=>a.classList.remove('open'));
  document.querySelectorAll('.stock-row').forEach(r=>r.classList.remove('expanded'));
  if (!isOpen) {
    row.style.display='table-row';
    if(arr) arr.classList.add('open');
    const sr=document.getElementById('row-'+dId);
    if(sr) sr.classList.add('expanded');
  }
}

function openModal(sym) {
  const stock = allPredictions.find(p=>p.Stock.replace('.NS','')===sym);
  if (!stock) return;
  const change = parseFloat(stock['LSTM_Return_%']           ||0);
  const vol    = parseFloat(stock['garch_volatility_percent']||0);
  const actual = parseFloat(stock['Current_Price']           ||0);
  const lstm   = parseFloat(stock['LSTM_Predicted_Price']    ||0);
  const ens    = parseFloat(stock['Ensemble_Prediction']     ||0);
  const sig    = stock['Signal']||'HOLD';
  const macro  = stock['macro_variables']||{};
  const kpi    = stock['kpi_variables']  ||{};
  document.getElementById('modalTitle').textContent    = sym+' · '+stock['Sector'];
  document.getElementById('modalSubtitle').textContent = stock['has_lstm_model']?'✓ LSTM Model':(stock['enhanced_source']||'Ensemble');
  const chgCls = change>0?'positive':change<0?'negative':'hold-color';
  const sigCls = sig==='BUY'?'sb-buy':sig==='HOLD'?'sb-hold':'sb-sell';
  document.getElementById('modalStats').innerHTML=`
    <div class="modal-stat"><div class="modal-stat-label">Actual Price</div><div class="modal-stat-val">₹${actual.toFixed(2)}</div></div>
    <div class="modal-stat"><div class="modal-stat-label">LSTM Predicted</div><div class="modal-stat-val">₹${lstm.toFixed(2)}</div></div>
    <div class="modal-stat"><div class="modal-stat-label">Ensemble</div><div class="modal-stat-val">₹${ens.toFixed(2)}</div></div>
    <div class="modal-stat"><div class="modal-stat-label">LSTM Return</div><div class="modal-stat-val ${chgCls}">${change>=0?'+':''}${change.toFixed(2)}%</div></div>
    <div class="modal-stat"><div class="modal-stat-label">GARCH Vol</div><div class="modal-stat-val">${vol.toFixed(2)}%</div></div>
    <div class="modal-stat"><div class="modal-stat-label">Signal</div><div class="modal-stat-val"><span class="sb ${sigCls}">${sig}</span></div></div>`;
  document.getElementById('modalMacro').innerHTML = Object.keys(macro).length
    ? Object.entries(macro).map(([k,v])=>`<div class="kpi-chip"><div class="kpi-label">${k}</div><div class="kpi-val">${typeof v==='number'?v.toFixed(2):v}</div></div>`).join('')
    : '<span style="color:var(--ink3)">No macro data</span>';
  document.getElementById('modalKpi').innerHTML = Object.keys(kpi).length
    ? Object.entries(kpi).map(([k,v])=>`<div class="kpi-chip kpi-chip--green"><div class="kpi-label">${k}</div><div class="kpi-val">${typeof v==='number'?v.toFixed(2):v}</div></div>`).join('')
    : '<span style="color:var(--ink3)">No KPI data</span>';
  document.getElementById('stockModal').classList.add('open');
}

function closeModal(e){if(e.target===document.getElementById('stockModal'))closeModalDirect();}

function closeModalDirect(){document.getElementById('stockModal').classList.remove('open');}

document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModalDirect();});

// ── Boot ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', ()=>{
  // Auto-login if session active
  const u = sessionStorage.getItem('pp_user');
  if(u){
    document.getElementById('loginPage').style.display='none';
    const hu=document.getElementById('hdrUser');
    hu.textContent='👤 '+u; hu.style.display='';
    document.getElementById('logoutBtn').style.display='';
    loadData(); startAutoRefresh();
  }
  // otherwise wait for manual login
});