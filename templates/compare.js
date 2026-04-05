// ═══════════ COMPARE STATE ════════════════════════════════════════
let cmpSelected = [];
let cmpCharts   = {};
const CMP_COLORS=['#2563eb','#059669','#d97706','#7c3aed'];
const CMP_BG    =['rgba(37,99,235,.7)','rgba(5,150,105,.7)','rgba(217,119,6,.7)','rgba(124,58,237,.7)'];

function initCompareTab(){
  const sel = document.getElementById('cmpSelect');
  if(sel.options.length <= 1){
    sel.innerHTML = '<option value="">+ Add a stock…</option>';
    [...allPredictions]
      .sort((a,b)=>a.Stock.localeCompare(b.Stock))
      .forEach(p=>{
        const sym = p.Stock.replace('.NS','');
        const opt = document.createElement('option');
        opt.value = sym;
        opt.textContent = sym+' ('+p.Sector+')';
        sel.appendChild(opt);
      });
  }
}

function addCmpStock(){
  const sel = document.getElementById('cmpSelect');
  const val = sel.value;
  if(!val){ return; }
  sel.value = '';
  if(cmpSelected.includes(val) || cmpSelected.length >= 4) return;
  cmpSelected.push(val);
  renderCmpTags();
  updateCmpBtn();
}

function removeCmpStock(sym){
  cmpSelected = cmpSelected.filter(s=>s!==sym);
  renderCmpTags();
  updateCmpBtn();
  if(cmpSelected.length < 2){
    document.getElementById('cmpEmpty').style.display='';
    document.getElementById('cmpResults').style.display='none';
  }
}

function clearCompare(){
  cmpSelected=[];
  renderCmpTags();
  updateCmpBtn();
  document.getElementById('cmpEmpty').style.display='';
  document.getElementById('cmpResults').style.display='none';
  Object.values(cmpCharts).forEach(c=>c&&c.destroy&&c.destroy());
  cmpCharts={};
}

function renderCmpTags(){
  document.getElementById('cmpTags').innerHTML =
    cmpSelected.map(s=>`<span class="cmp-tag">${s}<span class="cmp-tag-x" onclick="removeCmpStock('${s}')">✕</span></span>`).join('');
}

function updateCmpBtn(){
  document.getElementById('cmpRunBtn').disabled = cmpSelected.length < 2;
}

async function runCompare(){
  if(cmpSelected.length < 2) return;
  const btn = document.getElementById('cmpRunBtn');
  btn.textContent='Loading…'; btn.disabled=true;
  try{
    const r = await fetch('/api/compare?symbols='+encodeURIComponent(cmpSelected.join(',')));
    const d = await r.json();
    if(d.status!=='success'||!d.data||!d.data.length){
      alert('No data returned for selected stocks'); return;
    }
    renderCmpResults(d.data);
  } catch(e){ alert('Error: '+e.message); }
  finally{ btn.textContent='Compare →'; btn.disabled=false; updateCmpBtn(); }
}

function sf2(v){ return isFinite(parseFloat(v))?parseFloat(v):0; }

function renderCmpResults(data){
  document.getElementById('cmpEmpty').style.display='none';
  document.getElementById('cmpResults').style.display='';

  // Destroy old charts
  Object.values(cmpCharts).forEach(c=>c&&c.destroy&&c.destroy());
  cmpCharts={};

  const n = data.length;
  const maxRet = Math.max(...data.map(d=>sf2(d.lstm_return)));

  // ── Stat cards ──────────────────────────────────────────────────
  const cards = document.getElementById('cmpCards');
  cards.className = 'cmp-cards n'+n;
  cards.innerHTML = data.map((d,i)=>{
    const ret = sf2(d.lstm_return);
    const vol = sf2(d.volatility);
    const retCls = ret>0?'pos':ret<0?'neg':'hld';
    const sigCls = d.signal==='BUY'?'sb-buy':d.signal==='HOLD'?'sb-hold':'sb-sell';
    const secCls = 'sp-'+(d.sector||'').toLowerCase();
    const isWin  = d.lstm_return===maxRet;
    const color  = CMP_COLORS[i%CMP_COLORS.length];
    return `<div class="cmp-card${isWin?' winner':''}" style="--cmp-clr:${color}">
      ${isWin?'<div class="cmp-win-badge">★ BEST RETURN</div>':''}
      <div class="cmp-sym">${d.stock}</div>
      <div class="cmp-meta">
        <span class="s-pill ${secCls}">${d.sector}</span>
        <span class="sb ${sigCls}">${d.signal}</span>
        ${d.has_model?'<span class="sb sb-src">✓ Model</span>':''}
      </div>
      <div class="cmp-metrics">
        <div class="cmp-m"><div class="cmp-m-lbl">Actual Price</div><div class="cmp-m-val">₹${sf2(d.current_price).toFixed(2)}</div></div>
        <div class="cmp-m"><div class="cmp-m-lbl">LSTM Price</div><div class="cmp-m-val">₹${sf2(d.lstm_price).toFixed(2)}</div></div>
        <div class="cmp-m"><div class="cmp-m-lbl">LSTM Return</div><div class="cmp-m-val ${retCls}">${ret>=0?'+':''}${ret.toFixed(2)}%</div></div>
        <div class="cmp-m"><div class="cmp-m-lbl">GARCH Vol</div><div class="cmp-m-val">${vol.toFixed(2)}%</div></div>
        <div class="cmp-m"><div class="cmp-m-lbl">Ensemble</div><div class="cmp-m-val">₹${sf2(d.ensemble_price).toFixed(2)}</div></div>
        <div class="cmp-m"><div class="cmp-m-lbl">Ens Change</div><div class="cmp-m-val ${sf2(d.ens_change)>=0?'pos':'neg'}">${sf2(d.ens_change)>=0?'+':''}${sf2(d.ens_change).toFixed(2)}%</div></div>
      </div>
    </div>`;
  }).join('');

  // ── Price chart ──────────────────────────────────────────────────
  const pCtx = document.getElementById('cmpPriceChart');
  cmpCharts.price = new Chart(pCtx,{
    type:'bar',
    data:{
      labels:data.map(d=>d.stock),
      datasets:[
        {label:'Actual ₹',   data:data.map(d=>sf2(d.current_price)),  backgroundColor:CMP_BG[0],borderColor:CMP_COLORS[0],borderWidth:1.5,borderRadius:4},
        {label:'LSTM ₹',     data:data.map(d=>sf2(d.lstm_price)),     backgroundColor:CMP_BG[1],borderColor:CMP_COLORS[1],borderWidth:1.5,borderRadius:4},
        {label:'Ensemble ₹', data:data.map(d=>sf2(d.ensemble_price)), backgroundColor:CMP_BG[2],borderColor:CMP_COLORS[2],borderWidth:1.5,borderRadius:4},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{font:{size:11,weight:'bold'},color:'#3a4a60',padding:12}},
        tooltip:{backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,
          callbacks:{label:ctx=>`${ctx.dataset.label}: ₹${ctx.raw.toLocaleString('en-IN',{maximumFractionDigits:2})}`}}
      },
      scales:{
        x:{ticks:{font:{size:11},color:'#7a8da8'},grid:{display:false}},
        y:{ticks:{font:{size:11},color:'#7a8da8',callback:v=>'₹'+v.toLocaleString('en-IN',{maximumFractionDigits:0})},grid:{color:'#e8ecf4'}}
      }
    }
  });

  // ── Return & Volatility chart ────────────────────────────────────
  const rvCtx = document.getElementById('cmpRetVolChart');
  cmpCharts.retVol = new Chart(rvCtx,{
    type:'bar',
    data:{
      labels:data.map(d=>d.stock),
      datasets:[
        {label:'LSTM Return %', data:data.map(d=>sf2(d.lstm_return)),
         backgroundColor:data.map(d=>sf2(d.lstm_return)>=0?'rgba(5,150,105,.75)':'rgba(220,38,38,.75)'),
         borderColor:data.map(d=>sf2(d.lstm_return)>=0?'#059669':'#dc2626'),
         borderWidth:1.5,borderRadius:4,yAxisID:'y'},
        {label:'Volatility %', data:data.map(d=>sf2(d.volatility)),
         backgroundColor:'rgba(217,119,6,.6)',borderColor:'#d97706',
         borderWidth:1.5,borderRadius:4,yAxisID:'y1'},
      ]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{labels:{font:{size:11,weight:'bold'},color:'#3a4a60',padding:12}},
        tooltip:{backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,
          callbacks:{label:ctx=>`${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`}}
      },
      scales:{
        x:{ticks:{font:{size:11},color:'#7a8da8'},grid:{display:false}},
        y:{position:'left', title:{display:true,text:'Return %',font:{size:11,weight:'bold'},color:'#0f1b2d'},
           ticks:{font:{size:10},color:'#7a8da8',callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%'},grid:{color:'#e8ecf4'}},
        y1:{position:'right',title:{display:true,text:'Volatility %',font:{size:11,weight:'bold'},color:'#0f1b2d'},
            ticks:{font:{size:10},color:'#7a8da8',callback:v=>v.toFixed(1)+'%'},grid:{display:false}}
      }
    }
  });

  // ── Radar chart ──────────────────────────────────────────────────
  const allRet = data.map(d=>sf2(d.lstm_return));
  const allVol = data.map(d=>sf2(d.volatility));
  const allAct = data.map(d=>sf2(d.current_price));
  const allEns = data.map(d=>sf2(d.ens_change));

  const norm = (arr, invert=false) => {
    const mn=Math.min(...arr), mx=Math.max(...arr);
    if(mx===mn) return arr.map(()=>50);
    const n = arr.map(v=>Math.round((v-mn)/(mx-mn)*100));
    return invert ? n.map(v=>100-v) : n;
  };

  const radarDatasets = data.map((d,i)=>({
    label: d.stock,
    data: [
      norm(allRet)[i],
      norm(allVol, true)[i],
      norm(allAct)[i],
      norm(allEns)[i],
      d.signal==='BUY'?100:d.signal==='HOLD'?50:10,
    ],
    backgroundColor: CMP_COLORS[i]+'26',
    borderColor: CMP_COLORS[i],
    borderWidth: 2.5,
    pointBackgroundColor: CMP_COLORS[i],
    pointRadius: 5,
  }));

  const rCtx = document.getElementById('cmpRadarChart');
  cmpCharts.radar = new Chart(rCtx,{
    type:'radar',
    data:{
      labels:['Return','Stability\n(Low Vol)','Price','Ens. Outlook','Signal'],
      datasets: radarDatasets,
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{labels:{font:{size:11,weight:'bold'},color:'#3a4a60',padding:14}},
        tooltip:{backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,
          callbacks:{label:ctx=>`${ctx.dataset.label}: ${ctx.raw} (normalised)`}}
      },
      scales:{r:{
        min:0, max:100,
        ticks:{display:false},
        grid:{color:'rgba(221,227,238,.7)'},
        pointLabels:{font:{size:11,weight:'bold'},color:'#3a4a60'},
        angleLines:{color:'rgba(221,227,238,.5)'},
      }}
    }
  });

  // ── Comparison table ─────────────────────────────────────────────
  const fields = [
    {key:'current_price', label:'Actual Price',    fmt:v=>'₹'+sf2(v).toFixed(2), best:'max'},
    {key:'lstm_price',    label:'LSTM Price',       fmt:v=>'₹'+sf2(v).toFixed(2), best:'max'},
    {key:'ensemble_price',label:'Ensemble Price',   fmt:v=>'₹'+sf2(v).toFixed(2), best:'max'},
    {key:'lstm_return',   label:'LSTM Return %',    fmt:v=>(sf2(v)>=0?'+':'')+sf2(v).toFixed(2)+'%', best:'max'},
    {key:'ens_change',    label:'Ensemble Change %',fmt:v=>(sf2(v)>=0?'+':'')+sf2(v).toFixed(2)+'%', best:'max'},
    {key:'volatility',    label:'GARCH Vol %',      fmt:v=>sf2(v).toFixed(2)+'%', best:'min'},
    {key:'signal',        label:'Signal',           fmt:v=>v, best:null},
    {key:'sector',        label:'Sector',           fmt:v=>v, best:null},
  ];

  const thead = '<thead><tr><th>Metric</th>'+data.map(d=>'<th>'+d.stock+'</th>').join('')+'</tr></thead>';
  const tbody = '<tbody>'+fields.map(f=>{
    const vals = data.map(d=>d[f.key]);
    let bestIdx = -1;
    if(f.best==='max'){
      const mx=Math.max(...vals.map(v=>sf2(v)));
      bestIdx=vals.findIndex(v=>sf2(v)===mx);
    } else if(f.best==='min'){
      const mn=Math.min(...vals.map(v=>sf2(v)));
      bestIdx=vals.findIndex(v=>sf2(v)===mn);
    }
    const cells = vals.map((v,i)=>{
      const txt=f.fmt(v);
      return i===bestIdx?`<td><span class="best-cell">${txt}</span></td>`:`<td><span class="mono">${txt}</span></td>`;
    }).join('');
    return `<tr><td style="font-weight:600;color:var(--ink2);font-size:.78rem">${f.label}</td>${cells}</tr>`;
  }).join('');

  // ════════════ MACRO DATA COMPARISON ════════════════════════════════
  const macroKeys = data[0]?.macro ? Object.keys(data[0].macro) : [];
  let macroRows = '';
  if(macroKeys.length){
    macroRows += `<tr><td colspan="${n+1}" style="background:var(--blue-bg);font-size:.7rem;font-weight:700;color:var(--blue);padding:.5rem 1rem;letter-spacing:.5px">📌 MACRO VARIABLES</td></tr>`;
    macroKeys.forEach(k=>{
      const vals = data.map(d=>(d.macro&&d.macro[k]!=null)?d.macro[k]:'—');
      const cells = vals.map(v=>`<td><span class="mono">${typeof v==='number'?v.toFixed(2):v}</span></td>`).join('');
      macroRows += `<tr><td style="font-weight:600;color:var(--ink2);font-size:.78rem">${k}</td>${cells}</tr>`;
    });
  }

  // ════════════ KPI DATA COMPARISON - COLLECT ALL UNIQUE KPI KEYS ════════════════════════════════════════════
  // Get all unique KPI keys from all stocks across all sectors
  const allKpiKeys = new Set();
  data.forEach(d => {
    if(d.kpi) {
      Object.keys(d.kpi).forEach(k => allKpiKeys.add(k));
    }
  });
  
  const sortedKpiKeys = Array.from(allKpiKeys).sort();
  
  let kpiRows = '';
  if(sortedKpiKeys.length){
    kpiRows += `<tr><td colspan="${n+1}" style="background:var(--green-bg);font-size:.7rem;font-weight:700;color:var(--green);padding:.5rem 1rem;letter-spacing:.5px">📐 KPI INDICATORS</td></tr>`;
    
    sortedKpiKeys.forEach(k=>{
      const vals = data.map(d=>{
        if(d.kpi && d.kpi[k] != null) {
          return typeof d.kpi[k] === 'number' ? d.kpi[k].toFixed(2) : d.kpi[k];
        }
        return '—';
      });
      
      const cells = vals.map(v=>`<td><span class="mono">${v}</span></td>`).join('');
      kpiRows += `<tr><td style="font-weight:600;color:var(--ink2);font-size:.78rem">${k}</td>${cells}</tr>`;
    });
  }

  document.getElementById('cmpTable').innerHTML = thead+'<tbody>'+fields.map(f=>{
    const vals=data.map(d=>d[f.key]);
    let bestIdx=-1;
    if(f.best==='max'){const mx=Math.max(...vals.map(v=>sf2(v)));bestIdx=vals.findIndex(v=>sf2(v)===mx);}
    else if(f.best==='min'){const mn=Math.min(...vals.map(v=>sf2(v)));bestIdx=vals.findIndex(v=>sf2(v)===mn);}
    const cells=vals.map((v,i)=>{const txt=f.fmt(v);return i===bestIdx?`<td><span class="best-cell">${txt}</span></td>`:`<td><span class="mono">${txt}</span></td>`;}).join('');
    return `<tr><td style="font-weight:600;color:var(--ink2);font-size:.78rem">${f.label}</td>${cells}</tr>`;
  }).join('')+macroRows+kpiRows+'</tbody>';
}