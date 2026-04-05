// ═══════════ CHARTS ════════════════════════════════════════════════════
let chartInstances = {};
const dc = (key) => {if(chartInstances[key]){chartInstances[key].destroy();delete chartInstances[key];}}
const sf = (v, d=0) => isFinite(parseFloat(v)) ? parseFloat(v) : d;

const SIG_COLOR  = { BUY:'#059669', HOLD:'#6366f1', SELL:'#dc2626' };
const SIG_BG     = { BUY:'rgba(5,150,105,.75)', HOLD:'rgba(99,102,241,.75)', SELL:'rgba(220,38,38,.75)' };
const SEC_COLOR  = { BANK:'#0891b2', IT:'#7c3aed', ENERGY:'#d97706' };
const SEC_BG     = { BANK:'rgba(8,145,178,.7)', IT:'rgba(124,58,237,.7)', ENERGY:'rgba(217,119,6,.7)' };
const TF  = { size:12, weight:'bold' };
const TK  = { size:11 };
const TC  = '#7a8da8';
const GC  = '#e8ecf4';

function renderAllCharts(){
  chart1_returnRanking();
  chart2_riskReturn();
  chart3_pricePrediction();
  chart4_sectorRadar();
  chart5_signalDonut();
  chart6_candlestick();
  chart7_priceTimeline();
  chart8_volatilityTimeline();
  chart9_signalEvolution();
  chart10_returnDistribution();
}

function chart1_returnRanking() {
  dc('c1');
  const sorted = [...allPredictions].sort((a,b)=>sf(b['LSTM_Return_%'])-sf(a['LSTM_Return_%']));
  const labels = sorted.map(p=>p.Stock.replace('.NS',''));
  const values = sorted.map(p=>sf(p['LSTM_Return_%']));
  const colors = sorted.map(p=>SIG_BG[p.Signal||'HOLD']);
  const borders= sorted.map(p=>SIG_COLOR[p.Signal||'HOLD']);
  chartInstances.c1 = new Chart(document.getElementById('returnRankChart'),{
    type:'bar',
    data:{labels,datasets:[{label:'LSTM Return %',data:values,backgroundColor:colors,borderColor:borders,borderWidth:1.5,borderRadius:4}]},
    options:{
      indexAxis:'y', responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,callbacks:{label:ctx=>{const p=sorted[ctx.dataIndex];return [`Return: ${ctx.raw>=0?'+':''}${ctx.raw.toFixed(2)}%`, `Signal: ${p.Signal}`, `Sector: ${p.Sector}`];}}}},
      scales:{x:{title:{display:true,text:'LSTM Return (%)',font:TF,color:'#0f1b2d'},ticks:{font:TK,color:TC,callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%'},grid:{color:GC}},y:{ticks:{font:{size:11},color:TC},grid:{display:false}}},
    }
  });
}

function chart2_riskReturn() {
  dc('c2');
  const datasets = ['BANK','IT','ENERGY'].map(sector=>{
    const sp = allPredictions.filter(p=>p.Sector===sector);
    return {label:sector,data:sp.map(p=>({x: sf(p['garch_volatility_percent']),y: sf(p['LSTM_Return_%']),r: Math.max(6, Math.min(28, sf(p['Current_Price'])/80)),stock: p.Stock.replace('.NS',''),signal: p.Signal||'HOLD',})),backgroundColor:SEC_BG[sector],borderColor:SEC_COLOR[sector],borderWidth:1.5,};
  });
  chartInstances.c2 = new Chart(document.getElementById('riskReturnChart'),{
    type:'bubble',data:{datasets},options:{responsive:true, maintainAspectRatio:false,plugins:{legend:{labels:{font:TF,color:'#3a4a60',padding:14}},tooltip:{backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,callbacks:{label:ctx=>{const d=ctx.raw;return [`${d.stock} [${d.signal}]`,`Vol: ${d.x.toFixed(2)}%`,`Return: ${d.y>=0?'+':''}${d.y.toFixed(2)}%`];}}}},scales:{x:{title:{display:true,text:'GARCH Volatility (%)',font:TF,color:'#0f1b2d'},ticks:{font:TK,color:TC},grid:{color:GC}},y:{title:{display:true,text:'LSTM Return (%)',font:TF,color:'#0f1b2d'},ticks:{font:TK,color:TC,callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%'},grid:{color:GC}}}}
  });
}

function chart3_pricePrediction() {
  dc('c3');
  const top8 = [...allPredictions].sort((a,b)=>sf(b['LSTM_Return_%'])-sf(a['LSTM_Return_%'])).slice(0,8);
  const labels  = top8.map(p=>p.Stock.replace('.NS',''));
  const actual  = top8.map(p=>sf(p['Current_Price']));
  const lstm    = top8.map(p=>sf(p['LSTM_Predicted_Price']));
  const ens     = top8.map(p=>sf(p['Ensemble_Prediction']));
  chartInstances.c3 = new Chart(document.getElementById('pricePredChart'),{
    type:'bar',data:{labels,datasets:[{label:'Actual ₹',   data:actual, backgroundColor:'rgba(37,99,235,.7)', borderColor:'#2563eb', borderWidth:1.5, borderRadius:4},{label:'LSTM ₹',     data:lstm,   backgroundColor:'rgba(5,150,105,.7)', borderColor:'#059669', borderWidth:1.5, borderRadius:4},{label:'Ensemble ₹', data:ens,    backgroundColor:'rgba(217,119,6,.7)', borderColor:'#d97706', borderWidth:1.5, borderRadius:4},]},
    options:{responsive:true, maintainAspectRatio:false,interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{font:TF,color:'#3a4a60',padding:14}},tooltip:{backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,callbacks:{label:ctx=>`${ctx.dataset.label}: ₹${ctx.raw.toLocaleString('en-IN',{maximumFractionDigits:2})}`}}},scales:{x:{ticks:{font:{size:11},color:TC},grid:{display:false}},y:{ticks:{font:TK,color:TC,callback:v=>'₹'+v.toLocaleString('en-IN',{maximumFractionDigits:0})},grid:{color:GC}}}}
  });
}

function chart4_sectorRadar() {
  dc('c4');
  const h = allAnalysis.heatmap || {};
  const sectors = ['BANK','IT','ENERGY'];
  const raw = {return:   sectors.map(s=>(h[s]||{}).return   || 0),winRate:  sectors.map(s=>(h[s]||{}).win_rate  || 0),vol:      sectors.map(s=>(h[s]||{}).volatility|| 0),count:    sectors.map(s=>(h[s]||{}).stock_count||0),};
  const norm = (arr)=>{const mn=Math.min(...arr), mx=Math.max(...arr);if(mx===mn) return arr.map(()=>50);return arr.map(v=>Math.round((v-mn)/(mx-mn)*100));};
  const bankData  = [norm(raw.return)[0],  norm(raw.winRate)[0],  100-norm(raw.vol)[0],  norm(raw.count)[0]];
  const itData    = [norm(raw.return)[1],  norm(raw.winRate)[1],  100-norm(raw.vol)[1],  norm(raw.count)[1]];
  const energyData= [norm(raw.return)[2],  norm(raw.winRate)[2],  100-norm(raw.vol)[2],  norm(raw.count)[2]];
  const axes = ['Avg Return','Win Rate','Stability\n(Low Vol)','Stock Count'];
  chartInstances.c4 = new Chart(document.getElementById('sectorRadarChart'),{
    type:'radar',data:{labels:axes,datasets:[{label:'BANK',  data:bankData,   backgroundColor:'rgba(8,145,178,.15)',  borderColor:'#0891b2', borderWidth:2.5, pointBackgroundColor:'#0891b2', pointRadius:5},{label:'IT',    data:itData,     backgroundColor:'rgba(124,58,237,.15)', borderColor:'#7c3aed', borderWidth:2.5, pointBackgroundColor:'#7c3aed', pointRadius:5},{label:'ENERGY',data:energyData, backgroundColor:'rgba(217,119,6,.15)',  borderColor:'#d97706', borderWidth:2.5, pointBackgroundColor:'#d97706', pointRadius:5},]},
    options:{responsive:true, maintainAspectRatio:false,plugins:{legend:{labels:{font:TF,color:'#3a4a60',padding:14}},tooltip:{backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,callbacks:{label:ctx=>{const sector=ctx.dataset.label;const labels2=['Return','Win Rate','Stability','Stock Count'];return `${sector} ${labels2[ctx.dataIndex]}: ${ctx.raw}`;}}}},scales:{r:{min:0, max:100,ticks:{display:false},grid:{color:'rgba(221,227,238,.8)'},pointLabels:{font:{size:11,weight:'bold'},color:'#3a4a60'},angleLines:{color:'rgba(221,227,238,.6)'},}}}
  });
}

function chart5_signalDonut() {
  dc('c5');
  const s = allAnalysis.statistics || {};
  const buy  = s.buy_signals  || 0;
  const hold = s.hold_signals || 0;
  const sell = s.sell_signals || 0;
  if (!buy && !hold && !sell) return;
  document.getElementById('donutWinRate').textContent = (s.win_rate||0)+'%';
  chartInstances.c5 = new Chart(document.getElementById('signalDonutChart'),{
    type:'doughnut',
    data:{
      labels:[`BUY (${buy})`,`HOLD (${hold})`,`SELL (${sell})`],
      datasets:[{
        data:[buy,hold,sell],
        backgroundColor:['rgba(5,150,105,.85)','rgba(99,102,241,.85)','rgba(220,38,38,.85)'],
        borderColor:['#059669','#6366f1','#dc2626'],
        borderWidth:2,
        hoverOffset:8,
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      cutout:'65%',
      plugins:{
        legend:{position:'bottom',labels:{font:TF,color:'#3a4a60',padding:20,boxWidth:14}},
        tooltip:{
          backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,
          callbacks:{
            label:ctx=>{
              const pct=((ctx.raw/(buy+hold+sell))*100).toFixed(1);
              return `${ctx.label}: ${pct}% of portfolio`;
            }
          }
        }
      }
    }
  });
}

function chart6_candlestick() {
  dc('c6');
  const candles = allAnalysis.candle_data || [];
  if (!candles.length) return;

  const labels     = candles.map(c=>c.stock);
  const wickData   = candles.map(c=>[c.low, c.high]);
  const sigColors  = candles.map(c=>SIG_COLOR[c.signal||'HOLD']);
  const sigBgAlpha = candles.map(c=>SIG_BG[c.signal||'HOLD']);

  const candleBodyPlugin = {
    id:'candleBody',
    afterDraw(chart){
      const {ctx,scales:{x:xs,y:ys},chartArea} = chart;
      if (!xs||!ys) return;
      candles.forEach((c,i)=>{
        const xc = xs.getPixelForValue(i);
        const yo = ys.getPixelForValue(c.open);
        const yc = ys.getPixelForValue(c.close);
        const bw = Math.min(xs.width/candles.length*0.55, 40);
        const bTop = Math.min(yo,yc);
        const bH   = Math.max(Math.abs(yc-yo), 4);
        ctx.save();
        ctx.fillStyle   = sigColors[i];
        ctx.globalAlpha = 0.8;
        ctx.fillRect(xc-bw/2, bTop, bw, bH);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = sigColors[i];
        ctx.lineWidth   = 1.5;
        ctx.strokeRect(xc-bw/2, bTop, bw, bH);
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(xc-bw/2-4, yo); ctx.lineTo(xc-bw/2, yo); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(xc+bw/2, yc); ctx.lineTo(xc+bw/2+4, yc); ctx.stroke();
        ctx.restore();
      });
    }
  };

  chartInstances.c6 = new Chart(document.getElementById('candleChart'),{
    type:'bar',
    data:{
      labels,
      datasets:[{
        label:'Price Range',
        data:wickData,
        backgroundColor:sigBgAlpha,
        borderColor:sigColors,
        borderWidth:1.5,
        borderSkipped:false,
        barPercentage:0.65,
        categoryPercentage:0.85,
      }]
    },
    options:{
      responsive:true, 
      maintainAspectRatio:false,
      animation:{duration:700},
      plugins:{
        legend:{display:false},
        tooltip:{
          backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:2,
          titleFont:{size:13,weight:'bold'}, bodyFont:{size:12}, displayColors:false,
          callbacks:{
            title:ctx=>labels[ctx[0].dataIndex]||'',
            label:ctx=>{
              const c=candles[ctx.dataIndex];
              return [
                `Signal : ${c.signal}`,
                `Open   : ₹${c.open.toFixed(2)}  (Current)`,
                `Close  : ₹${c.close.toFixed(2)}  (LSTM)`,
                `High   : ₹${c.high.toFixed(2)}`,
                `Low    : ₹${c.low.toFixed(2)}`,
                `Return : ${c.return>=0?'+':''}${c.return.toFixed(2)}%`,
              ];
            }
          }
        }
      },
      scales:{
        x:{
          ticks:{
            font:{size:11,weight:'600'},
            color:TC,
            maxRotation:45,
            minRotation:0,
            padding:10
          },
          grid:{display:false}
        },
        y:{
          title:{display:true,text:'Price (₹)',font:TF,color:'#0f1b2d'},
          ticks:{
            font:TK,
            color:TC,
            callback:v=>'₹'+v.toLocaleString('en-IN',{maximumFractionDigits:0})
          },
          grid:{color:GC},
        }
      }
    },
    plugins:[candleBodyPlugin]
  });
}

function chart7_priceTimeline() {
  dc('c7');
  fetch('/api/timeseries/prices').then(r=>r.json()).then(d=>{
    if(d.status!=='success') {
      console.warn('Chart 7: API returned error',d.message);
      return;
    }
    const data = d.data || {};
    const dates = data.dates || [];
    const stocks = data.stocks || [];
    const tsData = data.data || {};
    
    if(!dates.length || !stocks.length) {
      console.warn('Chart 7: No dates or stocks found');
      return;
    }
    
    const datasets = [];
    const colors = ['#2563eb','#059669','#d97706','#7c3aed','#0891b2'];
    
    stocks.forEach((stock,si)=>{
      const stockKey = stock.endsWith('.NS') ? stock : stock + '.NS';
      const stockData = tsData[stockKey];
      
      if(!stockData) {
        console.warn(`No data for stock: ${stockKey}`);
        return;
      }
      
      const actualData = stockData.actual || [];
      const lstmData = stockData.lstm || [];
      const ensembleData = stockData.ensemble || [];
      
      if(actualData.length > 0) {
        datasets.push({
          label:`${stock} Actual`,
          data:actualData,
          borderColor:colors[si % colors.length],
          backgroundColor:'transparent',
          borderWidth:2.5,
          borderDash:[],
          tension:0.2,
          fill:false,
          pointRadius:1,
          pointBackgroundColor:colors[si % colors.length],
        });
      }
      
      if(lstmData.length > 0) {
        datasets.push({
          label:`${stock} LSTM`,
          data:lstmData,
          borderColor:colors[si % colors.length],
          backgroundColor:'transparent',
          borderWidth:2,
          borderDash:[5,5],
          tension:0.2,
          fill:false,
          pointRadius:0,
        });
      }
      
      if(ensembleData.length > 0) {
        datasets.push({
          label:`${stock} Ensemble`,
          data:ensembleData,
          borderColor:colors[si % colors.length],
          backgroundColor:'transparent',
          borderWidth:1.5,
          borderDash:[2,2],
          tension:0.2,
          fill:false,
          pointRadius:0,
        });
      }
    });

    if(datasets.length === 0) {
      console.warn('Chart 7: No datasets created');
      return;
    }

    chartInstances.c7 = new Chart(document.getElementById('priceTimelineChart'),{
      type:'line',
      data:{
        labels:dates,
        datasets:datasets
      },
      options:{
        responsive:true, 
        maintainAspectRatio:false,
        interaction:{mode:'index',intersect:false},
        plugins:{
          legend:{
            labels:{
              font:{size:10},
              color:'#7a8da8',
              padding:10
            },
            maxHeight:100,
            display:true
          },
          tooltip:{
            backgroundColor:'rgba(15,27,45,.95)',
            borderColor:'#2563eb',
            borderWidth:1,
            callbacks:{
              label:ctx=>{
                const value = ctx.raw;
                return `${ctx.dataset.label}: ₹${parseFloat(value).toFixed(2)}`;
              }
            }
          }
        },
        scales:{
          x:{
            ticks:{
              font:{size:11},
              color:'#7a8da8',
              maxRotation:45,
              minRotation:0
            },
            grid:{color:'#e8ecf4',display:false}
          },
          y:{
            title:{
              display:true,
              text:'Price (₹)',
              font:{size:12,weight:'bold'},
              color:'#0f1b2d'
            },
            ticks:{
              font:{size:11},
              color:'#7a8da8',
              callback:v=>'₹'+parseFloat(v).toLocaleString('en-IN',{maximumFractionDigits:0})
            },
            grid:{color:'#e8ecf4'}
          }
        }
      }
    });
  }).catch(e=>{
    console.error('Chart 7 fetch error:',e);
  });
}

function chart8_volatilityTimeline() {
  dc('c8');
  fetch('/api/timeseries/volatility').then(r=>r.json()).then(d=>{
    if(d.status!=='success') return;
    const data = d.data || {};
    const dates = data.dates || [];
    const sectorData = data.data || {};
    
    if(!dates.length) return;
    
    const datasets = [
      {label:'BANK',data:sectorData.BANK||[],borderColor:'#0891b2',backgroundColor:'rgba(8,145,178,.1)',borderWidth:2.5,tension:0.3,fill:true},
      {label:'IT',data:sectorData.IT||[],borderColor:'#7c3aed',backgroundColor:'rgba(124,58,237,.1)',borderWidth:2.5,tension:0.3,fill:true},
      {label:'ENERGY',data:sectorData.ENERGY||[],borderColor:'#d97706',backgroundColor:'rgba(217,119,6,.1)',borderWidth:2.5,tension:0.3,fill:true},
    ];

    chartInstances.c8 = new Chart(document.getElementById('volatilityTimelineChart'),{
      type:'line',
      data:{labels:dates,datasets},
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{
          legend:{labels:{font:TF,color:TC,padding:14}},
          tooltip:{backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,callbacks:{label:ctx=>`${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`}}
        },
        scales:{
          x:{ticks:{font:TK,color:TC},grid:{color:GC}},
          y:{title:{display:true,text:'Volatility (%)',font:TF,color:'#0f1b2d'},ticks:{font:TK,color:TC,callback:v=>v.toFixed(1)+'%'},grid:{color:GC}}
        }
      }
    });
  }).catch(e=>console.warn('Chart 8 error:',e));
}

function chart9_signalEvolution() {
  dc('c9');
  fetch('/api/timeseries/signals').then(r=>r.json()).then(d=>{
    if(d.status!=='success') return;
    const data = d.data || {};
    const dates = data.dates || [];
    const buyData = data.buy || [];
    const holdData = data.hold || [];
    const sellData = data.sell || [];
    
    if(!dates.length) return;
    
    const datasets = [
      {label:'BUY',data:buyData,borderColor:'#059669',backgroundColor:'rgba(5,150,105,.6)',borderWidth:1.5,fill:true,tension:0.3},
      {label:'HOLD',data:holdData,borderColor:'#6366f1',backgroundColor:'rgba(99,102,241,.6)',borderWidth:1.5,fill:true,tension:0.3},
      {label:'SELL',data:sellData,borderColor:'#dc2626',backgroundColor:'rgba(220,38,38,.6)',borderWidth:1.5,fill:true,tension:0.3},
    ];

    chartInstances.c9 = new Chart(document.getElementById('signalEvolutionChart'),{
      type:'line',
      data:{labels:dates,datasets},
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{
          legend:{labels:{font:TF,color:TC,padding:14}},
          tooltip:{backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,mode:'index',intersect:false}
        },
        scales:{
          x:{stacked:false,ticks:{font:TK,color:TC},grid:{color:GC}},
          y:{stacked:false,title:{display:true,text:'Count',font:TF,color:'#0f1b2d'},ticks:{font:TK,color:TC},grid:{color:GC}}
        }
      }
    });
  }).catch(e=>console.warn('Chart 9 error:',e));
}

function chart10_returnDistribution() {
  dc('c10');
  fetch('/api/timeseries/returns').then(r=>r.json()).then(d=>{
    if(d.status!=='success') return;
    const data = d.data || {};
    const dates = data.dates || [];
    const avgData = data.avg_return || [];
    const maxData = data.max_return || [];
    const minData = data.min_return || [];
    
    if(!dates.length) return;
    
    const datasets = [
      {label:'Max Return',data:maxData,borderColor:'#059669',backgroundColor:'rgba(5,150,105,.15)',borderWidth:1.5,fill:false,tension:0.3,pointRadius:2},
      {label:'Avg Return',data:avgData,borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,.3)',borderWidth:3,fill:false,tension:0.3,pointRadius:3},
      {label:'Min Return',data:minData,borderColor:'#dc2626',backgroundColor:'rgba(220,38,38,.15)',borderWidth:1.5,fill:false,tension:0.3,pointRadius:2},
    ];

    chartInstances.c10 = new Chart(document.getElementById('returnDistributionChart'),{
      type:'line',
      data:{labels:dates,datasets},
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{
          legend:{labels:{font:TF,color:TC,padding:14}},
          tooltip:{
            backgroundColor:'rgba(15,27,45,.95)',borderColor:'#2563eb',borderWidth:1,
            callbacks:{label:ctx=>`${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`}
          }
        },
        scales:{
          x:{ticks:{font:TK,color:TC},grid:{color:GC}},
          y:{
            title:{display:true,text:'Return (%)',font:TF,color:'#0f1b2d'},
            ticks:{font:TK,color:TC,callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%'},
            grid:{color:GC}
          }
        }
      }
    });
  }).catch(e=>console.warn('Chart 10 error:',e));
}