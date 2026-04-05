// ═══════════ LOGIN PAGE ════════════════════════════════════════════════
const USERS = { admin:'admin123', analyst:'pulse2026' };

(function initParticles(){
  const colors = ['#3b82f6','#8b5cf6','#06b6d4','#10b981','#f59e0b'];
  const con = document.getElementById('lpParticles');
  for (let i=0;i<20;i++){
    const p=document.createElement('div');
    p.className='lp';
    const sz=Math.random()*8+3;
    p.style.cssText=`width:${sz}px;height:${sz}px;left:${Math.random()*100}%;`+
      `background:${colors[i%colors.length]};`+
      `animation-duration:${Math.random()*14+8}s;`+
      `animation-delay:${Math.random()*8}s`;
    con.appendChild(p);
  }
})();

['lgUser','lgPass'].forEach(id=>{
  document.getElementById(id).addEventListener('keydown',e=>{ if(e.key==='Enter') doLogin(); });
});

function doLogin(){
  const u=document.getElementById('lgUser').value.trim().toLowerCase();
  const p=document.getElementById('lgPass').value;
  const err=document.getElementById('lgError');
  const btn=document.getElementById('lgBtn');
  if(USERS[u]&&USERS[u]===p){
    err.classList.remove('show');
    sessionStorage.setItem('pp_user',u);
    const page=document.getElementById('loginPage');
    page.classList.add('fade-out');
    setTimeout(()=>page.style.display='none',420);
    const hu=document.getElementById('hdrUser');
    hu.textContent='👤 '+u; hu.style.display='';
    document.getElementById('logoutBtn').style.display='';
    loadData(); startAutoRefresh();
  } else {
    err.classList.add('show');
    document.getElementById('lgPass').value='';
    document.getElementById('lgPass').focus();
    btn.textContent='Sign In →';
    btn.disabled=false;
  }
}

function doLogout(){
  sessionStorage.removeItem('pp_user');
  location.reload();
}