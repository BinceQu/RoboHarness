const b=document.body,btn=document.querySelector('.lang');
const pairs=[...document.querySelectorAll('.pair')];
function fit(){pairs.forEach(p=>{const t=p.querySelector('.pair-text'),f=p.querySelector('.pair-fig'),img=f.querySelector('img'),cap=f.querySelector('figcaption');
if(innerWidth<=760||!img.naturalWidth){p.style.gridTemplateColumns='';return}
const r=img.naturalHeight/img.naturalWidth,C=p.clientWidth;let w=.42*C;
for(let k=0;k<5;k++){p.style.gridTemplateColumns='minmax(0,1fr) '+w+'px';const ch=cap?cap.offsetHeight+10:0;
const lo=parseFloat(p.dataset.min||.28);w=Math.max(lo*C,Math.min(.56*C,(t.offsetHeight-ch-4)/r))}p.style.gridTemplateColumns='minmax(0,1fr) '+w+'px'})}
document.querySelectorAll('.pair-fig img').forEach(i=>{i.loading='eager';i.addEventListener('load',fit)});
addEventListener('resize',fit);addEventListener('load',fit);
function set(l){b.classList.toggle('l-zh',l==='zh');b.classList.toggle('l-en',l!=='zh');btn.textContent=l==='zh'?'EN':'中文';
document.documentElement.lang=l==='zh'?'zh-CN':'en';try{localStorage.setItem('rh-lang',l)}catch(e){}fit()}
btn.addEventListener('click',()=>set(b.classList.contains('l-zh')?'en':'zh'));
let saved='en';try{saved=localStorage.getItem('rh-lang')||'en'}catch(e){}set(saved);
document.querySelectorAll('a.zoom').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();
const box=document.createElement('div');box.className='lightbox';box.innerHTML='<img alt="">';box.firstChild.src=a.href;
const close=()=>{box.remove();document.removeEventListener('keydown',esc)};const esc=ev=>{if(ev.key==='Escape')close()};
box.addEventListener('click',close);document.addEventListener('keydown',esc);document.body.appendChild(box)}));
