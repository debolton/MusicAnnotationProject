CANVAS_HTML=r"""
<div id="hitl-v12">
  <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
    <button id="fit-width">Fit width</button><button id="fit-page">Fit page</button>
    <button data-mode="pan">PAN</button><button data-mode="add">ADD</button><button data-mode="erase">ERASE</button>
    <button id="undo">Undo</button><button id="redo">Redo</button>
    <label>Brush <input id="brush" type="range" min="2" max="80" value="14"></label>
    <label>Confidence <input id="conf" type="range" min="20" max="100" value="20"></label>
    <label>Opacity <input id="opacity" type="range" min="5" max="95" value="70"></label>
    <label><input id="original" type="checkbox" checked>Original</label><label><input id="prediction" type="checkbox" checked>Prediction</label>
    <label><input id="additions" type="checkbox" checked>Additions</label><label><input id="erasures" type="checkbox" checked>Erasures</label>
    <label><input id="final-only" type="checkbox">Final mask only</label>
  </div>
  <div id="canvas-wrap" style="height:72vh;overflow:hidden;border:1px solid #bbb;background:#ddd;touch-action:none"><canvas id="hitl-canvas"></canvas></div>
</div>
"""
CANVAS_JS=r"""
async (...args) => {
 const root=document.getElementById('hitl-v12'); if(!root)return;
 const canvas=root.querySelector('#hitl-canvas'),wrap=root.querySelector('#canvas-wrap'),ctx=canvas.getContext('2d');
 let state={mode:'pan',scale:1,ox:0,oy:0,img:null,draft:null,conf:null,add:null,erase:null,undo:[],redo:[],drawing:false,panning:false,last:null};
 const bridge=(id)=>document.querySelector('#'+id+' textarea');
 const bLoad=bridge('page-package'), bAdd=bridge('add-mask-out'), bErase=bridge('erase-mask-out');
 function img(src){return new Promise((res,rej)=>{let i=new Image();i.onload=()=>res(i);i.onerror=rej;i.src=src})}
 function blank(w,h){let c=document.createElement('canvas');c.width=w;c.height=h;return c}
 function fitWidth(){if(!state.img)return;state.scale=wrap.clientWidth/state.img.width;state.ox=0;state.oy=0;draw()}
 function fitPage(){if(!state.img)return;state.scale=Math.min(wrap.clientWidth/state.img.width,wrap.clientHeight/state.img.height);state.ox=(wrap.clientWidth-state.img.width*state.scale)/2;state.oy=(wrap.clientHeight-state.img.height*state.scale)/2;draw()}
 function clamp(){if(!state.img)return;let sw=state.img.width*state.scale,sh=state.img.height*state.scale;state.ox=Math.min(0,Math.max(wrap.clientWidth-sw,state.ox));state.oy=Math.min(0,Math.max(wrap.clientHeight-sh,state.oy));if(sw<wrap.clientWidth)state.ox=(wrap.clientWidth-sw)/2;if(sh<wrap.clientHeight)state.oy=(wrap.clientHeight-sh)/2}
 function maskFrom(src,w,h){let c=blank(w,h),x=c.getContext('2d');x.drawImage(src,0,0,w,h);return c}
 function draw(){if(!state.img)return;canvas.width=wrap.clientWidth;canvas.height=wrap.clientHeight;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.save();ctx.translate(state.ox,state.oy);ctx.scale(state.scale,state.scale);let W=state.img.width,H=state.img.height;ctx.drawImage(state.img,0,0);let finalOnly=root.querySelector('#final-only').checked;if(finalOnly){let tmp=blank(W,H),t=tmp.getContext('2d');t.fillStyle='black';t.fillRect(0,0,W,H);t.globalCompositeOperation='source-over';t.drawImage(state.draft,0,0);t.drawImage(state.add,0,0);t.globalCompositeOperation='destination-out';t.drawImage(state.erase,0,0);ctx.globalAlpha=1;ctx.drawImage(tmp,0,0)}else{let op=+root.querySelector('#opacity').value/100;if(root.querySelector('#prediction').checked){ctx.globalAlpha=op;ctx.drawImage(state.draft,0,0)}if(root.querySelector('#additions').checked){ctx.globalAlpha=.75;ctx.drawImage(state.add,0,0)}if(root.querySelector('#erasures').checked){ctx.globalAlpha=.75;ctx.drawImage(state.erase,0,0)}ctx.globalAlpha=1}ctx.restore()}
 function snapshot(){state.undo.push([state.add.toDataURL('image/png'),state.erase.toDataURL('image/png')]);if(state.undo.length>30)state.undo.shift();state.redo=[]}
 async function restore(pair){state.add=maskFrom(await img(pair[0]),state.img.width,state.img.height);state.erase=maskFrom(await img(pair[1]),state.img.width,state.img.height);draw();sync()}
 function sync(){if(bAdd){bAdd.value=state.add.toDataURL('image/png');bAdd.dispatchEvent(new Event('input',{bubbles:true}))}if(bErase){bErase.value=state.erase.toDataURL('image/png');bErase.dispatchEvent(new Event('input',{bubbles:true}))}}
 async function loadPackage(){if(!bLoad||!bLoad.value)return;let p;try{p=JSON.parse(bLoad.value)}catch(e){return}if(!p||!p.image)return;state.img=await img(p.image);let d=await img(p.draft);state.draft=maskFrom(d,state.img.width,state.img.height);state.add=blank(state.img.width,state.img.height);state.erase=blank(state.img.width,state.img.height);fitPage();sync()}
 root.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>state.mode=b.dataset.mode);root.querySelector('#fit-width').onclick=fitWidth;root.querySelector('#fit-page').onclick=fitPage;root.querySelector('#undo').onclick=async()=>{if(!state.undo.length)return;state.redo.push([state.add.toDataURL('image/png'),state.erase.toDataURL('image/png')]);await restore(state.undo.pop())};root.querySelector('#redo').onclick=async()=>{if(!state.redo.length)return;state.undo.push([state.add.toDataURL('image/png'),state.erase.toDataURL('image/png')]);await restore(state.redo.pop())};root.querySelectorAll('input').forEach(x=>x.oninput=draw);
 canvas.onpointerdown=e=>{if(!state.img)return;state.last=[e.clientX,e.clientY];if(state.mode==='pan'){state.panning=true}else{snapshot();state.drawing=true;canvas.setPointerCapture(e.pointerId)}};
 canvas.onpointermove=e=>{if(state.panning){state.ox+=e.clientX-state.last[0];state.oy+=e.clientY-state.last[1];state.last=[e.clientX,e.clientY];clamp();draw();return}if(!state.drawing)return;let r=canvas.getBoundingClientRect(),x=(e.clientX-r.left-state.ox)/state.scale,y=(e.clientY-r.top-state.oy)/state.scale,px=(state.last[0]-r.left-state.ox)/state.scale,py=(state.last[1]-r.top-state.oy)/state.scale,target=state.mode==='add'?state.add:state.erase,t=target.getContext('2d');t.strokeStyle=state.mode==='add'?'rgba(0,0,255,1)':'rgba(0,255,0,1)';t.lineWidth=+root.querySelector('#brush').value/state.scale;t.lineCap='round';t.beginPath();t.moveTo(px,py);t.lineTo(x,y);t.stroke();state.last=[e.clientX,e.clientY];draw()};
 canvas.onpointerup=()=>{if(state.drawing){state.drawing=false;sync()}state.panning=false};canvas.onpointercancel=canvas.onpointerup;
 canvas.onwheel=e=>{if(!state.img)return;e.preventDefault();if(e.ctrlKey){let r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,ix=(mx-state.ox)/state.scale,iy=(my-state.oy)/state.scale,f=e.deltaY<0?1.15:1/1.15;state.scale=Math.max(.08,Math.min(12,state.scale*f));state.ox=mx-ix*state.scale;state.oy=my-iy*state.scale}else state.oy-=e.deltaY;clamp();draw()};
 if(bLoad){new MutationObserver(loadPackage).observe(bLoad,{attributes:true,childList:true,subtree:true});bLoad.addEventListener('input',loadPackage);loadPackage()}
 return [];
}
"""
