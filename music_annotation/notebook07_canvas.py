CANVAS_HTML=r'''
<div id="hitl-app">
  <div id="hitl-toolbar">
    <button id="fitWidth">Fit width</button><button id="fitPage">Fit page</button>
    <button id="modePan" class="mode active">PAN</button><button id="modeAdd" class="mode">ADD</button><button id="modeErase" class="mode">ERASE</button>
    <button id="undo">Undo</button><button id="redo">Redo</button>
    <label>Brush <input id="brush" type="range" min="2" max="80" value="14"><span id="brushVal">14</span></label>
  </div>
  <div id="hitl-toggles">
    <label>Confidence <input id="threshold" type="range" min="0" max="100" value="20"><span id="thresholdVal">0.20</span></label>
    <label>Opacity <input id="opacity" type="range" min="0" max="100" value="70"><span id="opacityVal">70%</span></label>
    <label><input id="showOriginal" type="checkbox" checked>Original</label><label><input id="showPrediction" type="checkbox" checked>Prediction</label>
    <label><input id="showAdd" type="checkbox" checked>Additions</label><label><input id="showErase" type="checkbox" checked>Erasures</label><label><input id="finalOnly" type="checkbox">Final mask only</label>
  </div>
  <div id="canvas-wrap"><canvas id="hitl-canvas"></canvas></div>
  <div id="hitl-dock"><button id="saveAction" class="primary">Authorise reviewed mask</button></div>
</div>
'''

CANVAS_JS=r'''
async (...args)=>{
 const $=id=>document.getElementById(id), canvas=$('hitl-canvas'), ctx=canvas.getContext('2d',{alpha:false}), wrap=$('canvas-wrap');
 let page=null,img=null,draft=null,conf=null,addMask=null,eraseMask=null,scale=1,ox=0,oy=0,mode='pan',drawing=false,last=null;
 let ops=[],redoOps=[],activeOp=null,resizeTimer=null;
 const imgLoad=src=>new Promise((res,rej)=>{const i=new Image();i.onload=()=>res(i);i.onerror=rej;i.src=src});
 function maskCanvas(w,h){const c=document.createElement('canvas');c.width=w;c.height=h;return c}
 function setMode(m){mode=m;document.querySelectorAll('#hitl-toolbar .mode').forEach(b=>b.classList.toggle('active',b.id==='mode'+m[0].toUpperCase()+m.slice(1)));}
 function viewportHeight(){const vv=window.visualViewport;return vv?vv.height:window.innerHeight}
 function sizeToViewport(){const top=wrap.getBoundingClientRect().top;const available=Math.max(320,Math.min(720,viewportHeight()-top-74));wrap.style.height=available+'px';resize(false)}
 function resize(refit=false){const w=Math.max(280,Math.round(wrap.clientWidth)),h=Math.max(300,Math.round(wrap.clientHeight));const dpr=Math.min(window.devicePixelRatio||1,1.5);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);canvas.style.width=w+'px';canvas.style.height=h+'px'}if(refit&&page)fitPage();else render()}
 function fitPage(){if(!page)return;const w=wrap.clientWidth,h=wrap.clientHeight;scale=Math.min(w/page.width,h/page.height);ox=(w-page.width*scale)/2;oy=(h-page.height*scale)/2;render()}
 function fitWidth(){if(!page)return;scale=wrap.clientWidth/page.width;ox=0;oy=0;render()}
 function redrawMasks(){if(!page)return;for(const c of [addMask,eraseMask])c.getContext('2d').clearRect(0,0,c.width,c.height);for(const op of ops){const c=op.mode==='add'?addMask:eraseMask,x=c.getContext('2d');x.strokeStyle='white';x.fillStyle='white';x.lineWidth=op.brush;x.lineCap='round';x.lineJoin='round';if(op.points.length===1){x.beginPath();x.arc(op.points[0][0],op.points[0][1],op.brush/2,0,Math.PI*2);x.fill()}else{x.beginPath();op.points.forEach((p,i)=>i?x.lineTo(p[0],p[1]):x.moveTo(p[0],p[1]));x.stroke()}}render()}
 function point(e){const r=canvas.getBoundingClientRect();return {sx:e.clientX-r.left,sy:e.clientY-r.top,x:(e.clientX-r.left-ox)/scale,y:(e.clientY-r.top-oy)/scale}}
 function render(){const dpr=Math.min(window.devicePixelRatio||1,1.5),w=canvas.width/dpr,h=canvas.height/dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#9a9a9a';ctx.fillRect(0,0,w,h);if(!page||!img)return;ctx.save();ctx.translate(ox,oy);ctx.scale(scale,scale);if($('showOriginal').checked)ctx.drawImage(img,0,0,page.width,page.height);else{ctx.fillStyle='white';ctx.fillRect(0,0,page.width,page.height)}const alpha=+$('opacity').value/100;if($('showPrediction').checked){ctx.globalAlpha=alpha;ctx.drawImage(draft,0,0)}if($('showAdd').checked){ctx.globalAlpha=.72;ctx.drawImage(addMask,0,0)}if($('showErase').checked){ctx.globalAlpha=.72;ctx.drawImage(eraseMask,0,0)}ctx.globalAlpha=1;if($('finalOnly').checked){ctx.fillStyle='white';ctx.fillRect(0,0,page.width,page.height);ctx.globalCompositeOperation='source-over';ctx.drawImage(draft,0,0);ctx.drawImage(addMask,0,0);ctx.globalCompositeOperation='destination-out';ctx.drawImage(eraseMask,0,0);ctx.globalCompositeOperation='source-over'}ctx.restore()}
 function bridgeField(){return document.querySelector('#action-bridge textarea')||document.querySelector('#action-bridge input')}
 function sendAction(){const host=bridgeField();if(!host){alert('Action bridge unavailable. Reload the page.');return}const payload={action:'save',name:page?.name||'',add:addMask.toDataURL('image/png'),erase:eraseMask.toDataURL('image/png')};const proto=host.tagName==='TEXTAREA'?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(proto,'value').set.call(host,JSON.stringify(payload));host.dispatchEvent(new InputEvent('input',{bubbles:true,inputType:'insertText'}));host.dispatchEvent(new Event('change',{bubbles:true}))}
 async function ingest(raw){if(!raw)return;const next=JSON.parse(raw);const loaded=await Promise.all([imgLoad(next.image),imgLoad(next.draft),imgLoad(next.confidence)]);page=next;[img,draft,conf]=loaded;addMask=maskCanvas(page.width,page.height);eraseMask=maskCanvas(page.width,page.height);ops=[];redoOps=[];sizeToViewport();requestAnimationFrame(()=>requestAnimationFrame(fitPage))}
 window.hitlIngest=ingest;
 $('modePan').onclick=()=>setMode('pan');$('modeAdd').onclick=()=>setMode('add');$('modeErase').onclick=()=>setMode('erase');$('fitWidth').onclick=fitWidth;$('fitPage').onclick=fitPage;
 $('undo').onclick=()=>{if(!ops.length)return;redoOps.push(ops.pop());redrawMasks()};$('redo').onclick=()=>{if(!redoOps.length)return;ops.push(redoOps.pop());redrawMasks()};$('saveAction').onclick=sendAction;
 ['threshold','opacity','showOriginal','showPrediction','showAdd','showErase','finalOnly'].forEach(id=>$(id).oninput=()=>{if(id==='threshold')$('thresholdVal').textContent=(+$(id).value/100).toFixed(2);if(id==='opacity')$('opacityVal').textContent=$(id).value+'%';render()});$('brush').oninput=()=>$('brushVal').textContent=$('brush').value;
 canvas.addEventListener('contextmenu',e=>e.preventDefault());canvas.addEventListener('touchstart',e=>e.preventDefault(),{passive:false});canvas.addEventListener('touchmove',e=>e.preventDefault(),{passive:false});
 canvas.onpointerdown=e=>{e.preventDefault();canvas.setPointerCapture(e.pointerId);drawing=true;last=point(e);if(mode!=='pan'){activeOp={mode,brush:+$('brush').value,points:[[last.x,last.y]]}}};
 canvas.onpointermove=e=>{if(!drawing)return;e.preventDefault();const p=point(e);if(mode==='pan'){ox+=p.sx-last.sx;oy+=p.sy-last.sy;last=p;render()}else{activeOp.points.push([p.x,p.y]);last=p;ops.push(activeOp);redrawMasks();activeOp=ops.pop()}};
 canvas.onpointerup=e=>{e.preventDefault();if(activeOp){ops.push(activeOp);activeOp=null;redoOps=[];redrawMasks()}drawing=false;last=null};canvas.onpointercancel=canvas.onpointerup;
 canvas.onwheel=e=>{e.preventDefault();if(!page)return;if(e.ctrlKey){const p=point(e),f=e.deltaY<0?1.12:1/1.12;scale=Math.max(.08,Math.min(10,scale*f));ox=p.sx-p.x*scale;oy=p.sy-p.y*scale}else oy-=e.deltaY;render()};
 const schedule=()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>sizeToViewport(),120)};new ResizeObserver(schedule).observe(wrap);window.addEventListener('orientationchange',()=>setTimeout(()=>{sizeToViewport();fitPage()},250));if(window.visualViewport)window.visualViewport.addEventListener('resize',schedule);window.addEventListener('resize',schedule);
 setMode('pan');sizeToViewport();return [];
}
'''

CSS=r'''
#canvas-shell{height:auto!important;min-height:0!important}.bridge-offscreen{position:fixed!important;left:-10000px!important;top:0!important;width:1px!important;height:1px!important;overflow:hidden!important;opacity:0!important;pointer-events:none!important}
#hitl-app{display:flex;flex-direction:column;min-height:0;overflow:hidden;font-family:Arial,sans-serif}#hitl-toolbar,#hitl-toggles,#hitl-dock{display:flex;gap:6px;align-items:center;flex-wrap:wrap;padding:6px;background:#fff}#hitl-toolbar button,#hitl-dock button{min-height:42px;padding:7px 11px;border:1px solid #999;border-radius:6px;background:#f7f7f7}#hitl-toolbar button.active{background:#205493;color:#fff;border-color:#205493}#hitl-dock .primary{background:#205493;color:#fff}#canvas-wrap{position:relative;overflow:hidden;background:#999;touch-action:none;overscroll-behavior:contain;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}#hitl-canvas{display:block;touch-action:none;-webkit-user-select:none;user-select:none;-webkit-touch-callout:none}#hitl-toolbar label,#hitl-toggles label{font-size:14px;white-space:nowrap}
'''
