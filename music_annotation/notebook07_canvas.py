CANVAS_HTML=r'''
<div id="hitl-app">
  <div id="hitl-toolbar">
    <button data-mode="pan" class="active">Pan</button>
    <button data-mode="add">ADD</button>
    <button data-mode="erase">ERASE</button>
    <button id="undo">Undo</button><button id="redo">Redo</button>
    <button id="fitWidth">Fit width</button><button id="fitPage">Fit page</button>
    <label>Brush <input id="brush" type="range" min="2" max="80" value="14"><span id="brushVal">14</span></label>
    <label>Confidence <input id="threshold" type="range" min="20" max="100" value="20"><span id="thresholdVal">0.20</span></label>
    <label>Opacity <input id="opacity" type="range" min="5" max="95" value="70"><span id="opacityVal">70%</span></label>
  </div>
  <div id="hitl-toggles">
    <label><input id="showImage" type="checkbox" checked>Original</label>
    <label><input id="showPrediction" type="checkbox" checked>Prediction</label>
    <label><input id="showAdd" type="checkbox" checked>Additions</label>
    <label><input id="showErase" type="checkbox" checked>Erasures</label>
    <label><input id="finalOnly" type="checkbox">Final mask only</label>
    <span id="zoomLabel">100%</span>
  </div>
  <div id="canvas-wrap"><canvas id="hitl-canvas"></canvas></div>
  <div id="hitl-dock">
    <button id="saveAction" class="primary">Authorise reviewed mask</button>
  </div>
</div>
'''

CANVAS_JS=r'''

 const canvas=document.getElementById('hitl-canvas'),ctx=canvas.getContext('2d',{willReadFrequently:true}),wrap=document.getElementById('canvas-wrap');
 let page=null,img=null,draft=null,conf=null,addMask=null,eraseMask=null,scale=1,ox=0,oy=0,mode='pan',drawing=false,last=null,space=false,history=[],future=[];
 const $=id=>document.getElementById(id); const imgLoad=src=>new Promise((res,rej)=>{const i=new Image();i.onload=()=>res(i);i.onerror=rej;i.src=src});
 function sizeToViewport(){const shell=document.getElementById('canvas-shell'),app=document.getElementById('hitl-app');if(!shell||!app)return;const top=shell.getBoundingClientRect().top;const reserved=92;const available=Math.max(560,window.innerHeight-top-reserved);shell.style.height=available+'px';app.style.height=available+'px';wrap.style.height=Math.max(360,available-$('hitl-toolbar').offsetHeight-$('hitl-toggles').offsetHeight)+'px';resize()}
 function maskCanvas(w,h){const c=document.createElement('canvas');c.width=w;c.height=h;return c}
 function snapshot(){history.push({a:addMask.toDataURL(),e:eraseMask.toDataURL()});if(history.length>30)history.shift();future=[]}
 async function restore(s){const [a,e]=await Promise.all([imgLoad(s.a),imgLoad(s.e)]);addMask.getContext('2d').clearRect(0,0,addMask.width,addMask.height);eraseMask.getContext('2d').clearRect(0,0,eraseMask.width,eraseMask.height);addMask.getContext('2d').drawImage(a,0,0);eraseMask.getContext('2d').drawImage(e,0,0);render()}
 function resize(){canvas.width=wrap.clientWidth*devicePixelRatio;canvas.height=wrap.clientHeight*devicePixelRatio;canvas.style.width=wrap.clientWidth+'px';canvas.style.height=wrap.clientHeight+'px';render()}
 function fitPage(){if(!page)return;scale=Math.min(wrap.clientWidth/page.width,wrap.clientHeight/page.height);ox=(wrap.clientWidth-page.width*scale)/2;oy=(wrap.clientHeight-page.height*scale)/2;render()}
 function fitWidth(){if(!page)return;scale=wrap.clientWidth/page.width;ox=0;oy=0;render()}
 function render(){ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);ctx.clearRect(0,0,wrap.clientWidth,wrap.clientHeight);if(!page)return;ctx.save();ctx.translate(ox,oy);ctx.scale(scale,scale);
   if($('showImage').checked)ctx.drawImage(img,0,0);
   const threshold=+$('threshold').value/100,opacity=+$('opacity').value/100,finalOnly=$('finalOnly').checked;
   const tmp=maskCanvas(page.width,page.height),t=tmp.getContext('2d');t.drawImage(conf,0,0);const cd=t.getImageData(0,0,page.width,page.height);const dtx=maskCanvas(page.width,page.height).getContext('2d');dtx.drawImage(draft,0,0);const dd=dtx.getImageData(0,0,page.width,page.height).data;const ax=addMask.getContext('2d').getImageData(0,0,page.width,page.height).data,ex=eraseMask.getContext('2d').getImageData(0,0,page.width,page.height).data,out=t.createImageData(page.width,page.height);
   for(let i=0;i<out.data.length;i+=4){const c=cd.data[i]/255,isDraft=dd[i]>0,isAdd=ax[i+3]>0,isErase=ex[i+3]>0,final=(isDraft||isAdd)&&!isErase;let show=false,a=0;
     if(finalOnly){show=final;a=opacity*(isAdd?1:Math.max(.08,(c-.2)/.8));}
     else if($('showPrediction').checked&&isDraft&&!isErase&&c>=threshold){show=true;a=opacity*(.08+.92*Math.max(0,(c-threshold)/(1-threshold||1)));}
     if(show){out.data[i]=255;out.data[i+1]=0;out.data[i+2]=0;out.data[i+3]=Math.round(255*a)}
   }t.putImageData(out,0,0);ctx.drawImage(tmp,0,0);
   if(!finalOnly&&$('showAdd').checked){ctx.globalAlpha=.55;ctx.drawImage(addMask,0,0)}if(!finalOnly&&$('showErase').checked){ctx.globalAlpha=.55;ctx.drawImage(eraseMask,0,0)}ctx.restore();$('zoomLabel').textContent=Math.round(scale*100)+'%'}
 function point(ev){const r=canvas.getBoundingClientRect();return {sx:ev.clientX-r.left,sy:ev.clientY-r.top,x:(ev.clientX-r.left-ox)/scale,y:(ev.clientY-r.top-oy)/scale}}
 function stroke(target,a,b,colour){const c=target.getContext('2d');c.strokeStyle=colour;c.lineWidth=+$('brush').value/scale;c.lineCap='round';c.lineJoin='round';c.beginPath();c.moveTo(a.x,a.y);c.lineTo(b.x,b.y);c.stroke()}
 canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);drawing=true;last=point(e);if(mode!=='pan'&&!space)snapshot()};canvas.onpointermove=e=>{if(!drawing)return;const p=point(e);if(mode==='pan'||space||e.buttons===4){ox+=p.sx-last.sx;oy+=p.sy-last.sy}else if(mode==='add')stroke(addMask,last,p,'rgba(0,110,255,1)');else stroke(eraseMask,last,p,'rgba(0,190,90,1)');last=p;render()};canvas.onpointerup=()=>{drawing=false;last=null};
 canvas.onwheel=e=>{
   if(!page)return;

   // Ordinary wheel: bounded vertical navigation through the rendered page.
   if(!e.ctrlKey){
     e.preventDefault();
     oy-=e.deltaY;
     const displayedHeight=page.height*scale;
     if(displayedHeight<=wrap.clientHeight){
       oy=(wrap.clientHeight-displayedHeight)/2;
     }else{
       oy=Math.max(wrap.clientHeight-displayedHeight,Math.min(0,oy));
     }
     render();
     return;
   }

   // Ctrl+wheel: unrestricted pointer-centred canvas zoom.
   e.preventDefault();
   const p=point(e),factor=e.deltaY<0?1.15:1/1.15,newScale=Math.max(.08,Math.min(12,scale*factor));
   ox=p.sx-p.x*newScale;
   oy=p.sy-p.y*newScale;
   scale=newScale;
   render();
 };
 window.addEventListener('keydown',e=>{if(e.code==='Space')space=true;if((e.ctrlKey||e.metaKey)&&e.key==='z')$('undo').click()});window.addEventListener('keyup',e=>{if(e.code==='Space')space=false});window.addEventListener('resize',sizeToViewport);
 document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;document.querySelectorAll('[data-mode]').forEach(x=>x.classList.toggle('active',x===b))});
 $('undo').onclick=async()=>{if(!history.length)return;future.push({a:addMask.toDataURL(),e:eraseMask.toDataURL()});await restore(history.pop())};$('redo').onclick=async()=>{if(!future.length)return;history.push({a:addMask.toDataURL(),e:eraseMask.toDataURL()});await restore(future.pop())};$('fitWidth').onclick=fitWidth;$('fitPage').onclick=fitPage;
 ['showImage','showPrediction','showAdd','showErase','finalOnly','threshold','opacity'].forEach(id=>$(id).oninput=()=>{if(id==='threshold')$('thresholdVal').textContent=(+$('threshold').value/100).toFixed(2);if(id==='opacity')$('opacityVal').textContent=$('opacity').value+'%';render()});$('brush').oninput=()=>$('brushVal').textContent=$('brush').value;
 function bridgeField(){return document.querySelector('#action-bridge textarea')||document.querySelector('#action-bridge input');}
 function notesField(){return document.querySelector('#notes-field textarea')||document.querySelector('#notes-field input');}
 function sendAction(action){const host=bridgeField();if(!host){alert('Action bridge unavailable. Reload the page.');return;}const payload={action:action,name:page?.name||'',notes:notesField()?.value||''};if(action==='save'){payload.add=addMask.toDataURL('image/png');payload.erase=eraseMask.toDataURL('image/png');}const proto=host.tagName==='TEXTAREA'?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(proto,'value').set.call(host,JSON.stringify(payload));host.dispatchEvent(new InputEvent('input',{bubbles:true,inputType:'insertText'}));host.dispatchEvent(new Event('change',{bubbles:true}));}
 $('saveAction').onclick=()=>sendAction('save');
 async function ingest(raw){if(!raw)return;page=JSON.parse(raw);[img,draft,conf]=await Promise.all([imgLoad(page.image),imgLoad(page.draft),imgLoad(page.confidence)]);addMask=maskCanvas(page.width,page.height);eraseMask=maskCanvas(page.width,page.height);history=[];future=[];sizeToViewport();requestAnimationFrame(fitWidth)}
 window.hitlIngest=ingest;
 new MutationObserver(()=>{const el=document.querySelector('#page-package textarea');if(el&&el.value&&el.value!==window._lastPackage){window._lastPackage=el.value;ingest(el.value)}}).observe(document.body,{subtree:true,childList:true,characterData:true,attributes:true});
 setTimeout(()=>{sizeToViewport();const el=document.querySelector('#page-package textarea');if(el?.value)ingest(el.value)},500);setTimeout(sizeToViewport,1200);

'''

CSS=r'''
.gradio-container{max-width:100vw!important;padding:8px 8px 84px!important}
#canvas-shell{height:auto!important;min-height:0!important;padding:0!important;overflow:hidden!important}
#canvas-shell>div,#canvas-shell .html-container,#canvas-shell .prose{height:100%!important;min-height:0!important;padding:0!important;margin:0!important;overflow:hidden!important}
#hitl-app{height:100%;display:flex;flex-direction:column;background:#202124;color:#fff;font-family:Arial,sans-serif}
#hitl-toolbar,#hitl-toggles{display:flex;align-items:center;gap:8px;padding:7px 10px;background:#303134;flex-wrap:wrap}
#hitl-toolbar button,#hitl-dock button{padding:8px 13px;border:0;border-radius:7px;background:#5f6368;color:#fff;cursor:pointer}#hitl-toolbar button.active{background:#1a73e8}#hitl-dock button.primary{background:#ff6d00}
#hitl-toolbar label,#hitl-toggles label{display:flex;align-items:center;gap:4px;font-size:13px}
#canvas-wrap{flex:1;min-height:0;overflow:hidden;background:#555;cursor:crosshair}#hitl-canvas{display:block}
#hitl-dock{position:fixed;bottom:14px;left:50%;transform:translateX(-50%);z-index:9999;background:rgba(32,33,36,.96);padding:10px 14px;border-radius:15px;box-shadow:0 5px 22px rgba(0,0,0,.45);display:flex;gap:10px}
.bridge-offscreen{position:fixed!important;left:-10000px!important;top:-10000px!important;width:4px!important;height:4px!important;opacity:.01!important;pointer-events:none!important}#hitl-toolbar,#hitl-toggles,#hitl-toolbar label,#hitl-toggles label,#zoomLabel,#brushVal,#thresholdVal,#opacityVal{color:#f7f7f7!important}#hitl-toolbar button,#hitl-dock button{color:#fff!important;font-weight:700!important}footer{display:none!important}
'''
