import {TYPES,empty,fc,normalize,riparianStudy,measurements,example} from './map-core.mjs';
import * as maplibregl from './vendor/maplibre-gl.mjs';
const $=id=>document.getElementById(id),t=globalThis.turf;
const tell=(message,error=false)=>{$('message').textContent=message;$('message').classList.toggle('error',error);};
let map;
try {map=new maplibregl.Map({container:'map',center:[-98,39],zoom:3,maxZoom:20,style:{version:8,sources:{},layers:[{id:'canvas',type:'background',paint:{'background-color':'#dce8da'}}]},attributionControl:true});}
catch(e){tell('The GIS renderer needs WebGL. Enable hardware graphics or use another browser. Saved projects are unchanged.',true);throw e;}
map.addControl(new maplibregl.NavigationControl(),'top-right');map.addControl(new maplibregl.ScaleControl({unit:'metric'}));
await new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(Error('Map initialization timed out. Reload to try again.')),20000);map.once('load',()=>{clearTimeout(timeout);resolve();});}).catch(e=>{tell(e.message,true);throw e;});
let state=empty(),dirty=false,selected=null,drawing=null;
const visibility={};
function text(tag,value,parent){const e=document.createElement(tag);e.textContent=value;parent.append(e);return e;}
function upsert(id,data,color,weight=2,opacity=.18){
 const source=map.getSource(id);if(source){source.setData(data);return;}
 map.addSource(id,{type:'geojson',data});
 map.addLayer({id:id+'-fill',type:'fill',source:id,filter:['==',['geometry-type'],'Polygon'],paint:{'fill-color':color,'fill-opacity':opacity}});
 map.addLayer({id:id+'-line',type:'line',source:id,filter:['!=',['geometry-type'],'Point'],paint:{'line-color':color,'line-width':weight}});
 map.addLayer({id:id+'-point',type:'circle',source:id,filter:['==',['geometry-type'],'Point'],paint:{'circle-radius':6,'circle-color':color,'circle-stroke-width':2,'circle-stroke-color':'#fff'}});
}
function setVisible(k,value){visibility[k]=value;for(const suffix of ['-fill','-line','-point'])if(map.getLayer(k+suffix))map.setLayoutProperty(k+suffix,'visibility',value?'visible':'none');}
function fitFeatures(features,maxZoom=17){if(!features.length)return tell('No visible features. Load data or enable a layer.');const bounds=t.bbox(fc(features));map.fitBounds([[bounds[0],bounds[1]],[bounds[2],bounds[3]]],{padding:40,maxZoom,duration:500});}
function fit(){fitFeatures(Object.entries(state).filter(([k])=>visibility[k]!==false).flatMap(([,v])=>v));}
function selectFeature(f){selected=f;const host=$('feature-detail');host.replaceChildren();text('h2',String(f.properties.name||f.properties.dfm_id),host);const remove=text('button','Remove from working map',host);remove.onclick=()=>{if(!confirm('Remove this feature from the working map? Saved data changes only when you save the project.'))return;const k=f.properties.dfm_layer;state[k]=state[k].filter(x=>x.properties.dfm_id!==f.properties.dfm_id);if(k==='boundary'||k==='waterways')state.buffer=[];dirty=true;redraw();};const dl=document.createElement('dl');host.append(dl);for(const [key,value]of Object.entries(f.properties)){text('dt',key.replace(/^dfm_/,'').replaceAll('_',' '),dl);text('dd',typeof value==='object'?JSON.stringify(value):String(value),dl);}}
function redraw(){
 for(const id of ['scenario-buffer','scenario-retention'])map.getSource(id)?.setData(fc([]));
 $('layers').replaceChildren();$('feature-list').replaceChildren();
 for(const [k,meta]of Object.entries(TYPES)){
  state[k]??=[];const row=document.createElement('div');row.className='layer-row';const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.id=`show-${k}`;input.checked=visibility[k]!==false;label.append(input);const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=meta.color;label.append(swatch);text('span',meta.label,label);row.append(label);text('span',String(state[k].length),row);$('layers').append(row);
  upsert(k,fc(state[k]),meta.color,k==='roads'?4:k==='waterways'?3:2,k==='boundary'?0:k==='buffer'?.3:.18);setVisible(k,input.checked);input.onchange=()=>setVisible(k,input.checked);
  for(const f of state[k]){const b=text('button',`${meta.label} · ${f.properties.name||f.properties.dfm_id}`,$('feature-list'));b.onclick=()=>{selectFeature(f);fitFeatures([f]);};}
 }
 const features=Object.values(state).flat(),examples=features.filter(f=>f.properties.dfm_evidence==='Illustrative example').length;
 $('dataset-status').textContent=!features.length?'Empty map · load your landscape':examples===features.length?'Illustrative example · not a surveyed site':examples?'Mixed example and imported data · not a site assessment':'Imported map · sources not independently verified';
 $('feature-total').textContent=`${features.length} features · ${Object.values(state).filter(f=>f.length).length} populated layers`;$('list-count').textContent=`(${features.length})`;
 $('export').disabled=!features.length;$('clear-buffer').disabled=!state.buffer.length;$('buffer').disabled=!state.waterways.length||!state.boundary.length;
 try{const m=measurements(state,t);const fmt=(v,u)=>v===null?'—':`${v.toLocaleString(undefined,{maximumFractionDigits:2})} ${u}`;$('area').textContent=fmt(m.boundaryHa,'ha');$('buffer-area').textContent=fmt(m.bufferHa,'ha');$('roads-length').textContent=fmt(m.roadsKm,'km');$('water-length').textContent=fmt(m.waterKm,'km');}catch(e){for(const id of ['area','buffer-area','roads-length','water-length'])$(id).textContent='Unavailable';tell(e.message,true);}
 $('feature-detail').replaceChildren();text('h2','Feature details',$('feature-detail'));text('p','Select a feature on the map or in the feature list to inspect its source.',$('feature-detail'));
}
function replace(next,message){state=next;dirty=false;redraw();fit();tell(message);window.dispatchEvent(new Event('dfm-reset'));}
$('new-project').onclick=()=>{if(dirty&&!confirm('Clear this map? Export first to keep your changes.'))return;replace(empty(),'New map ready. Load a planning boundary, roads, and waterways.');};
$('example').onclick=()=>{if(dirty&&!confirm('Replace this map with the example? Export first to keep your changes.'))return;replace(example(),'Illustrative data loaded. These features do not represent a real site.');};
$('fit').onclick=fit;
$('file').onchange=async e=>{
 const file=e.target.files[0];if(!file)return;const kind=$('layer-type').value;
 try{if(file.size>5*1024*1024)throw Error('Choose a file smaller than 5 MB.');const parsed=JSON.parse(await file.text());const incoming=normalize(parsed,kind,file.name,t);let next=structuredClone(state);
  if(kind==='project'){if(dirty&&!confirm('Replace the current map with this exported map?'))return;next=incoming;}
  else{if(next[kind].length&&!confirm(`Replace the current ${TYPES[kind].label.toLowerCase()} layer?`))return;next[kind]=incoming[kind];if(kind==='waterways'||kind==='boundary')next.buffer=[];}
  const all=Object.values(next).flat();if(all.length>3000)throw Error('The map supports at most 3,000 features.');
  state=next;dirty=true;if(kind==='project')window.dispatchEvent(new CustomEvent('dfm-import',{detail:parsed.dfm_project||{}}));redraw();fit();tell(`${file.name} loaded. ${kind==='waterways'||kind==='boundary'?'The previous buffer study was cleared. ':''}Export to keep this map.`);
 }catch(error){tell(`Import failed: ${error.message} Your existing map was kept.`,true);}finally{e.target.value='';}
};
$('buffer').onclick=()=>{try{state.buffer=[riparianStudy(state,Number($('buffer-width').value),t)];dirty=true;redraw();$('show-buffer').checked=true;setVisible('buffer',true);tell('Buffer study created and clipped to the planning boundary. This is a proposed zone, not measured forest.');}catch(e){tell(e.message,true);}};
$('buffer-width').oninput=()=>{if(state.buffer.length)tell('Distance changed. Choose Create buffer to replace the existing study; its current settings remain in Feature details.');};
$('clear-buffer').onclick=()=>{state.buffer=[];dirty=true;redraw();tell('Buffer study removed.');};
$('export').onclick=()=>{const data=fc(Object.values(state).flat());data.dfm_project={...window.dfmProjectMeta?.(),version:2,exported_at:new Date().toISOString(),scope:'Mapping workspace; not an optimized or validated forest plan',coordinates:'WGS84 longitude/latitude',measurement_method:'Turf 7.2.0 approximate geodesic measurements'};const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/geo+json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='dfm-landscape.geojson';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);tell('Map exported with layer and source labels. Use Restore exported map to reopen it.');};
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
redraw();tell('Explore the imagery, go to coordinates, or open a saved project. Load example uses fictional geometry.');

window.dfm={getState:()=>structuredClone(state),isDirty:()=>dirty,markDirty:()=>{dirty=true;},saved:()=>{dirty=false;},setState(next){stopDraw();state={...empty(),...next};dirty=false;redraw();fit();},addObservation(f){state.observations.push(f);dirty=true;redraw();selectFeature(f);},showResult(result){upsert('scenario-buffer',fc(result.geometry.features.filter(f=>f.properties.dfm_layer==='buffer')),'#168eab',3,.2);upsert('scenario-retention',fc(result.geometry.features.filter(f=>f.properties.dfm_layer!=='buffer')),'#a95fff',3,.3);fitFeatures(result.geometry.features);tell('Saved scenario overlay: blue study zone and purple retained forest. Editing clears the overlay.');},tell};
function stopDraw(){drawing=null;map.getSource('sketch')?.setData(fc([]));map.doubleClickZoom.enable();$('draw-finish').disabled=$('draw-cancel').disabled=true;$('draw-start').disabled=false;}
$('draw-start').onclick=()=>{const kind=$('layer-type').value;if(!TYPES[kind]||kind==='buffer')return tell('Choose an input layer to draw.',true);drawing={kind,points:[]};$('draw-finish').disabled=$('draw-cancel').disabled=false;$('draw-start').disabled=true;map.doubleClickZoom.disable();tell('Click the map to place points, then choose Finish.');};
map.on('click',e=>{if(drawing){drawing.points.push([e.lngLat.lng,e.lngLat.lat]);const p=drawing.points;upsert('sketch',fc([{type:'Feature',properties:{},geometry:{type:p.length===1?'Point':'LineString',coordinates:p.length===1?p[0]:p}}]),'#c79917',3);return;}
 const layers=Object.keys(TYPES).flatMap(k=>[k+'-point',k+'-line',k+'-fill']);const hit=map.queryRenderedFeatures(e.point,{layers})[0];if(hit){const p=hit.properties;const f=state[p.dfm_layer]?.find(x=>x.properties.dfm_id===p.dfm_id);if(f)selectFeature(f);}});
$('draw-cancel').onclick=stopDraw;
$('draw-finish').onclick=()=>{try{if(!drawing)return;const {kind,points}=drawing;const point=kind==='observations',line=TYPES[kind].geometry.includes('LineString');if(points.length<(point?1:line?2:3))throw Error('Place more points before finishing.');if(point&&points.length!==1)throw Error('An observation is one point. Cancel and place one point.');const geometry={type:point?'Point':line?'LineString':'Polygon',coordinates:point?points[0]:line?points:[[...points,points[0]]]};const f={type:'Feature',geometry,properties:{dfm_id:crypto.randomUUID(),name:'Drawn '+TYPES[kind].label}};const normalized=normalize(f,kind,'Drawn in DFM; unverified',t);state[kind].push(...normalized[kind]);if(kind==='boundary'||kind==='waterways')state.buffer=[];dirty=true;stopDraw();redraw();tell('Feature added. Record its source and save the project.');}catch(e){tell(e.message,true);}};
window.addEventListener('dfm-reset',stopDraw);
await import('./workspace.mjs');
const {setupGIS}=await import('./gis.mjs');setupGIS({map,upsert,dfm:window.dfm,tell,turf:t});
