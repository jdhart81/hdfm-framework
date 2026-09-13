import {TYPES,empty,fc,normalize,riparianStudy,measurements,example} from './map-core.mjs';
const $=id=>document.getElementById(id),t=globalThis.turf;
const tell=(message,error=false)=>{$('message').textContent=message;$('message').classList.toggle('error',error);};
if(!globalThis.L||!t){tell('The mapping tools could not load. Please reload this page.',true);throw Error('Mapping libraries unavailable');}
const map=L.map('map',{preferCanvas:true,worldCopyJump:false}).setView([0.0125,.0125],15);
map.attributionControl.addAttribution('Example geometry: DFM');L.control.scale({imperial:false}).addTo(map);
const base=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'});
base.on('tileerror',()=>tell('Background tiles are unavailable. Your loaded layers still work.',true));
let state=example(),dirty=false,rendered={},selected=null,scenarioOverlay=null;
function text(tag,value,parent){const e=document.createElement(tag);e.textContent=value;parent.append(e);return e;}
function selectFeature(f,layer){selected=f;const host=$('feature-detail');host.replaceChildren();text('h2',String(f.properties.name||f.properties.dfm_id),host);const remove=document.createElement('button');remove.textContent='Remove from working map';remove.onclick=()=>{if(!confirm('Remove this feature from the working map? Saved data changes only when you save the project.'))return;const k=f.properties.dfm_layer;state[k]=state[k].filter(x=>x.properties.dfm_id!==f.properties.dfm_id);if(k==='boundary'||k==='waterways')state.buffer=[];dirty=true;redraw();};host.append(remove);const dl=document.createElement('dl');host.append(dl);for(const [name,value]of Object.entries(f.properties)){text('dt',name.replace(/^dfm_/,'').replaceAll('_',' '),dl);text('dd',typeof value==='object'?JSON.stringify(value):String(value),dl);}if(layer?.getBounds)map.fitBounds(layer.getBounds(),{maxZoom:17,padding:[35,35]});}
function fit(){const layers=Object.values(rendered).filter(g=>map.hasLayer(g));const group=L.featureGroup(layers);if(group.getLayers().length&&group.getBounds().isValid())map.fitBounds(group.getBounds(),{padding:[30,30],maxZoom:17});else tell('No visible features. Load data or enable a layer.');}
function redraw(){
 if(scenarioOverlay){map.removeLayer(scenarioOverlay);scenarioOverlay=null;}
 const visibility=Object.fromEntries(Object.keys(TYPES).map(k=>[k,$(`show-${k}`)?.checked??true]));Object.values(rendered).forEach(g=>map.removeLayer(g));rendered={};$('layers').replaceChildren();$('feature-list').replaceChildren();
 for(const [k,meta]of Object.entries(TYPES)){
  const row=document.createElement('div');row.className='layer-row';const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.id=`show-${k}`;input.checked=visibility[k];label.append(input);const swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=meta.color;label.append(swatch);text('span',meta.label,label);row.append(label);text('span',String(state[k].length),row).className='layer-count';$('layers').append(row);
  const style={color:meta.color,weight:k==='roads'?4:k==='waterways'?3:2,fillColor:meta.color,fillOpacity:k==='boundary'?0:k==='buffer'?.3:.18,dashArray:k==='boundary'?'8 6':k==='retention'?'5 4':undefined};
  const group=L.geoJSON(fc(state[k]),{style,onEachFeature(f,layer){const tooltip=document.createElement('span');tooltip.textContent=String(f.properties.name||f.properties.dfm_id);layer.bindTooltip(tooltip);layer.on('click',()=>selectFeature(f));const btn=document.createElement('button');btn.textContent=`${meta.label} · ${f.properties.name||f.properties.dfm_id}`;btn.onclick=()=>selectFeature(f,layer);$('feature-list').append(btn);}});rendered[k]=group;if(visibility[k])group.addTo(map);input.onchange=()=>input.checked?group.addTo(map):map.removeLayer(group);
 }
 const features=Object.values(state).flat(),examples=features.filter(f=>f.properties.dfm_evidence==='Illustrative example').length;
 $('dataset-status').textContent=!features.length?'Empty map · load your landscape':examples===features.length?'Illustrative example · not a surveyed site':examples?'Mixed example and imported data · not a site assessment':'Imported map · sources not independently verified';
 $('feature-total').textContent=`${features.length} features · ${Object.values(state).filter(f=>f.length).length} populated layers`;$('list-count').textContent=`(${features.length})`;
 $('export').disabled=!features.length;$('clear-buffer').disabled=!state.buffer.length;$('buffer').disabled=!state.waterways.length||!state.boundary.length;
 try{const m=measurements(state,t);const fmt=(v,u)=>v===null?'—':`${v.toLocaleString(undefined,{maximumFractionDigits:2})} ${u}`;$('area').textContent=fmt(m.boundaryHa,'ha');$('buffer-area').textContent=fmt(m.bufferHa,'ha');$('roads-length').textContent=fmt(m.roadsKm,'km');$('water-length').textContent=fmt(m.waterKm,'km');}catch(e){for(const id of ['area','buffer-area','roads-length','water-length'])$(id).textContent='Unavailable';tell(`Could not calculate measurements: ${e.message}`,true);}
 $('feature-detail').replaceChildren();text('h2','Feature details',$('feature-detail'));text('p','Select a feature on the map or in the feature list to inspect its source.',$('feature-detail'));
}
function replace(next,message){state=next;dirty=false;redraw();fit();tell(message);window.dispatchEvent(new Event('dfm-reset'));}
$('new-project').onclick=()=>{if(dirty&&!confirm('Clear this map? Export first to keep your changes.'))return;replace(empty(),'New map ready. Load a planning boundary, roads, and waterways.');};
$('example').onclick=()=>{if(dirty&&!confirm('Replace this map with the example? Export first to keep your changes.'))return;replace(example(),'Illustrative data loaded. These features do not represent a real site.');};
$('fit').onclick=fit;$('basemap').onchange=()=>{$('basemap').checked?base.addTo(map):map.removeLayer(base);};
$('file').onchange=async e=>{
 const file=e.target.files[0];if(!file)return;const kind=$('layer-type').value;
 try{if(file.size>5*1024*1024)throw Error('Choose a file smaller than 5 MB.');const parsed=JSON.parse(await file.text());const incoming=normalize(parsed,kind,file.name,t);let next=structuredClone(state);
  if(kind==='project'){if(dirty&&!confirm('Replace the current map with this exported map?'))return;next=incoming;}
  else{if(next[kind].length&&!confirm(`Replace the current ${TYPES[kind].label.toLowerCase()} layer?`))return;next[kind]=incoming[kind];if(kind==='waterways'||kind==='boundary')next.buffer=[];}
  const all=Object.values(next).flat();if(all.length>3000)throw Error('The map supports at most 3,000 features.');
  state=next;dirty=true;if(kind==='project')window.dispatchEvent(new CustomEvent('dfm-import',{detail:parsed.dfm_project||{}}));redraw();fit();tell(`${file.name} loaded. ${kind==='waterways'||kind==='boundary'?'The previous buffer study was cleared. ':''}Export to keep this map.`);
 }catch(error){tell(`Import failed: ${error.message} Your existing map was kept.`,true);}finally{e.target.value='';}
};
$('buffer').onclick=()=>{try{state.buffer=[riparianStudy(state,Number($('buffer-width').value),t)];dirty=true;redraw();$('show-buffer').checked=true;rendered.buffer.addTo(map);tell('Buffer study created and clipped to the planning boundary. This is a proposed zone, not measured forest.');}catch(e){tell(e.message,true);}};
$('buffer-width').oninput=()=>{if(state.buffer.length)tell('Distance changed. Choose Create buffer to replace the existing study; its current settings remain in Feature details.');};
$('clear-buffer').onclick=()=>{state.buffer=[];dirty=true;redraw();tell('Buffer study removed.');};
$('export').onclick=()=>{const data=fc(Object.values(state).flat());data.dfm_project={...window.dfmProjectMeta?.(),version:2,exported_at:new Date().toISOString(),scope:'Mapping workspace; not an optimized or validated forest plan',coordinates:'WGS84 longitude/latitude',measurement_method:'Turf 7.2.0 approximate geodesic measurements'};const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/geo+json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='dfm-landscape.geojson';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);tell('Map exported with layer and source labels. Use Restore exported map to reopen it.');};
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
redraw();fit();tell('Explore the example, or choose New map and load your own GeoJSON layers.');

// A small public interface keeps saved-project controls independent of the map renderer.
window.dfm={getState:()=>structuredClone(state),isDirty:()=>dirty,markDirty:()=>{dirty=true;},saved:()=>{dirty=false;},setState(next){state=next;dirty=false;redraw();fit();},showResult(result){if(scenarioOverlay)map.removeLayer(scenarioOverlay);scenarioOverlay=L.geoJSON(result.geometry,{style:f=>({color:f.properties.dfm_layer==='buffer'?'#168eab':'#cf9bff',weight:3,fillOpacity:.25,dashArray:'5 4'})}).addTo(map);if(scenarioOverlay.getBounds().isValid())map.fitBounds(scenarioOverlay.getBounds(),{padding:[35,35]});tell('Saved scenario overlay shown: blue buffer zone and purple retained forest. Editing the working map clears this overlay.');},tell};
let drawing=null,sketch=null;
function stopDraw(){drawing=null;if(sketch)map.removeLayer(sketch);sketch=null;map.doubleClickZoom.enable();$('draw-finish').disabled=$('draw-cancel').disabled=true;$('draw-start').disabled=false;}
$('draw-start').onclick=()=>{const kind=$('layer-type').value;if(!TYPES[kind]||kind==='buffer')return tell('Choose an input layer to draw.',true);drawing={kind,points:[]};$('draw-finish').disabled=$('draw-cancel').disabled=false;$('draw-start').disabled=true;map.doubleClickZoom.disable();tell('Click the map to place points, then choose Finish.');};
map.on('click',e=>{if(!drawing)return;drawing.points.push([e.latlng.lng,e.latlng.lat]);if(sketch)map.removeLayer(sketch);sketch=L.polyline(drawing.points.map(([x,y])=>[y,x]),{color:'#e9df64',dashArray:'4 4'}).addTo(map);});
$('draw-cancel').onclick=stopDraw;
$('draw-finish').onclick=()=>{try{if(!drawing)return;const {kind,points}=drawing;const line=TYPES[kind].geometry.includes('LineString');if(points.length<(line?2:3))throw Error('Place more points before finishing.');const geometry={type:line?'LineString':'Polygon',coordinates:line?points:[[...points,points[0]].flat(0)]};const f={type:'Feature',geometry,properties:{dfm_id:crypto.randomUUID(),name:'Drawn '+TYPES[kind].label}};const normalized=normalize(f,kind,'Drawn in DFM; unverified',t);state[kind].push(...normalized[kind]);if(kind==='boundary'||kind==='waterways')state.buffer=[];dirty=true;stopDraw();redraw();tell('Feature added. Record its source and save the project.');}catch(e){tell(e.message,true);}};
await import('./workspace.mjs');
