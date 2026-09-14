import {PROVIDERS,providerSpec,parseCoordinates,gpsFix,ecadExchange} from './gis-core.mjs';
export function setupGIS({map,upsert,dfm,tell,turf}){
 const $=id=>document.getElementById(id);let watch=null,epoch=0,fix=null,following=false,backgroundKey='none';
 $('imagery-date').value=new Date(Date.now()-86400000).toISOString().slice(0,10);$('imagery-date').max=new Date().toISOString().slice(0,10);$('imagery-date').min='2000-02-24';
 function background(){try{const key=$('background').value,p=providerSpec(key,$('imagery-date').value);if(map.getLayer('imagery'))map.removeLayer('imagery');if(map.getSource('imagery'))map.removeSource('imagery');backgroundKey=key;
 if(p.url){map.addSource('imagery',{type:'raster',tiles:[p.url],tileSize:256,maxzoom:p.maxzoom,attribution:p.attribution});map.addLayer({id:'imagery',type:'raster',source:'imagery',paint:{'raster-opacity':1}},'observations-fill');}
 $('imagery-date').disabled=key!=='satellite';$('imagery-info').textContent=p.description+(key==='satellite'?` Requested UTC date: ${$('imagery-date').value}. Availability is not guaranteed.`:'')+' Imagery is background context; it does not populate analytical layers.';
 }catch(e){tell(e.message,true);}}
 $('background').onchange=background;$('imagery-date').onchange=background;
 map.on('error',e=>{if(e.sourceId==='imagery')$('imagery-info').textContent=PROVIDERS[backgroundKey].label+': imagery could not load here. Check your connection, date and coverage. Project layers remain available.';});
 function coordinateReadout(){const c=map.getCenter();$('coordinates').textContent=`Map center: ${c.lat.toFixed(6)}, ${c.lng.toFixed(6)} · WGS84 · zoom ${map.getZoom().toFixed(1)}`;}
 map.on('moveend',coordinateReadout);coordinateReadout();
 $('go-location').onclick=()=>{try{map.flyTo({center:parseCoordinates($('map-location').value),zoom:15});}catch(e){tell(e.message,true);}};
 $('map-location').onkeydown=e=>{if(e.key==='Enter')$('go-location').click();};
 function clearGraphics(){map.getSource('gps-accuracy')?.setData({type:'FeatureCollection',features:[]});map.getSource('gps-position')?.setData({type:'FeatureCollection',features:[]});}
 function stop(message='GPS stopped.'){epoch++;if(watch!==null)navigator.geolocation.clearWatch(watch);watch=null;following=false;fix=null;clearGraphics();$('gps-stop').disabled=true;$('gps-save').disabled=true;$('gps-info').textContent=message;}
 function start(follow){stop();if(!navigator.geolocation){$('gps-info').textContent='This browser does not provide location.';return;}following=follow;const generation=epoch;$('gps-stop').disabled=false;$('gps-info').textContent='Waiting for permission and a fresh location…';
 const success=position=>{if(generation!==epoch)return;try{fix=gpsFix(position);const {longitude:lon,latitude:lat,accuracy}=fix;upsert('gps-position',{type:'FeatureCollection',features:[{type:'Feature',properties:{},geometry:{type:'Point',coordinates:[lon,lat]}}]},'#1659d6',2);
 upsert('gps-accuracy',{type:'FeatureCollection',features:accuracy<100000?[turf.circle([lon,lat],Math.max(accuracy,.1)/1000,{units:'kilometers',steps:48})]:[]},'#1659d6',1,.12);
 $('gps-save').disabled=false;$('gps-info').textContent=`${following?'Following GPS':'Location'}: ${lat.toFixed(6)}, ${lon.toFixed(6)} · accuracy ±${Math.round(accuracy)} m${accuracy>100?' · low accuracy':''} · ${new Date(fix.timestamp).toLocaleTimeString()} · not saved.`;if(following||!follow)map.easeTo({center:[lon,lat],zoom:Math.max(map.getZoom(),15)});
 }catch(e){fix=null;$('gps-save').disabled=true;clearGraphics();$('gps-info').textContent=e.message;}};
 const failure=e=>{if(generation!==epoch)return;stop(e.code===1?'Location permission was denied. Use coordinates or enable location in browser settings.':e.code===3?'Location timed out. Try again outdoors or enter coordinates.':'Location is unavailable. Try again or enter coordinates.');};
 const options={enableHighAccuracy:true,maximumAge:0,timeout:15000};if(follow)watch=navigator.geolocation.watchPosition(success,failure,options);else navigator.geolocation.getCurrentPosition(success,failure,options);
 }
 $('gps-locate').onclick=()=>start(false);$('gps-follow').onclick=()=>start(true);$('gps-stop').onclick=()=>stop();
 map.on('dragstart',()=>{if(following){following=false;$('gps-info').textContent+=' Camera following paused while you explore; GPS still updates. Choose Follow GPS to recenter.';}});
 const staleTimer=setInterval(()=>{if(fix&&Date.now()-fix.timestamp>30000){fix=null;$('gps-save').disabled=true;clearGraphics();$('gps-info').textContent='Last location is stale. Waiting for a fresh fix, or choose Locate me.';}},5000);
 window.addEventListener('pagehide',()=>{stop();clearInterval(staleTimer);});
 $('gps-save').onclick=()=>{if(!fix||Date.now()-fix.timestamp>30000){tell('Request a fresh GPS fix before saving.',true);return;}const f={type:'Feature',geometry:{type:'Point',coordinates:[fix.longitude,fix.latitude]},properties:{dfm_id:crypto.randomUUID(),dfm_layer:'observations',name:'GPS field point',dfm_source:'Browser device location',dfm_evidence:'Device observation; not a professional survey',horizontal_accuracy_m:fix.accuracy,observed_at:new Date(fix.timestamp).toISOString(),measurement_method:'device_gps'}};dfm.addObservation(f);tell('GPS point added to the working map with accuracy and time. Save project to store it.');};
 $('ecad-export').onclick=()=>{try{const data=ecadExchange(dfm.getState(),window.dfmProjectMeta?.()||{});if(!data.features.length)throw Error('Add or load features first.');const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/geo+json'})),a=document.createElement('a');a.href=url;a.download='dfm-ecad-evidence.geojson';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);tell('ECAD evidence GeoJSON exported. Import it using ECAD’s Import evidence GeoJSON tool. Original identity and source notes are included; this is not automatic synchronization.');}catch(e){tell(e.message,true);}};
 background();
}
