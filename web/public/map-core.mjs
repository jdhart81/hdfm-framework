export const TYPES={observations:{label:"Field observations",color:"#d25449",geometry:["Point"]},waterbody:{label:'Open water surface',color:'#336da0',geometry:['Polygon','MultiPolygon']},boundary:{label:'Planning boundary',color:'#485f51',geometry:['Polygon','MultiPolygon']},forest:{label:'Existing forest',color:'#4c8148',geometry:['Polygon','MultiPolygon']},units:{label:'Management units',color:'#b59448',geometry:['Polygon','MultiPolygon']},retention:{label:'Proposed forest connections',color:'#a163a9',geometry:['Polygon','MultiPolygon']},buffer:{label:'Riparian buffer study',color:'#258c83',geometry:['Polygon','MultiPolygon']},waterways:{label:'Waterways',color:'#1687be',geometry:['LineString','MultiLineString']},roads:{label:'Existing roads',color:'#bd6639',geometry:['LineString','MultiLineString']}};
export const empty=()=>Object.fromEntries(Object.keys(TYPES).map(k=>[k,[]]));
export const fc=features=>({type:'FeatureCollection',features});
export function normalize(data,kind,source,turf){
 if(data.crs&&!String(data.crs?.properties?.name).match(/(CRS84|4326)$/))throw Error('Reproject this file to WGS84 longitude/latitude before importing.');
 const features=data.type==='FeatureCollection'?data.features:data.type==='Feature'?[data]:null;
 if(!Array.isArray(features)||!features.length)throw Error('Choose a non-empty GeoJSON Feature or FeatureCollection.');
 if(features.length>2000)throw Error('Use no more than 2,000 features per import.');
 const result=empty();const ids=Object.fromEntries(Object.keys(TYPES).map(k=>[k,new Set()]));let vertices=0;
 for(let i=0;i<features.length;i++){
  const f=features[i];const k=kind==='project'?f?.properties?.dfm_layer:kind;
  if(!TYPES[k]||!TYPES[k].geometry.includes(f?.geometry?.type)) {
   const valid = Object.keys(TYPES).join(', ');
   if (!TYPES[k]) throw Error(`Feature ${i+1} has invalid dfm_layer "${k}". Expected one of: ${valid}.`);
   throw Error(`Feature ${i+1} (${k}): expected ${TYPES[k].geometry.join(' or ')}.`);
  }
  if(f.type!=='Feature'||!f.geometry.coordinates?.length)throw Error(`Feature ${i+1} has no geometry.`);
  const coord=c=>{if(!Array.isArray(c)||!c.length)throw Error('Empty coordinates are not supported.');if(typeof c[0]==='number'){vertices++;if(c.length<2||c.some(x=>!Number.isFinite(x))||Math.abs(c[0])>180||Math.abs(c[1])>85)throw Error('Use finite longitude/latitude coordinates between ±180° and ±85°.');}else c.forEach(coord);};coord(f.geometry.coordinates);
  if(vertices>20000)throw Error('Simplify the file to at most 20,000 coordinates.');
  if(!turf.booleanValid(f))throw Error(`Feature ${i+1} has invalid geometry. Check closed polygon rings and line coordinates.`);
  if(f.geometry.type.includes('Polygon')&&turf.kinks(f).features.length)throw Error(`Feature ${i+1} has self-crossing polygon rings.`);
  const b=turf.bbox(f);if(b[2]-b[0]>5||b[3]-b[1]>5)throw Error('Use a local landscape extent under 5°; antimeridian geometry is not supported.');
  const p=f.properties&&typeof f.properties==='object'&&!Array.isArray(f.properties)?f.properties:{};
  const id=String(p.dfm_id??f.id??`${k}-${i+1}`);if(ids[k].has(id))throw Error(`Duplicate feature ID ${id} in ${TYPES[k].label}.`);ids[k].add(id);
  result[k].push({type:'Feature',geometry:structuredClone(f.geometry),properties:{...p,dfm_layer:k,dfm_id:String(p.dfm_id??f.id??`${k}-${i+1}`),dfm_source:kind==='project'?String(p.dfm_source||source):source,dfm_evidence:kind==='project'?String(p.dfm_evidence||'User supplied; unverified'):'User supplied; unverified'}});
 }
 const b=turf.bbox(fc(Object.values(result).flat()));if(b[2]-b[0]>5||b[3]-b[1]>5)throw Error('Import one local landscape under 5° across.');
 return result;
}
export function mergePolygons(features,turf){if(!features.length)return null;return features.length===1?structuredClone(features[0]):turf.union(fc(features));}
export function riparianStudy(state,width,turf){
 if(!Number.isFinite(width)||width<5||width>500)throw Error('Choose a distance from 5 to 500 meters.');
 if(!state.boundary.length||!state.waterways.length)throw Error('Load a planning boundary and waterways first.');
 if(state.waterways.length>100||state.boundary.length>100)throw Error('Buffer studies support up to 100 waterway and boundary features.');
 const buffered=turf.buffer(fc(state.waterways),width,{units:'meters',steps:8});
 if(!buffered?.features?.length)throw Error('Could not construct a buffer from these waterways.');
 const merged=mergePolygons(buffered.features,turf),boundary=mergePolygons(state.boundary,turf);
 const clipped=turf.intersect(fc([merged,boundary]));if(!clipped)throw Error('These waterways do not produce a buffer inside the boundary.');
 clipped.properties={dfm_id:'riparian-study',name:'Riparian buffer study',dfm_layer:'buffer',dfm_source:'Waterway buffers unioned and clipped to planning boundary',dfm_evidence:state.waterways.some(f=>f.properties.dfm_evidence==='Illustrative example')?'Illustrative example':'Derived geometry; not field verified',distance_each_side_m:width,source_waterway_ids:state.waterways.map(f=>f.properties.dfm_id),source_boundary_ids:state.boundary.map(f=>f.properties.dfm_id),method:'Turf 7.2.0 buffer/union/intersect; approximate'};
 return clipped;
}
export function measurements(state,turf){const a=k=>{const p=mergePolygons(state[k],turf);return p?turf.area(p)/10000:null;};return {boundaryHa:a('boundary'),bufferHa:a('buffer'),roadsKm:state.roads.length?turf.length(fc(state.roads),{units:'kilometers'}):null,waterKm:state.waterways.length?turf.length(fc(state.waterways),{units:'kilometers'}):null};}
export function example(){const s=empty();const add=(k,name,type,coordinates)=>s[k].push({type:'Feature',geometry:{type,coordinates},properties:{dfm_id:`${k}-${s[k].length+1}`,name,dfm_layer:k,dfm_source:'DFM illustrative dataset — fictional geometry at equatorial coordinates',dfm_evidence:'Illustrative example'}});
 add('boundary','Example planning boundary','Polygon',[[[0,0],[.025,0],[.025,.025],[0,.025],[0,0]]]);
 add('waterways','Example main waterway','LineString',[[.007,.025],[.01,.02],[.014,.016],[.013,.01],[.017,.005],[.02,0]]);
 add('waterways','Example tributary','LineString',[[.024,.024],[.02,.02],[.014,.016]]);
 add('roads','Example existing access','LineString',[[0,.004],[.005,.008],[.008,.015],[.009,.022],[.014,.025]]);
 add('roads','Example access branch','LineString',[[.005,.008],[.011,.005],[.016,.004],[.023,.006]]);
 add('forest','Example existing forest','Polygon',[[[.017,.014],[.024,.013],[.024,.021],[.021,.022],[.017,.014]]]);
 add('forest','Example upland forest','Polygon',[[[.001,.016],[.006,.015],[.007,.022],[.002,.023],[.001,.016]]]);
 add('units','Example management unit','Polygon',[[[.001,.001],[.007,.001],[.007,.004],[.003,.006],[.001,.001]]]);return s;
}
