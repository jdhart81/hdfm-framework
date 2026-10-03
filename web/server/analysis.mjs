import * as t from '@turf/turf';
import {checkConnectivitySync, toLandscapePackage} from '@viridis/dfm-core';
import {normalize,empty,fc,mergePolygons,riparianStudy} from '../public/map-core.mjs';
export const VERSION='dfm-geometric-0.3.0';
export function validate(data){
 if(!data || typeof data!=='object')throw Error('A map is required.');
 const features=Object.values(data).flat();
 if(!features.length)return empty();
 return normalize(fc(features),'project','User supplied',t);
}
export function assess(state,sources={}){
 const issues=[];for(const k of ['boundary','roads','waterways','forest'])if(!state[k]?.length)issues.push({level:'error',layer:k,message:`Load ${k} before assessing this landscape.`});
 for(const [k,v] of Object.entries(state)){if(!v.length||k==='buffer')continue;const s=sources[k];if(!s?.title||!s?.license||!s?.date)issues.push({level:'warning',layer:k,message:`Record the source, date and reuse terms for ${k}.`});}
 if(Object.values(state).flat().some(f=>f.properties?.dfm_evidence==='Illustrative example'))issues.push({level:'warning',layer:'all',message:'Contains fictional example geometry. Results are demonstrations.'});
 if(!state.waterbody?.length)issues.push({level:'warning',layer:'waterbody',message:'No open-water polygons supplied. Water surface cannot be fully excluded from forest estimates.'});
 return issues;
}
export function analyze(state,params,sources){
 const issues=assess(state,sources);if(issues.some(i=>i.level==='error'))throw Error(issues.filter(i=>i.level==='error').map(i=>i.message).join(' '));
 const width=Number(params.waterWidth),roadWidth=Number(params.roadWidth);
 if(!Number.isFinite(roadWidth)||roadWidth<1||roadWidth>100)throw Error('Road surface width must be between 1 and 100 m.');
 const all=Object.values(state).flat();let coords=0;t.coordEach(fc(all),()=>coords++);if(coords>6000||all.length>300)throw Error('Simplify this landscape to 300 features and 6,000 coordinates for hosted analysis.');
 const boundary=mergePolygons(state.boundary,t),zone=riparianStudy(state,width,t);
 const clip=p=>p?t.intersect(fc([boundary,p])):null;
 const area=p=>p?t.area(p)/10000:0;
 const union=fs=>mergePolygons(fs.filter(Boolean),t);
 const forest=clip(union(state.forest));
 const roads=clip(union(t.buffer(fc(state.roads),roadWidth/2,{units:'meters',steps:8}).features));
 const water=clip(union(state.waterbody||[]));
 const exclusion=union([roads,water]);
 const subtract=(a,b)=>a&&b?t.difference(fc([a,b])):a;
 const netForest=subtract(forest,exclusion);
 const links=clip(union(state.retention));
 const proposal=union([zone,links]);
 const retained=netForest?t.intersect(fc([netForest,proposal])):null;
 const units=clip(union(state.units));
 const overlap=units&&retained?t.intersect(fc([units,retained])):null;
 // Coincident crossings and bridge passability require separate field assessment.
 const crossings=t.lineIntersect(fc(state.roads),fc(state.waterways)).features;
 const inside=crossings.filter(f=>t.booleanPointInPolygon(f,boundary));
 const geo=fc([zone,retained&&{...retained,properties:{dfm_layer:'retention',name:'Forest within proposed retention',dfm_evidence:'Derived; unverified'}}].filter(Boolean));
 return {method:VERSION,parameters:{waterWidth:width,roadWidth},issues,metrics:{boundaryHa:area(boundary),bufferHa:area(zone),forestHa:area(forest),netForestHa:area(netForest),retainedForestHa:area(retained),roadSurfaceHa:area(roads),openWaterHa:area(water),managementOverlapHa:area(overlap),crossingCandidates:inside.length},geometry:geo,limitations:['Geometric screening only; no ecological viability, compliance, genetic outcome or optimality is established.','Road surface is modeled from a constant assumed full width; inspect actual widths and bridges.','Waterways are centerlines. Open-water exclusion requires the optional waterbody polygons.','Scenario differences are comparable only when their saved project revision is identical.','Areas are approximate geodesic hectares; no road surface is counted as retained forest.']};
}

// Corridor connectivity (DFM Build Spec phase 2). The same @viridis/dfm-core check that
// VergeCommon runs on treatment plans, fed from a saved workspace project.
const pick=(f,keys)=>({type:'Feature',geometry:f.geometry,properties:Object.fromEntries(keys.filter(k=>f.properties?.[k]!==undefined).map(k=>[k,f.properties[k]]))});
/** Build dfm-core inputs from a saved project. Retained habitat = proposed connections + riparian buffer (when boundary and waterways exist). */
export function corridorInput(state,params){
 const waterWidth=Number(params.waterWidth),roadWidth=Number(params.roadWidth),minWidthM=Number(params.minWidthM);
 const all=Object.values(state).flat();let coords=0;t.coordEach(fc(all),()=>coords++);if(coords>6000||all.length>300)throw Error('Simplify this landscape to 300 features and 6,000 coordinates for the hosted corridor check.');
 const retained=(state.retention||[]).map(f=>pick(f,['dfm_id','name']));
 if(state.boundary?.length&&state.waterways?.length&&Number.isFinite(waterWidth)){const zone=riparianStudy(state,waterWidth,t);retained.push({...zone,properties:{dfm_id:'riparian-study',name:zone.properties.name}});}
 return {
  boundary:(state.boundary||[]).map(f=>pick(f,['dfm_id','name'])),
  coreAreas:(state.cores||[]).map(f=>pick(f,['dfm_id','core_class','evidence_id','name'])),
  retained,
  roads:(state.roads||[]).map(f=>pick(f,['dfm_id','width_m','name'])),
  water:(state.waterbody||[]).map(f=>pick(f,['dfm_id','name'])),
  crossings:(state.crossings||[]).map(f=>pick(f,['dfm_id','passage','name'])),
  treatments:(state.treatments||[]).map(f=>pick(f,['dfm_id','period','intensity','corridor_permitted','reason','name'])),
  parcels:[],
  params:{minWidthM,minWidthSource:String(params.minWidthSource??'').trim().slice(0,300),roadWidthM:roadWidth},
 };
}
/** Run the corridor check and return it with a Landscape Package for VergeCommon or offline use. */
export function checkCorridors(state,params,{name,sources}={}){
 const input=corridorInput(state,params);
 const result=checkConnectivitySync(input);
 return {result,package:toLandscapePackage(input,{name,generator:'viridis-dfm-workspace',sources})};
}
