import {readFileSync} from 'node:fs';import vm from 'node:vm';import assert from 'node:assert/strict';
import {example,fc,empty,normalize,riparianStudy,measurements} from '../public/map-core.mjs';
const context={};vm.createContext(context);vm.runInContext(readFileSync(new URL('../public/vendor/turf.min.js',import.meta.url),'utf8'),context);const t=context.turf;
const s=example(),m=measurements(s,t);assert(m.boundaryHa>770&&m.boundaryHa<775);assert(m.roadsKm>0&&m.waterKm>0);
const a=riparianStudy(s,50,t),b=riparianStudy(s,100,t);assert(t.area(b)>t.area(a));assert(t.area(a)<t.area(s.boundary[0]));assert(!t.difference(fc([a,s.boundary[0]])));
const duplicate=structuredClone(s);duplicate.waterways.push(structuredClone(s.waterways[0]));assert(Math.abs(t.area(riparianStudy(duplicate,50,t))-t.area(a))<.1);
s.buffer=[a];const restored=normalize(fc(Object.values(s).flat()),'project','export.geojson',t);assert.equal(restored.buffer[0].properties.distance_each_side_m,50);assert.equal(restored.waterways[0].properties.dfm_evidence,'Illustrative example');assert.equal(restored.roads.length,s.roads.length);
const imported=normalize(fc(s.roads),'roads','roads.geojson',t);assert.equal(imported.roads[0].properties.dfm_source,'roads.geojson');assert.equal(imported.roads[0].properties.dfm_evidence,'User supplied; unverified');
assert.throws(()=>normalize(fc(s.roads),'boundary','x',t));assert.throws(()=>normalize(fc([]),'roads','x',t));assert.throws(()=>normalize({type:'Feature',properties:{},geometry:{type:'LineString',coordinates:[[500000,20000],[600000,30000]]}},'roads','x',t));
assert.throws(()=>normalize(t.polygon([[[0,0],[1,1],[1,0],[0,1],[0,0]]]),'boundary','x',t));assert.throws(()=>riparianStudy(empty(),50,t));assert.throws(()=>riparianStudy(s,NaN,t));assert.throws(()=>riparianStudy(s,0,t));
const outside=structuredClone(s);outside.waterways=[t.lineString([[1,1],[1.01,1.01]])];assert.throws(()=>riparianStudy(outside,50,t));
console.log('PASS: known area, lengths, width sensitivity, boundary clipping, overlapping-buffer union, export/import provenance, incompatible geometry, invalid coordinates, self-crossing polygons, missing inputs and disjoint extent.');

assert.throws(()=>normalize(fc([s.roads[0],s.roads[0]]),'roads','duplicates',t));
