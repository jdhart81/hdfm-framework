#!/usr/bin/env node
import {readFile,writeFile} from 'node:fs/promises';
import {analyze,validate} from '../server/analysis.mjs';
import {normalize} from '../public/map-core.mjs';
import * as turf from '@turf/turf';
export async function analyzeFile(input,waterWidth,roadWidth){
 const raw=await readFile(input);if(raw.length>2*1024*1024)throw Error('Input exceeds 2 MB.');const doc=JSON.parse(raw);
 const record=doc.result||doc;
 const state=record.input?validate(record.input):normalize(doc,'project',input,turf);
 const sources=record.sources||doc.dfm_project?.sources||{};
 return {...analyze(state,{waterWidth,roadWidth},sources),input:state,sources,projectName:record.projectName||doc.dfm_project?.name||'Offline landscape assessment'};
}
const args=process.argv.slice(2);
try{
 if(args.length!==4)throw Error('Usage: npm run analyze -- INPUT.geojson OUTPUT.json WATER_BUFFER_M ROAD_FULL_WIDTH_M');
 const result=await analyzeFile(args[0],Number(args[2]),Number(args[3]));
 await writeFile(args[1],JSON.stringify(result,null,2)+'\n',{flag:'wx'});
 console.log(`Assessment written to ${args[1]}. ${result.issues.length} data warnings. Geometric screening only.`);
}catch(e){console.error(e.message);process.exitCode=1;}
