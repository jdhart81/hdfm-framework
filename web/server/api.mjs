import {validate,assess,analyze} from './analysis.mjs';
const json=(x,status=200)=>Response.json(x,{status,headers:{'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'}});
const fail=(m,s=400)=>{throw Object.assign(Error(m),{status:s});};
// User-input failures (plain Error from validation/geometry) become 400s; programming errors stay internal.
const input=fn=>{try{return fn();}catch(e){if(e?.constructor===Error&&!e.status&&!e.storage)e.status=400;throw e;}};
const clean=(s,max=120)=>typeof s==='string'?s.trim().slice(0,max):'';
async function body(req){if(!req.headers.get('content-type')?.startsWith('application/json'))fail('Send JSON.',415);const raw=await req.text();if(new TextEncoder().encode(raw).length>2*1024*1024)fail('Project exceeds 2 MB. Simplify the data.',413);let v;try{v=JSON.parse(raw);}catch{fail('Invalid JSON.');}if(!v||typeof v!=='object'||Array.isArray(v))fail('Send a JSON object.');return v;}
export async function api(req,env){
 try{
 const url=new URL(req.url),parts=url.pathname.split('/').filter(Boolean),method=req.method;
 const owner=req.headers.get('oai-authenticated-user-id');if(!owner)fail('Sign in to save and open projects.',401);
 if(!['GET','HEAD'].includes(method)){
 if(req.headers.get('origin')!==url.origin||req.headers.get('sec-fetch-site')==='cross-site')fail('Use this website to submit changes.',403);
 }
 if(!env.DB)fail('Project storage is not configured.',503);
 const db=env.DB;
 if(parts[1]==='account'&&method==='GET')return json({signedIn:true,service:'Viridis DFM private pilot',billing:'Not enabled',limits:{projectMB:2,analysisFeatures:300,analysisCoordinates:6000}});
 if(parts[1]!=='projects')fail('Not found.',404);
 const id=parts[2];
 if(!id&&method==='GET'){const r=await db.prepare('SELECT id,name,revision,created,updated FROM projects WHERE owner=? ORDER BY updated DESC').bind(owner).all();return json(r.results);}
 if(!id&&method==='POST'){
 const b=await body(req),name=clean(b.name);if(!name)fail('Give the project a name.');const data=input(()=>validate(b.data)),sources=input(()=>validSources(b.sources)),now=new Date().toISOString(),id=crypto.randomUUID();
 await db.prepare('INSERT INTO projects (id,owner,name,data,sources,revision,created,updated) VALUES (?,?,?,?,?,1,?,?)').bind(id,owner,name,JSON.stringify(data),JSON.stringify(sources),now,now).run();return json({id,name,revision:1,data,sources,created:now,updated:now},201);
 }
 const p=await db.prepare('SELECT * FROM projects WHERE id=? AND owner=?').bind(id,owner).first();if(!p)fail('Project not found.',404);
 if(parts.length===3&&method==='GET')return json({...p,owner:undefined,data:JSON.parse(p.data),sources:JSON.parse(p.sources)});
 if(parts.length===3&&method==='PUT'){
 const b=await body(req);if(b.revision!==p.revision)fail('This project changed elsewhere. Reopen it before saving; your current map is still available to export.',409);
 const name=clean(b.name);if(!name)fail('Give the project a name.');const data=input(()=>validate(b.data)),sources=input(()=>validSources(b.sources)),now=new Date().toISOString();
 const r=await db.prepare('UPDATE projects SET name=?,data=?,sources=?,revision=revision+1,updated=? WHERE id=? AND owner=? AND revision=?').bind(name,JSON.stringify(data),JSON.stringify(sources),now,id,owner,b.revision).run();if(r.meta.changes!==1)fail('Another save occurred. Reopen before saving.',409);return json({id,name,data,sources,revision:b.revision+1,updated:now});
 }
 if(parts.length===3&&method==='DELETE'){await db.prepare('DELETE FROM scenarios WHERE project=? AND owner=?').bind(id,owner).run();const r=await db.prepare('DELETE FROM projects WHERE id=? AND owner=?').bind(id,owner).run();if(r.meta.changes!==1)fail('Project not found.',404);return json({deleted:id});}
 if(parts[3]==='scenarios'&&parts[4]&&parts.length===5&&method==='DELETE'){const r=await db.prepare('DELETE FROM scenarios WHERE id=? AND project=? AND owner=?').bind(parts[4],id,owner).run();if(r.meta.changes!==1)fail('Scenario not found.',404);return json({deleted:parts[4]});}
 if(parts[3]==='validation'&&method==='GET')return json({issues:assess(JSON.parse(p.data),JSON.parse(p.sources)),revision:p.revision});
 if(parts[3]==='scenarios'&&method==='GET'){const r=await db.prepare('SELECT id,name,revision,result,created FROM scenarios WHERE project=? AND owner=? ORDER BY created DESC').bind(id,owner).all();return json(r.results.map(s=>({...s,result:JSON.parse(s.result)})));}
 if(parts[3]==='scenarios'&&method==='POST'){
 const b=await body(req);if(b.revision!==p.revision)fail('Save or reopen the latest project before creating a scenario.',409);const name=clean(b.name);if(!name)fail('Give the scenario a name.');
 const data=JSON.parse(p.data),sources=JSON.parse(p.sources),result=input(()=>analyze(data,b,sources));result.sources=sources;result.input=data;result.projectName=p.name;
 const id2=crypto.randomUUID(),now=new Date().toISOString();await db.prepare('INSERT INTO scenarios (id,project,owner,name,revision,result,created) VALUES (?,?,?,?,?,?,?)').bind(id2,id,owner,name,p.revision,JSON.stringify(result),now).run();return json({id:id2,name,revision:p.revision,result,created:now},201);
 }
 fail('Route or method not supported.',404);
 }catch(e){if(e.status)return json({error:e.message},e.status);if(e.storage||/SQL|database|D1|constraint/i.test(e.message||''))return json({error:'Storage could not complete the request. Your map has been kept; try again.'},503);console.error('DFM API internal error',e);return json({error:'The server could not complete the request.'},500);}
}
function validSources(s){const out={};if(!s||typeof s!=='object')return out;for(const [k,v]of Object.entries(s)){if(!v||typeof v!=='object')continue;if(!['boundary','forest','units','retention','waterways','roads','waterbody','observations'].includes(k))continue;out[k]={title:clean(v.title,300),url:clean(v.url,1000),date:clean(v.date,30),license:clean(v.license,300),notes:clean(v.notes,1500)};}return out;}
