import {validate,assess,analyze} from './analysis.mjs';
const json=(x,status=200)=>Response.json(x,{status,headers:{'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'}});
const fail=(m,s=400)=>{throw Object.assign(Error(m),{status:s});};
const clean=(s,max=120)=>typeof s==='string'?s.trim().slice(0,max):'';
async function body(req){if(!req.headers.get('content-type')?.startsWith('application/json'))fail('Send JSON.',415);const raw=await req.text();if(new TextEncoder().encode(raw).length>2*1024*1024)fail('Project exceeds 2 MB. Simplify the data.',413);try{return JSON.parse(raw);}catch{fail('Invalid JSON.');}}
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
 const b=await body(req),name=clean(b.name);if(!name)fail('Give the project a name.');const data=validate(b.data),sources=validSources(b.sources),now=new Date().toISOString(),id=crypto.randomUUID();
 await db.prepare('INSERT INTO projects (id,owner,name,data,sources,revision,created,updated) VALUES (?,?,?,?,?,1,?,?)').bind(id,owner,name,JSON.stringify(data),JSON.stringify(sources),now,now).run();return json({id,name,revision:1,data,sources,created:now,updated:now},201);
 }
 const p=await db.prepare('SELECT * FROM projects WHERE id=? AND owner=?').bind(id,owner).first();if(!p)fail('Project not found.',404);
 if(parts.length===3&&method==='GET')return json({...p,owner:undefined,data:JSON.parse(p.data),sources:JSON.parse(p.sources)});
 if(parts.length===3&&method==='PUT'){
 const b=await body(req);if(b.revision!==p.revision)fail('This project changed elsewhere. Reopen it before saving; your current map is still available to export.',409);
 const name=clean(b.name);if(!name)fail('Give the project a name.');const data=validate(b.data),sources=validSources(b.sources),now=new Date().toISOString();
 const r=await db.prepare('UPDATE projects SET name=?,data=?,sources=?,revision=revision+1,updated=? WHERE id=? AND owner=? AND revision=?').bind(name,JSON.stringify(data),JSON.stringify(sources),now,id,owner,b.revision).run();if(r.meta.changes!==1)fail('Another save occurred. Reopen before saving.',409);return json({id,name,data,sources,revision:b.revision+1,updated:now});
 }
 if(parts[3]==='validation'&&method==='GET')return json({issues:assess(JSON.parse(p.data),JSON.parse(p.sources)),revision:p.revision});
 if(parts[3]==='scenarios'&&method==='GET'){const r=await db.prepare('SELECT id,name,revision,result,created FROM scenarios WHERE project=? AND owner=? ORDER BY created DESC').bind(id,owner).all();return json(r.results.map(s=>({...s,result:JSON.parse(s.result)})));}
 if(parts[3]==='scenarios'&&method==='POST'){
 const b=await body(req);if(b.revision!==p.revision)fail('Save or reopen the latest project before creating a scenario.',409);const name=clean(b.name);if(!name)fail('Give the scenario a name.');
 const data=JSON.parse(p.data),sources=JSON.parse(p.sources),result=analyze(data,b,sources);result.sources=sources;result.input=data;result.projectName=p.name;
 const id2=crypto.randomUUID(),now=new Date().toISOString();await db.prepare('INSERT INTO scenarios (id,project,owner,name,revision,result,created) VALUES (?,?,?,?,?,?,?)').bind(id2,id,owner,name,p.revision,JSON.stringify(result),now).run();return json({id:id2,name,revision:p.revision,result,created:now},201);
 }
 fail('Route or method not supported.',404);
 }catch(e){if(!e.status&&/SQL|database|D1|constraint/i.test(e.message))return json({error:'Storage could not complete the request. Your map has been kept; try again.'},503);return json({error:e.message},e.status||400);}
}
function validSources(s={}){const out={};for(const [k,v]of Object.entries(s)){if(!['boundary','forest','units','retention','waterways','roads','waterbody','observations'].includes(k))continue;out[k]={title:clean(v.title,300),url:clean(v.url,1000),date:clean(v.date,30),license:clean(v.license,300),notes:clean(v.notes,1500)};}return out;}
