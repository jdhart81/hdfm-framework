import initSqlJs from 'sql.js';import {readFile,writeFile,mkdir,rename} from 'node:fs/promises';
export async function sqlite(file=null){const SQL=await initSqlJs();let bytes;if(file)try{bytes=await readFile(file);}catch(e){if(e.code!=='ENOENT')throw e;}const db=new SQL.Database(bytes);let queue=Promise.resolve();
 const persist=()=>{if(!file)return Promise.resolve();queue=queue.then(async()=>{await writeFile(file+'.tmp',db.export());await rename(file+'.tmp',file);});return queue;};
 return {async migrate(sql){db.exec(sql);await persist();},prepare(sql){let values=[];const st={bind(...v){values=v;return st;},async all(){const q=db.prepare(sql);try{q.bind(values);const results=[];while(q.step())results.push(q.getAsObject());return {results};}finally{q.free();}},async first(){return (await st.all()).results[0]||null;},async run(){db.run(sql,values);const changes=db.getRowsModified();await persist();return {meta:{changes}};}};return st;}};
}
