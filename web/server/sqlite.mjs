import initSqlJs from 'sql.js';import {readFile,writeFile,rename} from 'node:fs/promises';
// Single-process SQLite (sql.js) persisted by atomic file replacement.
// Invariants: a write is acknowledged only after it reaches disk; a failed disk
// write rolls the in-memory database back to the last persisted state and fails
// every write made since then (none is reported as saved); one failure does not
// block later writes.
export async function sqlite(file=null){const SQL=await initSqlJs();let bytes;if(file)try{bytes=await readFile(file);}catch(e){if(e.code!=='ENOENT')throw e;}let db=new SQL.Database(bytes);let queue=Promise.resolve(),generation=0,lastGood=file?db.export():null;
 const storageError=cause=>Object.assign(Error('Storage write failed; the change was not saved.'),{storage:true,cause});
 const persist=g=>{if(!file)return Promise.resolve();const write=queue.then(async()=>{if(g!==generation)throw storageError(Error('An earlier write failed.'));try{const out=db.export();await writeFile(file+'.tmp',out);await rename(file+'.tmp',file);lastGood=out;}catch(e){generation++;db.close();db=new SQL.Database(lastGood);throw storageError(e);}});queue=write.catch(()=>{});return write;};
 return {async migrate(sql){const g=generation;db.exec(sql);await persist(g);},prepare(sql){let values=[];const st={bind(...v){values=v;return st;},async all(){const q=db.prepare(sql);try{q.bind(values);const results=[];while(q.step())results.push(q.getAsObject());return {results};}finally{q.free();}},async first(){return (await st.all()).results[0]||null;},async run(){const g=generation;db.run(sql,values);const changes=db.getRowsModified();await persist(g);return {meta:{changes}};}};return st;}};
}
