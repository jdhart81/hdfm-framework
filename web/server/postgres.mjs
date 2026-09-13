import pg from 'pg';
// The same owner-scoped API can use PostgreSQL in a self-hosted deployment.
export async function postgres(connectionString){const pool=new pg.Pool({connectionString,max:5});await pool.query('SELECT 1');return {prepare(sql){let values=[];let n=0;const query=sql.replace(/\?/g,()=>'$'+(++n));const st={bind(...v){values=v;return st;},async all(){return {results:(await pool.query(query,values)).rows};},async first(){return (await st.all()).results[0]||null;},async run(){return {meta:{changes:(await pool.query(query,values)).rowCount}};}};return st;}};}
