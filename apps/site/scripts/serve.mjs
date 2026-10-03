// Local preview of dist/ (development only; production uses the container in Dockerfile).
import http from 'node:http';
import {readFile, stat} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dist = path.join(path.dirname(path.dirname(fileURLToPath(import.meta.url))), 'dist');
const types = {'.html': 'text/html; charset=utf-8', '.css': 'text/css', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.txt': 'text/plain', '.xml': 'application/xml'};
const port = Number(process.env.PORT || 4300);
http.createServer(async (req, res) => {
  let p = path.normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname));
  let f = path.join(dist, p);
  if (!f.startsWith(dist)) { res.writeHead(400); return res.end(); }
  try { if ((await stat(f)).isDirectory()) f = path.join(f, 'index.html'); res.writeHead(200, {'content-type': types[path.extname(f)] || 'application/octet-stream'}); res.end(await readFile(f)); }
  catch { res.writeHead(404, {'content-type': types['.html']}); res.end(await readFile(path.join(dist, '404.html'))); }
}).listen(port, '127.0.0.1', () => console.log(`http://127.0.0.1:${port}/`));
