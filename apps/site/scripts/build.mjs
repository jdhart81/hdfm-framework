// Build the static site into dist/: front matter + includes, no runtime JavaScript.
import {readFile, writeFile, mkdir, rm, cp, readdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const src = path.join(root, 'src'), dist = path.join(root, 'dist');
const config = JSON.parse(await readFile(path.join(root, 'site.config.json'), 'utf8'));
// The origin is set only in site.config.json: an https scheme and host, no path or trailing slash.
if (!/^https:\/\/[a-z0-9.-]+$/.test(config.origin)) throw new Error('site.config.json: origin must be https://host with no path');
const read = f => readFile(path.join(src, f), 'utf8');
const [head, foot] = await Promise.all([read('partials/head.html'), read('partials/foot.html')]);
const esc = s => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

async function pages(dir = src, rel = '') {
  const out = [];
  for (const e of await readdir(dir, {withFileTypes: true})) {
    if (e.name === 'partials') continue;
    if (e.isDirectory()) out.push(...await pages(path.join(dir, e.name), path.join(rel, e.name)));
    else if (e.name.endsWith('.html')) out.push(path.join(rel, e.name));
  }
  return out;
}

await rm(dist, {recursive: true, force: true});
await mkdir(path.join(dist, 'fonts'), {recursive: true});
const urls = [];
for (const page of await pages()) {
  const raw = await read(page);
  const m = raw.match(/^---\n([\s\S]*?)\n---\n/);
  if (!m) throw new Error(`${page}: missing front matter`);
  const meta = Object.fromEntries(m[1].split('\n').map(l => [l.slice(0, l.indexOf(':')).trim(), l.slice(l.indexOf(':') + 1).trim()]));
  let body = raw.slice(m[0].length);
  for (const [, name] of body.matchAll(/<!--#include ([\w.-]+)-->/g)) body = body.replace(`<!--#include ${name}-->`, await read('partials/' + name));
  const nav = k => (meta.nav === k ? ' aria-current="page"' : '');
  const html = (head + body + foot)
    .replaceAll('{{title}}', esc(meta.title)).replaceAll('{{description}}', esc(meta.description)).replaceAll('{{path}}', meta.path)
    .replaceAll('{{nav-check}}', nav('check')).replaceAll('{{nav-unbroken}}', nav('unbroken')).replaceAll('{{nav-format}}', nav('format'))
    .replaceAll('{{contact}}', esc(config.contact)).replaceAll('{{origin}}', config.origin);
  if (/\{\{[\w-]+\}\}/.test(html)) throw new Error(`${page}: unreplaced placeholder`);
  await mkdir(path.join(dist, path.dirname(page)), {recursive: true});
  await writeFile(path.join(dist, page), html);
  if (page !== '404.html') urls.push(config.origin + meta.path);
}
for (const f of ['styles.css', 'favicon.svg']) await cp(path.join(src, f), path.join(dist, f));
await writeFile(path.join(dist, 'robots.txt'), `User-agent: *\nAllow: /\nSitemap: ${config.origin}/sitemap.xml\n`);
const fonts = {
  '@fontsource-variable/archivo/files': ['archivo-latin-wdth-normal.woff2'],
  '@fontsource/source-serif-4/files': ['source-serif-4-latin-400-normal.woff2', 'source-serif-4-latin-400-italic.woff2', 'source-serif-4-latin-600-normal.woff2'],
};
for (const [dir, files] of Object.entries(fonts)) for (const f of files) await cp(path.join(root, 'node_modules', dir, f), path.join(dist, 'fonts', f));
// SIL Open Font License notices travel with the font files.
await cp(path.join(root, 'node_modules/@fontsource-variable/archivo/LICENSE'), path.join(dist, 'fonts', 'Archivo-OFL.txt'));
await cp(path.join(root, 'node_modules/@fontsource/source-serif-4/LICENSE'), path.join(dist, 'fonts', 'SourceSerif4-OFL.txt'));
await writeFile(path.join(dist, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls.sort().map(u => `  <url><loc>${u}</loc></url>`).join('\n')}\n</urlset>\n`);
console.log(`Built ${urls.length + 1} pages into dist/.`);
