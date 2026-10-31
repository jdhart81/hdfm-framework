// Build the static site into dist/: front matter + includes, no runtime JavaScript.
import {readFile, writeFile, mkdir, rm, cp, readdir} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const src = path.join(root, 'src'), dist = path.join(root, 'dist');
const config = JSON.parse(await readFile(path.join(root, 'site.config.json'), 'utf8'));
// The origin is set only in site.config.json: an https scheme and host, no path or trailing slash.
if (!/^https:\/\/[a-z0-9.-]+$/.test(config.origin)) throw new Error('site.config.json: origin must be https://host with no path');
if (!/^https:\/\/[a-z0-9.-]+\/$/.test(config.vergecommon ?? '')) throw new Error('site.config.json: vergecommon must be https://host/');
// VergeCommon's woodland page says whether woodland projects are open, so this site never has to;
// its #first-woodlot section is the Unbroken Woods pilot contact.
const vergecommonWoodland = new URL('woodland/', config.vergecommon).href;
const read = f => readFile(path.join(src, f), 'utf8');
const [head, foot] = await Promise.all([read('partials/head.html'), read('partials/foot.html')]);
const esc = s => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

/**
 * Expand <!--#include name.html key=value ...--> directives, recursively (depth <= 3).
 * Arguments replace {{arg:key}} inside the included partial before its own includes
 * expand, so one partial can be included several times with unique IDs.
 */
async function expand(text, depth = 0) {
  if (depth > 3) throw new Error('Includes nest more than 3 deep.');
  let out = '', last = 0;
  for (const m of text.matchAll(/<!--#include ([\w.-]+)((?: [\w-]+=[\w-]+)*)-->/g)) {
    let part = await read('partials/' + m[1]);
    for (const [, k, v] of m[2].matchAll(/ ([\w-]+)=([\w-]+)/g)) part = part.replaceAll(`{{arg:${k}}}`, v);
    if (part.includes('{{arg:')) throw new Error(`${m[1]}: an {{arg:...}} placeholder has no value in this include.`);
    out += text.slice(last, m.index) + await expand(part, depth + 1);
    last = m.index + m[0].length;
  }
  return out + text.slice(last);
}

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
  const body = await expand(raw.slice(m[0].length));
  const nav = k => (meta.nav === k ? ' aria-current="page"' : '');
  const html = (head + body + foot)
    .replaceAll('{{title}}', esc(meta.title)).replaceAll('{{description}}', esc(meta.description)).replaceAll('{{path}}', meta.path)
    .replaceAll('{{nav-check}}', nav('check')).replaceAll('{{nav-unbroken}}', nav('unbroken')).replaceAll('{{nav-format}}', nav('format'))
    .replaceAll('{{vergecommon}}', esc(config.vergecommon))
    .replaceAll('{{vergecommon-woodland}}', esc(vergecommonWoodland)).replaceAll('{{vergecommon-first-woodlot}}', esc(vergecommonWoodland + '#first-woodlot'))
    // The 404 page is served for every missing address: never indexed, no canonical.
    .replaceAll('{{canonical}}', page === '404.html' ? '<meta name="robots" content="noindex">' : `<link rel="canonical" href="${config.origin}${meta.path}">`);
  if (/\{\{[\w-]+\}\}/.test(html)) throw new Error(`${page}: unreplaced placeholder`);
  await mkdir(path.join(dist, path.dirname(page)), {recursive: true});
  await writeFile(path.join(dist, page), html);
  if (page !== '404.html') urls.push(config.origin + meta.path);
}
for (const f of ['styles.css', 'favicon.svg']) await cp(path.join(src, f), path.join(dist, f));
// llms.txt: where a language model or agent should look, in the llms.txt layout (one H1, a summary
// quote, then sections of links). The origin and VergeCommon address come from site.config.json.
const repo = 'https://github.com/jdhart81/hdfm-framework';
await writeFile(path.join(dist, 'llms.txt'), [
  '# Dendritic Forest Management',
  '',
  '> Open-source (MIT) landscape planning that keeps working woods connected. The dfm-core engine checks a treatment or harvest plan against a landscape\'s retained corridors and reports which links between core areas it would break, and drafts an old-growth spine along streams, valleys and ridges that never counts a plantation as old growth. It checks structural connectivity only: it does not establish species movement, genetic exchange, regulatory compliance or old-growth condition.',
  '',
  'Agents: help a person understand the method, prepare a Landscape Package, or reproduce a check. Say what the check establishes and nothing more. dfm-core is not on npm yet; use it from the repository. Plans for real land are run by the person or their forester, and woodland co-ops run them through VergeCommon.',
  '',
  '## Method and data',
  '',
  `- [The method](${config.origin}/): what the corridor check and the old-growth spine do, and what they do not establish`,
  `- [Data format](${config.origin}/data-format/): the Landscape Package v1, one JSON file of WGS84 GeoJSON layers with the minimum corridor width and a source record per layer`,
  `- [dfm-core engine](${repo}/blob/main/packages/dfm-core/README.md): checkConnectivity, the old-growth spine, any biome, native planting rules and the package schema`,
  `- [Research status](${repo}/blob/main/docs/RESEARCH_STATUS.md): which claims are established and which are open questions`,
  '',
  '## Use it',
  '',
  `- [Quickstart](${repo}/blob/main/QUICKSTART.md): exact commands and expected output for a landscape assessment, a corridor check and the research toolkit`,
  `- [Web workspace](${repo}/blob/main/web/README.md): draw or import layers, compare retention scenarios and export reports; runs locally with no account`,
  `- [Woodland co-ops on VergeCommon](${vergecommonWoodland}): neighbors check every harvest plan against shared corridor layers; that page says whether woodland projects are open`,
  `- [Unbroken Woods](${config.origin}/unbroken/): the first-season campaign with consulting foresters and woodland owners`,
  '',
  '## Agent guide',
  '',
  `- [AGENTS.md](${repo}/blob/main/AGENTS.md): rules for agents helping people use DFM, and for coding agents contributing`,
  '',
  '## Optional',
  '',
  `- [Source code](${repo}): MIT; cite with CITATION.cff`,
  `- [VergeCommon](${config.vergecommon}): open-source software for anyone to start a conservation co-op; its agent index is at ${config.vergecommon}llms.txt`,
  '- [Global Soil Intelligence Project](https://globalsoilintelligence.com/llms.txt): open soil observation library with read-only agent interfaces',
  '',
].join('\n'));
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
