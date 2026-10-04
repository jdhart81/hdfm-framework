// Checks on the built site: structure, links, privacy and claims.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, readdir, stat} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dist = path.join(path.dirname(path.dirname(fileURLToPath(import.meta.url))), 'dist');
async function files(dir = dist) {
  const out = [];
  for (const e of await readdir(dir, {withFileTypes: true})) {
    const f = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...await files(f)); else out.push(f);
  }
  return out;
}
const pages = (await files()).filter(f => f.endsWith('.html'));
const html = Object.fromEntries(await Promise.all(pages.map(async f => [path.relative(dist, f), await readFile(f, 'utf8')])));

test('every page has a language, one h1, a title, a description and a canonical URL', () => {
  assert.ok(pages.length >= 4);
  for (const [name, h] of Object.entries(html)) {
    assert.match(h, /<html lang="en">/, name);
    assert.equal((h.match(/<h1[\s>]/g) || []).length, 1, name);
    assert.match(h, /<title>[^<]{10,}<\/title>/, name);
    assert.match(h, /<meta name="description" content="[^"]{30,}">/, name);
    if (name === '404.html') {
      assert.match(h, /<meta name="robots" content="noindex">/, '404 page is not indexed');
      assert.doesNotMatch(h, /rel="canonical"/, '404 page has no canonical');
    } else assert.match(h, /<link rel="canonical" href="https:\/\/dendriticforest\.com\//, name);
  }
});

test('every internal link and asset resolves to a built file', async () => {
  for (const [name, h] of Object.entries(html))
    for (const [, url] of h.matchAll(/(?:href|src)="(\/[^"#?]*)/g)) {
      let target = path.join(dist, url);
      try { if ((await stat(target)).isDirectory()) target = path.join(target, 'index.html'); await stat(target); }
      catch { assert.fail(`${name} links to missing ${url}`); }
    }
});

test('no third-party requests: no external scripts, styles, fonts or images', () => {
  for (const [name, h] of Object.entries(html)) {
    assert.doesNotMatch(h, /<script/i, name);
    const loaded = h.replace(/<link rel="canonical"[^>]*>/g, '');
    assert.doesNotMatch(loaded, /<(?:link|img|source)[^>]+(?:href|src)="https?:/i, name);
  }
});

test('claims stay within what the check establishes', () => {
  const banned = [/guarantee/i, /certif/i, /protects? (?:old growth|species|wildlife)/i, /carbon credits? (?:earned|issued)/i, /proven/i];
  for (const [name, h] of Object.entries(html)) {
    const text = h.replace(/<[^>]+>/g, ' ');
    for (const b of banned) assert.doesNotMatch(text, b, `${name}: ${b}`);
  }
  assert.match(html['index.html'], /does not establish/);
  assert.match(html['unbroken/index.html'], /We will not say/);
});

test('the campaign page and its pilot call to action exist', () => {
  const h = html['unbroken/index.html'];
  assert.match(h, /Every harvest\. Every woodlot\. One unbroken forest\./);
  assert.match(h, /id="pilot"/);
  assert.match(h, /href="https:\/\/[^"]+">Tell us about your woodlot/);
});

test('the map illustration is labelled for screen readers and marked as not a real site', () => {
  for (const name of ['index.html', 'unbroken/index.html']) {
    assert.match(html[name], /<svg[^>]+role="img"[^>]+aria-labelledby="map-title map-desc"/);
    assert.match(html[name], /Illustration, not a real site/);
  }
});

test('sitemap lists the public pages', async () => {
  const xml = await readFile(path.join(dist, 'sitemap.xml'), 'utf8');
  for (const u of ['/', '/unbroken/', '/data-format/']) assert.match(xml, new RegExp(`<loc>https://dendriticforest\\.com${u.replace(/\//g, '\\/')}</loc>`));
  assert.doesNotMatch(xml, /404/);
});

test('the site origin comes only from site.config.json and is used in canonicals, sitemap and robots', async () => {
  const {origin} = JSON.parse(await readFile(path.join(dist, '..', 'site.config.json'), 'utf8'));
  assert.equal(origin, 'https://dendriticforest.com');
  const robots = await readFile(path.join(dist, 'robots.txt'), 'utf8');
  assert.match(robots, new RegExp(`^Sitemap: ${origin.replace(/\./g, '\\.')}/sitemap\\.xml$`, 'm'));
  for (const f of await files()) {
    if (!/\.(html|xml|txt)$/.test(f) || f.includes(`${path.sep}fonts${path.sep}`)) continue;
    assert.doesNotMatch(await readFile(f, 'utf8'), /dendriticforest\.org/, path.relative(dist, f));
  }
});

test('every in-page and cross-page #fragment points at an element that exists', () => {
  const ids = page => new Set([...html[page].matchAll(/\sid="([^"]+)"/g)].map(m => m[1]));
  for (const [name, h] of Object.entries(html))
    for (const [, target, frag] of h.matchAll(/href="([^"#]*)#([^"]+)"/g)) {
      if (/^https?:/.test(target)) continue;
      const page = target === '' ? name : path.join(target.replace(/^\//, ''), target.endsWith('/') || target === '' ? 'index.html' : '');
      assert.ok(html[page], `${name}: ${target}#${frag} names a page that was not built`);
      assert.ok(ids(page).has(frag), `${name}: #${frag} not found on ${page}`);
    }
});

test('every page links to VergeCommon, and nothing promises what is not live yet', async () => {
  const {vergecommon} = JSON.parse(await readFile(path.join(dist, '..', 'site.config.json'), 'utf8'));
  for (const [name, h] of Object.entries(html)) {
    assert.ok(h.includes(`href="${vergecommon}"`), `${name} links to VergeCommon`);
    // Woodland projects are behind a feature flag in VergeCommon; dfm-core is not on npm yet.
    assert.doesNotMatch(h, /Start a woodland project/i, name);
    assert.doesNotMatch(h, /npm install @viridis/i, name);
    assert.doesNotMatch(h, /with the VergeCommon conservation co-ops/i, name);
  }
  assert.match(html['data-format/index.html'], /not on npm yet/);
  assert.match(html['index.html'], /being tested in VergeCommon/);
});

test('no inline styles or scripts, which the site CSP would block', () => {
  for (const [name, h] of Object.entries(html)) {
    assert.doesNotMatch(h, /\sstyle="/, `${name}: style attribute`);
    assert.doesNotMatch(h, /<style[\s>]/, `${name}: <style> element`);
    assert.doesNotMatch(h, /<script[\s>]|\son[a-z]+="/, `${name}: script or event handler`);
  }
});

test('every table sits in a keyboard-scrollable, labelled region', () => {
  for (const [name, h] of Object.entries(html)) {
    const tables = (h.match(/<table[\s>]/g) || []).length;
    const wrapped = (h.match(/<div class="table-scroll" role="region" aria-label="[^"]+" tabindex="0">\s*<table/g) || []).length;
    assert.equal(wrapped, tables, name);
  }
});

test('every id on a page is unique, including the reused spine geometry', () => {
  for (const [name, h] of Object.entries(html)) {
    const ids = [...h.matchAll(/\sid="([^"]+)"/g)].map(m => m[1]);
    const dupes = ids.filter((x, i) => ids.indexOf(x) !== i);
    assert.deepEqual(dupes, [], `${name} repeats ${dupes.join(', ')}`);
    for (const [, ref] of h.matchAll(/(?:href|url\()"?#([\w-]+)/g)) assert.ok(ids.includes(ref), `${name}: #${ref} has no target`);
  }
});

test('the spine section shows four stages in time order, decorative panels hidden from screen readers', () => {
  const h = html['index.html'];
  const section = h.slice(h.indexOf('<section id="spine">'), h.indexOf('</section>', h.indexOf('<section id="spine">')));
  assert.deepEqual([...section.matchAll(/class="spine-map spine-panel stage-([a-d])"/g)].map(m => m[1]), ['a', 'b', 'c', 'd']);
  assert.equal((section.match(/aria-hidden="true" focusable="false"/g) || []).length, 4);
  assert.deepEqual([...section.matchAll(/<h3>([^<]+)<\/h3>/g)].map(m => m[1]), ['Map it whole', 'Build it piece by piece', 'Let it grow old', 'Plan for change']);
  assert.match(section, /Age is not condition/);
});

test('the hero map plays its sequence only for people who allow motion, and its final frame tells the whole story', async () => {
  const css = await readFile(path.join(dist, 'styles.css'), 'utf8');
  const motion = css.slice(css.indexOf('@media (prefers-reduced-motion:no-preference){'));
  assert.match(motion, /^@media \(prefers-reduced-motion:no-preference\)\{[^@]*\.spine-hero \.piece\{animation/);
  // Outside the motion block nothing is hidden for good: pieces, numbers and the later climate line show by default.
  const still = css.slice(0, css.indexOf('@media (prefers-reduced-motion:no-preference){'));
  assert.doesNotMatch(still, /\.spine-hero[^{]*\{[^}]*(?:opacity:0|display:none)/);
  assert.match(still, /\.spine-map \.yr\.final\{opacity:1\}/);
  const hero = html['index.html'];
  for (const n of [1, 2, 3, 4, 5, 6, 7, 8, 9]) assert.match(hero, new RegExp(`class="piece p${n}" clip-path="url\\(#pc${n}-hero\\)"`));
});

test('the data format page documents every layer and parameter in the package schema', async t => {
  let schema;
  try { schema = JSON.parse(await readFile(path.join(dist, '..', '..', '..', 'packages', 'dfm-schema', 'landscape-package.schema.json'), 'utf8')); }
  catch (e) {
    // The container build copies only apps/site; the repository's CI step runs this check.
    if (e.code === 'ENOENT') return t.skip('the schema is outside this build context');
    throw e;
  }
  const page = html['data-format/index.html'];
  for (const layer of Object.keys(schema.properties.layers.properties)) assert.ok(page.includes(`<code>${layer}</code>`), `layer ${layer}`);
  for (const param of Object.keys(schema.properties.params.properties)) assert.ok(page.includes(`<code>${param}</code>`), `parameter ${param}`);
});

test('the spine section claims only what the engine tests', () => {
  const home = html['index.html'];
  // Loops are tested for single points of failure; the page must not promise that a loop always gives a way around.
  assert.doesNotMatch(home, /second way around/i);
  assert.match(home, /shows where one disturbance/);
  assert.match(home, /Age is not condition/);
});
