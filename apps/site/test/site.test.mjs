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
    assert.match(h, /<link rel="canonical" href="https:\/\/dendriticforest\.com\//, name);
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
