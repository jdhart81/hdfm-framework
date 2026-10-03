// Phase 0 hardening: storage failure recovery, error classification, deletion, startup checks.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, readdir, mkdtemp, rm, mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {webcrypto} from 'node:crypto';
import {sqlite} from '../server/sqlite.mjs';
import {api} from '../server/api.mjs';
import {example} from '../public/map-core.mjs';

globalThis.crypto ??= webcrypto;
const migrations = async DB => {
  for (const f of (await readdir(new URL('../drizzle', import.meta.url))).filter(f => f.endsWith('.sql')).sort())
    await DB.migrate(await readFile(new URL('../drizzle/' + f, import.meta.url), 'utf8'));
};
const request = (p, method = 'GET', body, owner = 'alice') => new Request('https://dfm.test/api/' + p, {
  method,
  headers: {'oai-authenticated-user-id': owner, 'Content-Type': 'application/json', Origin: 'https://dfm.test'},
  body: body === undefined ? undefined : JSON.stringify(body),
});

test('a failed disk write is not reported as saved, rolls back, and does not block later writes', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'dfm-'));
  try {
    const file = path.join(dir, 'dfm.sqlite');
    const DB = await sqlite(file);
    await migrations(DB);
    const env = {DB};
    let r = await api(request('projects', 'POST', {name: 'Kept', data: example()}), env);
    assert.equal(r.status, 201);

    await mkdir(file + '.tmp'); // the atomic-write temp path is now a directory: the next write fails (EISDIR)
    r = await api(request('projects', 'POST', {name: 'Lost', data: example()}), env);
    assert.equal(r.status, 503);
    assert.doesNotMatch(await r.text(), /EISDIR|tmp/);
    const list = await (await api(request('projects'), env)).json();
    assert.deepEqual(list.map(p => p.name), ['Kept'], 'the failed write must be rolled back in memory');

    await rm(file + '.tmp', {recursive: true});
    r = await api(request('projects', 'POST', {name: 'After recovery', data: example()}), env);
    assert.equal(r.status, 201, 'one failure must not block later writes');
    const reopened = await sqlite(file);
    const names = (await reopened.prepare('SELECT name FROM projects ORDER BY name').all()).results.map(x => x.name);
    assert.deepEqual(names, ['After recovery', 'Kept']);
  } finally {
    await rm(dir, {recursive: true, force: true});
  }
});

test('bad input is a 400 with its message; non-object bodies are rejected', async () => {
  const DB = await sqlite(); await migrations(DB); const env = {DB};
  const bad = example(); bad.roads[0].geometry.coordinates = [[0, 0], [999, 0]];
  let r = await api(request('projects', 'POST', {name: 'x', data: bad}), env);
  assert.equal(r.status, 400);
  assert.match((await r.json()).error, /longitude\/latitude/);
  r = await api(request('projects', 'POST', null), env);
  assert.equal(r.status, 400);
  r = await api(request('projects', 'POST', {name: 'x', data: example(), sources: {roads: null}}), env);
  assert.equal(r.status, 201, 'a null source entry is ignored, not a crash');
});

test('unexpected internal errors are 500 without internal detail', async () => {
  const env = {DB: {prepare() { throw new TypeError('secret internal detail'); }}};
  const r = await api(request('projects'), env);
  assert.equal(r.status, 500);
  assert.doesNotMatch(await r.text(), /secret internal detail/);
});

test('owners can delete scenarios and projects; other owners cannot', async () => {
  const DB = await sqlite(); await migrations(DB); const env = {DB};
  const p = await (await api(request('projects', 'POST', {name: 'Delete me', data: example()}), env)).json();
  const s = await (await api(request(`projects/${p.id}/scenarios`, 'POST', {name: 'S', revision: 1, waterWidth: 50, roadWidth: 6}), env)).json();
  assert.ok(s.id);
  assert.equal((await api(request(`projects/${p.id}/scenarios/${s.id}`, 'DELETE', undefined, 'bob'), env)).status, 404);
  assert.equal((await api(request(`projects/${p.id}/scenarios/${s.id}`, 'DELETE'), env)).status, 200);
  assert.equal((await (await api(request(`projects/${p.id}/scenarios`), env)).json()).length, 0);
  await api(request(`projects/${p.id}/scenarios`, 'POST', {name: 'S2', revision: 1, waterWidth: 50, roadWidth: 6}), env);
  assert.equal((await api(request(`projects/${p.id}`, 'DELETE', undefined, 'bob'), env)).status, 404);
  assert.equal((await api(request(`projects/${p.id}`, 'DELETE'), env)).status, 200);
  assert.equal((await api(request(`projects/${p.id}`), env)).status, 404);
  const orphans = await DB.prepare('SELECT COUNT(*) AS n FROM scenarios WHERE project=?').bind(p.id).first();
  assert.equal(orphans.n, 0, 'deleting a project removes its scenarios');
});

test('external hosting refuses to start without a strong password or with a non-https origin', () => {
  const run = env => spawnSync(process.execPath, ['server/local.mjs'], {cwd: new URL('..', import.meta.url), env: {...process.env, PORT: '0', ...env}, timeout: 5000, encoding: 'utf8'});
  let r = run({PUBLIC_ORIGIN: 'https://dfm.example.org', DFM_PASSWORD: ''});
  assert.equal(r.status, 1); assert.match(r.stderr, /DFM_PASSWORD/);
  r = run({PUBLIC_ORIGIN: 'http://dfm.example.org', DFM_PASSWORD: 'a-long-enough-secret'});
  assert.equal(r.status, 1); assert.match(r.stderr, /https/);
  r = run({PUBLIC_ORIGIN: 'https://dfm.example.org/path', DFM_PASSWORD: 'a-long-enough-secret'});
  assert.equal(r.status, 1);
});
