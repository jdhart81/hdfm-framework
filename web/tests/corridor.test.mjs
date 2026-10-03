// Phase 2: corridor check in the workspace, using @viridis/dfm-core on saved projects.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile, readdir} from 'node:fs/promises';
import {webcrypto} from 'node:crypto';
import * as turf from '@turf/turf';
import {fromLandscapePackage, checkConnectivitySync} from '@viridis/dfm-core';
import {sqlite} from '../server/sqlite.mjs';
import {api} from '../server/api.mjs';
import {normalize} from '../public/map-core.mjs';

globalThis.crypto ??= webcrypto;
const M = 111195.0802335329, pt = (x, y) => [x / M, y / M];
const rect = (x0, y0, x1, y1) => ({type: 'Polygon', coordinates: [[pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1), pt(x0, y0)]]});
const F = (layer, geometry, properties = {}) => ({type: 'Feature', geometry, properties: {dfm_layer: layer, ...properties}});
function landscape(treatments = []) {
  return {
    boundary: [F('boundary', rect(0, 0, 2000, 1000), {dfm_id: 'b'})],
    retention: [F('retention', rect(400, 440, 1600, 560), {dfm_id: 'corridor-1'})],
    cores: [F('cores', rect(100, 300, 400, 700), {dfm_id: 'core-A', core_class: 'old-growth-candidate'}), F('cores', rect(1600, 300, 1900, 700), {dfm_id: 'core-B', core_class: 'riparian-core'})],
    roads: [F('roads', {type: 'LineString', coordinates: [pt(1000, 0), pt(1000, 1000)]}, {dfm_id: 'road-1'})],
    crossings: [F('crossings', {type: 'Point', coordinates: pt(1000, 500)}, {dfm_id: 'crossing-1', passage: 'verified'})],
    treatments,
  };
}
const cut = F('treatments', rect(700, 300, 800, 800), {dfm_id: 'harvest-3', intensity: 'clearcut'});
async function database() {
  const DB = await sqlite();
  for (const f of (await readdir(new URL('../drizzle', import.meta.url))).filter(f => f.endsWith('.sql'))) await DB.migrate(await readFile(new URL('../drizzle/' + f, import.meta.url), 'utf8'));
  return DB;
}
const request = (path, method = 'GET', body, owner = 'alice') => new Request('https://dfm.test/api/' + path, {method, headers: {'oai-authenticated-user-id': owner, 'Content-Type': 'application/json', Origin: 'https://dfm.test'}, body: body ? JSON.stringify(body) : undefined});
const params = {minWidthM: 100, minWidthSource: 'Test policy', waterWidth: 50, roadWidth: 6};

test('saved project: corridor check passes without treatments and fails naming the unit that cuts the corridor', async () => {
  const DB = await database(), env = {DB};
  const flat = d => Object.values(d).flat();
  let p = await (await api(request('projects', 'POST', {name: 'Corridor test', data: landscape()}), env)).json();
  let r = await api(request(`projects/${p.id}/connectivity`, 'POST', {...params, revision: 1}), env);
  assert.equal(r.status, 200, await r.clone().text());
  let out = await r.json();
  assert.equal(out.result.status, 'pass', out.result.reasons.join(' '));
  assert.deepEqual(out.result.linkedAfter, [{a: 'core-A', b: 'core-B'}]);
  p = await (await api(request(`projects/${p.id}`, 'PUT', {name: 'Corridor test', data: {...landscape([cut])}, sources: {}, revision: 1}), env)).json();
  out = await (await api(request(`projects/${p.id}/connectivity`, 'POST', {...params, revision: 2}), env)).json();
  assert.equal(out.result.status, 'fail');
  assert.deepEqual(out.result.lostLinks[0].causes, ['harvest-3']);
  assert.equal((await api(request(`projects/${p.id}/connectivity`, 'POST', {...params, revision: 1}), env)).status, 409, 'stale revision');
  assert.equal((await api(request(`projects/${p.id}/connectivity`, 'POST', {...params, revision: 2}, 'bob'), env)).status, 404, 'other owner');
  assert.ok(flat(landscape()).length > 0);
});

test('the Landscape Package from the workspace reproduces the same result in dfm-core (I8)', async () => {
  const DB = await database(), env = {DB};
  const p = await (await api(request('projects', 'POST', {name: 'Package test', data: landscape([cut])}), env)).json();
  const out = await (await api(request(`projects/${p.id}/connectivity`, 'POST', {...params, revision: 1}), env)).json();
  assert.equal(out.package.dfm_package, '1.0');
  const again = checkConnectivitySync(fromLandscapePackage(out.package));
  assert.equal(again.inputChecksum, out.result.inputChecksum);
  assert.equal(again.status, 'fail');
});

test('missing width source or cores is incomplete (400-free), not a pass', async () => {
  const DB = await database(), env = {DB};
  const p = await (await api(request('projects', 'POST', {name: 'Incomplete', data: landscape()}), env)).json();
  const out = await (await api(request(`projects/${p.id}/connectivity`, 'POST', {...params, minWidthSource: '', revision: 1}), env)).json();
  assert.equal(out.result.status, 'incomplete');
});

test('drawn corridor layers get conservative defaults', () => {
  const core = normalize({type: 'Feature', geometry: rect(0, 0, 10, 10), properties: {}}, 'cores', 'Drawn', turf).cores[0];
  assert.equal(core.properties.core_class, 'old-growth-candidate');
  const crossing = normalize({type: 'Feature', geometry: {type: 'Point', coordinates: pt(5, 5)}, properties: {passage: 'magic'}}, 'crossings', 'Drawn', turf).crossings[0];
  assert.equal(crossing.properties.passage, 'assumed');
  const unit = normalize({type: 'Feature', geometry: rect(0, 0, 10, 10), properties: {}}, 'treatments', 'Drawn', turf).treatments[0];
  assert.equal(unit.properties.intensity, 'unrecorded');
});
