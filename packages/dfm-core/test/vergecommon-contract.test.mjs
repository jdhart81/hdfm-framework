// Contract with VergeCommon's woodland projects (VergeCommon v0.10.0 vendors dfm-core 0.2.0).
// VergeCommon stores each treatment plan's corridor check, and a steward's "Download check
// inputs" gives the Landscape Package the plan was checked against. dfm-core must read that
// package and reproduce the stored result and its input checksum exactly; if this test fails,
// a change here would break reproducibility of plans stored in VergeCommon.
//
// fixtures/vergecommon-plan-package.json is the output of `node scripts/dfm-contract-fixture.mjs`
// in the VergeCommon repository (synthetic co-op, one consented woodlot, a two-unit plan).
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import Ajv2020 from 'ajv/dist/2020.js';
import {checkConnectivity, checkConnectivitySync, fromLandscapePackage, toLandscapePackage, canonicalHashSync, ENGINE_VERSION} from '../src/index.mjs';

const read = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const fixture = await read('../fixtures/vergecommon-plan-package.json');
const {package: pkg, stored} = fixture;

test('a VergeCommon check package is a valid v1 Landscape Package', async () => {
  const ajv = new Ajv2020({allErrors: true, strict: false});
  const valid = ajv.compile(await read('../../dfm-schema/landscape-package.schema.json'));
  assert.ok(valid(pkg), JSON.stringify(valid.errors));
  assert.equal(pkg.generator, 'VergeCommon');
  // VergeCommon's canonical check input: every v1 layer present, boundary empty (co-op parcels
  // stand in for it), consent status on each woodlot.
  assert.deepEqual(pkg.layers.boundary, []);
  assert.ok(pkg.layers.parcels.length && pkg.layers.parcels.every(p => ['covered', 'none'].includes(p.properties.consent)));
  assert.equal(stored.inputForm, 'landscape-package-1.0');
});

test('dfm-core reproduces the result and input checksum VergeCommon stored', async () => {
  // A stored check names the engine that made it. Later engines must give the same result for the
  // same input (they may name themselves differently), so plans stored in VergeCommon stay
  // reproducible as dfm-core moves on.
  assert.match(stored.engine, /^dfm-connectivity-\d+\.\d+\.\d+$/);
  assert.match(ENGINE_VERSION, /^dfm-connectivity-\d+\.\d+\.\d+$/);
  const input = fromLandscapePackage(pkg);
  assert.equal(canonicalHashSync(input), stored.inputChecksum);
  for (const result of [checkConnectivitySync(input), await checkConnectivity(input)]) {
    assert.equal(result.inputChecksum, stored.inputChecksum);
    assert.equal(result.status, stored.status);
    assert.deepEqual(result.lostLinks, stored.lostLinks);
    assert.deepEqual(result.pinchedLinks, stored.pinchedLinks);
    // VergeCommon stores numbers rounded to 1e-7.
    for (const k of ['committedM2', 'proposedM2'])
      assert.ok(Math.abs(result.consent[k] - stored.consent[k]) < 1e-6, k);
  }
});

test('writing the package back changes nothing VergeCommon would check', () => {
  const input = fromLandscapePackage(pkg);
  const again = fromLandscapePackage(JSON.parse(JSON.stringify(toLandscapePackage(input, {name: pkg.name, generator: 'VergeCommon', created: pkg.created}))));
  assert.deepEqual(again, input);
  assert.equal(canonicalHashSync(again), stored.inputChecksum);
});
