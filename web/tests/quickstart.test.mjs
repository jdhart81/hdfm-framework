import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

// Locks the worked example in QUICKSTART.md: if the method changes these
// numbers, the quickstart's expected output must be updated with it.
const expected = {
  boundaryHa: 772.772,
  bufferHa: 47.374,
  forestHa: 96.442,
  netForestHa: 96.442,
  retainedForestHa: 2.62,
  roadSurfaceHa: 3.069,
  openWaterHa: 0,
  managementOverlapHa: 0,
  crossingCandidates: 4,
};

test('QUICKSTART worked example reproduces from the command line', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'dfm-quickstart-'));
  try {
    const out = join(dir, 'assessment.json');
    const { stdout } = await promisify(execFile)(
      process.execPath,
      ['scripts/analyze.mjs', 'examples/fictional-watershed.geojson', out, '50', '6'],
    );
    assert.match(stdout, /7 data warnings\. Geometric screening only\./);
    const result = JSON.parse(await readFile(out, 'utf8'));
    assert.equal(result.method, 'dfm-geometric-0.3.0');
    assert.equal(result.issues.length, 7);
    for (const [key, value] of Object.entries(expected)) {
      assert.equal(Number(result.metrics[key].toFixed(3)), value, key);
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
