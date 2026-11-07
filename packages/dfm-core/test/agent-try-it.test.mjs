// Locks the commands in the repository README's "Try it from your agent" section: each one runs
// as written from packages/dfm-core, and the woodlot command prints the output the README shows.
// If the engine changes that output, update the README with it.
import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {mkdtemp, readFile, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const pkgRoot = fileURLToPath(new URL('..', import.meta.url));
const readmeUrl = new URL('../../../README.md', import.meta.url);
const skip = !existsSync(readmeUrl) && 'repository README not present';

async function section() {
  const readme = await readFile(readmeUrl, 'utf8');
  const start = readme.indexOf('\n## Try it from your agent\n');
  assert.ok(start >= 0, 'README has a "Try it from your agent" section');
  const end = readme.indexOf('\n## ', start + 1);
  const text = readme.slice(start, end < 0 ? undefined : end);
  const blocks = [...text.matchAll(/```(\w*)\n([\s\S]*?)```/g)].map(m => ({lang: m[1], body: m[2]}));
  const commands = blocks.filter(b => b.lang === 'sh').map(b => b.body.match(/node --input-type=module -e "([^"]*)"(?: (\S+))?\n/)).filter(Boolean);
  return {text, blocks, commands};
}

const run = (code, args = []) => promisify(execFile)(process.execPath, ['--input-type=module', '-e', code, ...args], {cwd: pkgRoot});

test('the README woodlot command prints the output the README shows', {skip}, async () => {
  const {blocks, commands} = await section();
  assert.equal(commands.length, 2, 'two node commands: the woodlot and a Landscape Package');
  const expected = blocks[blocks.findIndex(b => b.body.includes('node --input-type=module')) + 1];
  assert.equal(expected.lang, '', 'expected output follows the woodlot command');
  const {stdout} = await run(commands[0][1]);
  assert.equal(stdout.trim(), expected.body.trim());
});

test('the README Landscape Package command runs on a package file', {skip}, async () => {
  const {text, commands} = await section();
  assert.match(text, /does not establish species movement/);
  const {package: pkg} = JSON.parse(await readFile(new URL('../fixtures/vergecommon-plan-package.json', import.meta.url), 'utf8'));
  const dir = await mkdtemp(join(tmpdir(), 'dfm-try-it-'));
  try {
    const file = join(dir, 'landscape.json');
    await writeFile(file, JSON.stringify(pkg));
    const {stdout} = await run(commands[1][1], [file]);
    const [status, ...rest] = stdout.trim().split(' ');
    assert.equal(status, 'fail');
    const out = JSON.parse(rest.join(' '));
    assert.deepEqual(out.lostLinks.map(l => l.causes), [['harvest-3']]);
    assert.deepEqual(out.pinchedLinks, []);
  } finally {
    await rm(dir, {recursive: true, force: true});
  }
});
