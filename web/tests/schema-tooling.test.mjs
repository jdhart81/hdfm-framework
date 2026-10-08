import test from 'node:test';
import assert from 'node:assert/strict';
import {cp, mkdtemp, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import {createRequire} from 'node:module';
import {runInNewContext} from 'node:vm';
import initSqlJs from 'sql.js';

const web = fileURLToPath(new URL('../', import.meta.url));
const kit = join(web, 'node_modules/drizzle-kit/bin.cjs');
const require = createRequire(import.meta.url);

test('legacy schema-loader transforms keep their CommonJS and ESM behavior', async () => {
  const {transformSync, transform} = require('@esbuild-kit/core-utils');
  const source = 'type Value = {answer: number}; const value: Value = {answer: 42}; export const answer: number = value.answer;';
  const cjs = transformSync(source, join(web, 'probe.cts'));
  const module = {exports: {}};
  runInNewContext(cjs.code, {module, exports: module.exports});
  assert.equal(module.exports.answer, 42);
  assert.equal(cjs.map.version, 3);
  const esm = await transform(source, join(web, 'probe.mts'));
  assert.equal((await import('data:text/javascript;base64,' + Buffer.from(esm.code).toString('base64'))).answer, 42);
  assert.equal(esm.map.version, 3);

  const directory = await mkdtemp(join(web, '.schema-loader-'));
  try {
    await writeFile(join(directory, 'package.json'), '{"type":"module"}\n');
    await writeFile(join(directory, 'value.ts'), 'export const answer: number = 42;\n');
    await writeFile(join(directory, 'probe.mts'), "import {answer} from './value.ts';\nconsole.log(JSON.stringify({answer, esm: import.meta.url.endsWith('probe.mts')}));\n");
    const result = spawnSync(process.execPath, ['--loader', require.resolve('@esbuild-kit/esm-loader'), join(directory, 'probe.mts')], {cwd: directory, encoding: 'utf8'});
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.deepEqual(JSON.parse(result.stdout), {answer: 42, esm: true});
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});

async function files(directory, prefix = '') {
  const result = {};
  for (const entry of await readdir(directory, {withFileTypes: true})) {
    const relative = join(prefix, entry.name);
    if (entry.isDirectory()) Object.assign(result, await files(join(directory, entry.name), relative));
    else result[relative] = await readFile(join(directory, entry.name), 'utf8');
  }
  return result;
}

test('schema tooling reads TypeScript ESM config and preserves existing migrations', async () => {
  // Keep imports resolvable through web/node_modules, but never generate in the real migration directory.
  const directory = await mkdtemp(join(web, '.schema-tooling-'));
  try {
    await writeFile(join(directory, 'package.json'), '{"type":"module"}\n');
    await cp(join(web, 'db'), join(directory, 'db'), {recursive: true});
    await cp(join(web, 'drizzle'), join(directory, 'drizzle'), {recursive: true});
    await writeFile(join(directory, 'settings.ts'), "export const settings = {dialect: 'sqlite' as const, schema: './db/schema.ts', out: './drizzle'};\n");
    await writeFile(join(directory, 'drizzle.config.ts'), "import {settings} from './settings.ts';\nexport default settings;\n");
    const before = await files(join(directory, 'drizzle'));
    const generate = () => {
      const result = spawnSync(process.execPath, [kit, 'generate', '--config=drizzle.config.ts', '--name=schema-tooling-probe'], {cwd: directory, encoding: 'utf8'});
      assert.equal(result.status, 0, result.stdout + result.stderr);
      return result.stdout + result.stderr;
    };
    assert.match(generate(), /No schema changes/);
    assert.deepEqual(await files(join(directory, 'drizzle')), before);

    const schema = join(directory, 'db/schema.ts');
    await writeFile(schema, await readFile(schema, 'utf8') + "\nexport const schemaToolingProbe = sqliteTable('schema_tooling_probe', {id: text('id').primaryKey(), revision: integer('revision').notNull()});\n");
    generate();
    const after = await files(join(directory, 'drizzle'));
    for (const [name, content] of Object.entries(before)) {
      if (name !== 'meta/_journal.json') assert.equal(after[name], content, `Existing migration changed: ${name}`);
    }
    const originalJournal = JSON.parse(before['meta/_journal.json']);
    const journal = JSON.parse(after['meta/_journal.json']);
    assert.equal(journal.entries.length, originalJournal.entries.length + 1);
    assert.deepEqual(journal.entries.slice(0, -1), originalJournal.entries);
    assert.deepEqual({...journal, entries: []}, {...originalJournal, entries: []});
    const addedSql = Object.keys(after).filter(name => name.endsWith('.sql') && !(name in before));
    assert.equal(addedSql.length, 1);
    assert.equal(journal.entries.at(-1).tag + '.sql', addedSql[0]);
    assert.match(after[addedSql[0]], /CREATE TABLE `schema_tooling_probe`/);

    const SQL = await initSqlJs();
    const database = new SQL.Database();
    try {
      for (const name of Object.keys(after).filter(name => name.endsWith('.sql')).sort()) database.run(after[name]);
      assert.deepEqual(database.exec('PRAGMA table_info(schema_tooling_probe)')[0].values.map(row => [row[1], row[2], row[3], row[5]]), [
        ['id', 'TEXT', 1, 1], ['revision', 'INTEGER', 1, 0]
      ]);
      assert.deepEqual(database.exec("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")[0].values.flat(), ['projects', 'scenarios', 'schema_tooling_probe']);
    } finally {
      database.close();
    }
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});
