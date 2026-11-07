// Regenerate examples/*.json from the dfm-core fixtures. Deterministic: a fixed creation
// date, so rerunning gives byte-identical files.  node scripts/build-examples.mjs
import {writeFile, readFile} from 'node:fs/promises';
import {toLandscapePackage, deriveSpine} from '@viridis/dfm-core';
import {woodlot} from '../../dfm-core/fixtures/woodlot.mjs';
import {watershed, withAges} from '../../dfm-core/fixtures/watershed.mjs';
import {prairie} from '../../dfm-core/fixtures/prairie.mjs';

const CREATED = '2026-11-07T00:00:00.000Z';
const out = new URL('../examples/', import.meta.url);
const meta = name => ({name, generator: 'dfm-core fixtures', created: CREATED});
const save = (file, doc) => writeFile(new URL(file, out), `${JSON.stringify(doc)}\n`);

const ws = watershed();
const wsSpine = {...ws, retained: withAges(deriveSpine(ws).features)};

const examples = {
  'woodlot-clearcut': {about: 'Fictional woodlot: a clearcut across the only corridor between two core areas. The check fails and names the unit.', tryWith: ['check_plan'],
    package: toLandscapePackage(woodlot({treatments: ['harvest-3']}), meta('Fictional woodlot, clearcut plan'))},
  'woodlot-thinning': {about: 'The same woodlot with the same footprint recorded as a permitted light thinning. The check passes with a warning.', tryWith: ['check_plan'],
    package: toLandscapePackage(woodlot({treatments: ['thin-3']}), meta('Fictional woodlot, thinning plan'))},
  'watershed-lines': {about: 'Fictional watershed with streams, ridge and saddle links, four cores and six holdings. Draft the spine from its lines.', tryWith: ['draft_spine', 'spine_network'],
    package: toLandscapePackage(ws, meta('Fictional watershed, spine lines'))},
  'watershed-spine': {about: 'The same watershed with the drafted spine saved as retained habitat, with stand ages. Use it for projections, climate routes and the next holdings.', tryWith: ['check_plan', 'project_spine', 'climate_routes', 'next_holdings'],
    package: toLandscapePackage(wsSpine, meta('Fictional watershed, reviewed spine'))},
  'prairie-stepping-stones': {about: 'Fictional prairie: wet-meadow patches link remnants as stepping stones at a 100 m crossing distance, with an exit to the next landscape north.', tryWith: ['check_plan', 'spine_network', 'climate_routes'],
    package: toLandscapePackage(prairie(), meta('Fictional prairie, stepping stones'))},
};
for (const [name, doc] of Object.entries(examples)) await save(`${name}.json`, doc);

const vc = JSON.parse(await readFile(new URL('../../dfm-core/fixtures/vergecommon-plan-package.json', import.meta.url), 'utf8'));
await save('vergecommon-plan.json', {about: 'Synthetic VergeCommon woodland co-op plan, with the check result VergeCommon stored for it. dfm-core reproduces it exactly.', tryWith: ['check_plan'], package: vc.package, stored: vc.stored, origin: vc.source});
console.log(`wrote ${Object.keys(examples).length + 1} examples`);
