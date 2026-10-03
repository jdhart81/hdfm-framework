# dendriticforest.org

Static site for Dendritic Forest Management and the **Unbroken Woods** campaign (`/unbroken/`, also reached at unbrokenwoods.org). No runtime JavaScript, no third-party requests; fonts (Archivo, Source Serif 4, SIL OFL) are self-hosted.

```sh
npm ci
npm test          # builds dist/ and checks links, metadata, privacy and claims
npm run serve     # preview dist/ at http://127.0.0.1:4300
```

Pages live in `src/` with a small front-matter block; `src/partials/` holds the shared header, footer and map illustration. `site.config.json` sets the canonical origin and the pilot contact link.

## Hosting

Production runs on the VergeCommon server as an unprivileged nginx container (`Dockerfile`, port 8080) on the private `vergecommon` Docker network. VergeCommon's Caddy terminates TLS for dendriticforest.org and redirects unbrokenwoods.org to `/unbroken/`. The deployment runbook is in the VergeCommon repository: `docs/DFM_SITE.md`.

## Content rules

Say what the corridor check establishes and nothing more: corridors kept connected at the minimum width, old-growth *candidates*. Do not claim protected or verified old growth, carbon outcomes or species benefits. `test/site.test.mjs` fails on the most common overclaims.
