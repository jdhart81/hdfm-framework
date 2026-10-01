# Contributing

Use a focused branch and pull request. Describe the concrete input, observed behavior, expected behavior and reproducible example. Do not upload private landowner datasets or credentials.

## Working application

From `web/`: run `npm ci`, `npm test`, and `npm run build`. Geometry changes need a hand-checkable fixture covering units, overlaps and exclusions. Preserve owner-scoped queries, revision conflict protection, old scenario records and immutable database migrations. Do not place a deployment-specific Site ID or secrets in the public checkout.

## Research toolkit

From `hdfm-framework/`: install the package and run `python -m pytest -q`. Read `docs/RESEARCH_STATUS.md` at the repository root before changing scientific claims. A passing test is not field validation. Use deterministic cases and distinguish minimum length, objective score and ecological outcome.

## Releases

The public repository contains the reusable application. The Viridis deployment checkout has its own hosting identity. Promote reviewed `web/` source to that checkout while preserving its `.openai/hosting.json`, then build and deploy. Never copy a developer's `.data`, `.env`, credentials, Git metadata or `node_modules` into a release.

## Sign your commits (DCO)

Pull requests from forks need a `Signed-off-by` line on every commit, matching the commit author's email:

    Signed-off-by: Your Name <you@example.com>

`git commit -s` adds it, and `git rebase --signoff origin/main` fixes an existing branch. Signing off
certifies the Developer Certificate of Origin 1.1 (https://developercertificate.org): you wrote the
change, or you have the right to submit it under this repository's license. The DCO check blocks
unsigned commits.

## License of contributions

Contributions are licensed under the same license as the files they change (see `LICENSE`), with no
additional terms. Don't submit work you can't license that way.

## Never commit

Credentials, API keys, private keys, `.env` files, customer or partner data, or wallet files. The secret
scan blocks known key formats. If you find a leaked secret, report it privately as described in
`SECURITY.md`.

## Names and marks

The license does not cover Viridis names, logos or certification marks. See `TRADEMARKS.md`.
