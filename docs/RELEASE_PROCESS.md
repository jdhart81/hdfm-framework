# Source and deployment

`web/` is the public application source. The Viridis Site checkout uses the same application files and an owner-specific `.openai/hosting.json` which is not committed here.

For a release, promote reviewed `web/` files into the deployment checkout while preserving its hosting identity and excluding local data, environment files, dependencies and Git metadata. Run the application tests and build, commit and push that exact Site source, package its output, and deploy a saved version. Verify the terminal deployment result and retain the audience settings. Never use a public contributor's manifest to replace the live project's identity.

Do not edit applied SQL migrations. Existing scenario records retain their original method versions; new methods belong in separate comparison groups. A code release is not authorization to expose private projects or enable billing.
