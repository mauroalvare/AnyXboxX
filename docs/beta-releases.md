# Windows x64 beta releases

French version in [docs/fr/versions-beta.md](fr/versions-beta.md).

1. Land the optimizations you want in **`beta/optimizations`** on [the fork](https://github.com/mauroalvare/AnyXboxX/tree/beta/optimizations). The workflow fetches that branch and the original repository's `main` at the start of each run.
2. In the fork, open **Actions → Windows x64 beta → Run workflow**, select `main`, and start the run. Each run creates its own `beta/integration-…` branch and builds and tests that exact commit.
3. Open [Releases](https://github.com/mauroalvare/AnyXboxX/releases) after the run succeeds. Under **Assets** of the new `beta-…` pre-release, download `AnyXboxX-windows-x64.zip`. The release notes and `revisions.json` in the ZIP give the two input SHAs and the tested commit.
4. Extract the ZIP, run `Play-AnyXboxX.cmd`, and select the folder with your own game copy. See the [Windows requirements](running-gta6.md#requirements).

If integration fails, open **Summary** on the Actions run for the conflicting paths. Merge upstream `main` into `beta/optimizations` in your checkout, resolve the listed paths and push the result, then start a new run. The starting branch also pins outstanding optimization topics in `.github/beta-optimization-topics.txt`. For a conflict reported under a topic name, merge that pinned commit into `beta/optimizations`, resolve the paths, and remove the corresponding entry once the commit is included.

GitHub may reject the integration branch push when upstream introduces workflow files that are new to the fork. In that case, set the fork's `BETA_INTEGRATION_TOKEN` Actions secret to a fine-grained token scoped to this fork with write access to **Contents** and **Workflows**. The final release uses the Actions token.
