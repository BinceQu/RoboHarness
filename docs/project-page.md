# Project homepage

The project page is prepared for
[cbq349.github.io/RoboHarness](https://cbq349.github.io/RoboHarness/).
The source is in [`website/`](../website). It uses HTML, CSS and JavaScript,
with no Node.js build dependencies or third-party scripts.

## Build and preview

From the repository root, with Python 3.10 or later:

```bash
python3 scripts/build_site.py
python3 -m http.server 15079 --bind 127.0.0.1 --directory _site
```

Open `http://127.0.0.1:15079/`. Choose an unused local port if needed.
The generated `_site/` directory is ignored by Git. No simulator, model
endpoint, datasets or submodule checkout is needed to build the page.

## Content and score provenance

The build reads the checked-in `tasks/task*.json` manifests for the historical
Q-score table, task selector and exact step limits. It validates each referenced
prompt hash. Directory-reported scores are authoritative; the build does not
substitute values from conflicting evaluator JSON.

The separate validation panel uses the checked-in
[`gpu5-20260930-r5/report.json`](../validation_results/gpu5-20260930-r5/report.json)
snapshot. Its partial means include only cases with official scoring hashes and
normal `model_done` or `evaluator_end` finishes. Wall-clock-truncated cases are
excluded. The panel explicitly identifies r5 as diagnostic and incomplete for
reproduction verification. Update its copy and source together when a new
validation round is published.

The page and READMEs use the original manuscript's overview, keypoint-tracking,
object-catalog and robot-data comparison figures, plus its RoboHarness wordmark.
Original PDFs and browser-friendly renders are in [`docs/assets/`](assets).
The [asset manifest](assets/sources.json) records source paths in the supplied
`roboharness.zip`, SHA-256 hashes and the rendering process. Only outer white
margins were cropped; figure content and plotted values were not edited.
Paper figures are identified as historical research results. Dataset and
third-party terms remain applicable; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Deployment

In the repository's **Settings → Pages**, select **GitHub Actions** as the
publishing source. The [`pages.yml` workflow](../.github/workflows/pages.yml)
builds and deploys changes to the page, task manifests, prompts or validation
records on `main`. It can also be run manually. Only the generated `_site/`
files enter the Pages artifact; local run data and credentials are not included.

For branch-based publishing, build into a fresh directory, commit its contents
to a separate `gh-pages` branch, and select **Deploy from a branch → gh-pages →
/(root)** in Pages settings. Include the generated `.nojekyll` file. The branch
must contain the generated site at its root, not the source `website/` template.
This mode requires rebuilding and pushing the site after source changes.

A successful Git push does not confirm that Pages is live. Check the Pages
deployment status and open the published URL without signing in. If GitHub
reports that Actions is disabled for the account, repository permissions alone
cannot restore it; see [GitHub's account-state guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#managing-github-actions-permissions-for-your-repository).
