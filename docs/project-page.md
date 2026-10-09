# Project homepage

The project page is prepared for
[bincequ.github.io/RoboHarness](https://bincequ.github.io/RoboHarness/).
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
Q-score table, task cards and exact step limits. It validates each referenced
prompt hash. Directory-reported scores are authoritative; the build does not
substitute values from conflicting evaluator JSON.

The page presents the supplied manuscript's abstract, overview, object catalog
and robot-data comparison. The 100-object catalog is visible beside a description
of the objects, environments and success criterion from the manuscript. The
abstract retains the manuscript wording, with LaTeX citations removed and the
open-source release statement updated to reflect the public repository. The full
task table is expandable; evaluation notes remain alongside it. A final
"Get started" section highlights the open-source release, links to the GitHub
repository homepage, and shows three commands to clone and set up the project.
The manuscript source and its hash are
recorded in the [asset manifest](assets/sources.json).

The page uses the original manuscript's overview, object catalog and robot-data
comparison figures, plus its RoboHarness wordmark. The READMEs also include the
keypoint-tracking figure.
Original PDFs and browser-friendly renders are in [`docs/assets/`](assets).
The [asset manifest](assets/sources.json) records source paths in the supplied
`roboharness.zip`, SHA-256 hashes and the rendering process. Only outer white
margins were cropped; figure content and plotted values were not edited.
Paper figures are identified as historical research results. Dataset and
third-party terms remain applicable; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The READMEs embed the checked-in PNGs using relative paths under `docs/assets/`.
The website build copies those same files into its generated `assets/` directory.
When replacing an image, commit the PNG alongside its source PDF and update the
image hash in the asset manifest. After publication, verify the repository,
README images and project page without signing in.

## Task cards and recorded rollouts

The page order is Abstract → Method → Task results → Generalization → Get started.
The Method section contains only its heading and the full promotional video.
`website/method-video.json` records its source directory and the checked-in
published snapshot. It began with `roboharness_promo_v5.mp4`; the selected source
file is recorded in the manifest and preserved without re-encoding.
The build validates the video and poster hashes, copies the
assets under `assets/method/`, and includes their hashes in the playback URLs.

Use `python3 scripts/method_video.py` to refresh the checked-in snapshot from
the source directory. `build_site.py --latest-method` and the offline packager
select the highest numbered completed `roboharness_promo_vN.mp4`. A newly written
file must settle and pass full decoding (or have a matching successful validation
report); an unfinished export leaves the existing offline package available.
Engineering files and preview images are not candidates.

Each task has a card with its title and full official instruction above two
equal-height panels: the head-camera video on the left and the historical
per-instance Q-scores on the right. Previous and next arrows at the bottom
switch tasks and wrap at either end. Keyboard arrow keys also switch tasks;
the video retains its native playback controls. On narrow screens, the score
chart uses horizontal rows while the video and scores remain side by side.

`website/task-instructions.json` pins the instructions to the upstream
`StanfordVL/BEHAVIOR-1K` task catalog, with the source commit and content hashes.
The builder validates the task names and instruction hashes.
Use `?task=task05#task-results`
to link to a specific task.

`website/rollouts.json` selects
the highest final Q among the available scored recordings for each task; ties
use the shortest recording. It records candidate scores, the selected evaluator
result, video and poster hashes, and any missing tail. The builder checks the
selection and hashes before copying these assets into the site.

Recordings are available for task01, task02, task05, task07 and task08. Task06
has only an interrupted recording and is explicitly unscored. Task00, task03
and task09 have no usable head-camera recording. The recorded-run score in the
video caption is independent of the historical scores in the chart.

The native evaluation recorder combines wrist cameras on the left with a
448 × 448 head-camera view on the right. The published clips retain that right
view, the original 30 fps, full available duration and simulator timeline.
They use H.264/yuv420p, CRF 26 and faststart, without audio or source metadata.
Playback defaults to 4×; the speed button cycles through 1×, 2×, 4× and 8×.
Reduced-motion preferences disable autoplay and the card transition.
Task02's recovered
recording ends 6.2 simulator seconds before its evaluator result. Original files
were preserved; every published video was fully decoded before publication.

Videos and posters live in [`docs/assets/rollouts/`](assets/rollouts). The public
build copies only those referenced by the recording manifest, alongside the
paper figures. Raw agent traces and standalone reasoning exports are kept locally
and are excluded from the Pages artifact. Rebuilding an existing output directory also
removes trace JSON exported by earlier versions of the page.

## Deployment

The deployment job is skipped while the repository is private.

In the repository's **Settings → Pages**, select **GitHub Actions** as the
publishing source. The [`pages.yml` workflow](../.github/workflows/pages.yml)
builds and deploys changes to the page, task manifests or prompts on `main`.
It can also be run manually. Only the generated `_site/`
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

## Offline copy and laptop sync

Package the complete page into a standalone folder, including the Method film,
six published head-camera videos, images, original figure PDFs, scores and instructions:

```bash
python3 scripts/package_site.py --output ~/roboharness_homepage --zip ~/roboharness_homepage.zip
```

Copy the generated folder to the laptop. Double-click `index.html` or `open.cmd`
to browse without a connection, Python installation or local web server. External
GitHub and official-task links still require a connection. Legal documents are
included locally. The package includes a SHA-256 manifest and Windows sync tools;
passwords, private keys and separate agent-trace exports are excluded.

The configured laptop location is
`C:\Users\16593\Desktop\robotics\roboharness_homepage`. `sync.cmd` pulls changed
files from `/home/bince/roboharness_homepage` on 12023 using
`bince@221.12.22.151` and SSH port `12023` with the laptop's existing credentials.
`enable-sync.cmd` enables a background check every minute and starts it again at
Windows sign-in; `disable-sync.cmd` stops it. Transfers
are checked before replacing files. Network failures preserve the offline copy.
The laptop does not upload edits or delete personal files.

Use `--watch` on the server to refresh the package after website source changes.
The package's `README-offline.md` describes the SSH alias and port options.
The 12023 user service `roboharness-homepage-publisher.service` checks every
30 seconds and refreshes the folder and ZIP. It also watches the Method export
directory and publishes new completed versions to Pages. Publication uses a
dedicated generated checkout under `.local/method-publisher/`; it fetches current
`main`, commits only the Method video, poster and snapshot manifest, and pushes
that commit. It never stages files from the working repository. Failed publication
retries on the next check while the updated offline package remains available.
Its unit is included in `scripts/` and enables `package_site.py --publish-method`.
