#!/usr/bin/env python3
"""Build the former project site with redirects to the organization homepage.

Keep its static assets so existing image and PDF links still work. HTML pages
move to the new origin while preserving the path, query string and fragment.
"""

import argparse
import html
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = "https://roboharness-lab.github.io"
OLD_PREFIX = "/RoboHarness"


def redirect_page(relative):
    path = relative.as_posix()
    if path.endswith("index.html"):
        path = path.removesuffix("index.html")
    fallback = html.escape(f"{DESTINATION}/{path}", quote=True)
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex">
  <link rel="canonical" href="{fallback}">
  <title>RoboHarness</title>
  <script>
    (() => {{
      const prefix = {json.dumps(OLD_PREFIX)};
      let path = location.pathname;
      if (path === prefix || path.startsWith(prefix + "/")) {{
        path = path.slice(prefix.length);
      }}
      const destination = new URL({json.dumps(DESTINATION)});
      destination.pathname = path || "/";
      destination.search = location.search;
      destination.hash = location.hash;
      location.replace(destination.href);
    }})();
  </script>
  <noscript><meta http-equiv="refresh" content="0; url={fallback}"></noscript>
</head>
<body>
  <p>RoboHarness has moved to <a href="{fallback}">roboharness-lab.github.io</a>.</p>
</body>
</html>
'''


def build(site, output):
    site, output = site.resolve(), output.resolve()
    if output == site or output.is_relative_to(site) or site.is_relative_to(output):
        raise ValueError("Choose a separate output directory")
    for page in ("index.html", "v1/index.html", "v2/index.html"):
        if not (site / page).is_file():
            raise ValueError(f"Build the project site first: missing {page}")
    # Refuse to overwrite an existing directory or retain stale files.
    shutil.copytree(site, output)
    pages = list(output.rglob("*.html"))
    for page in pages:
        page.write_text(redirect_page(page.relative_to(output)))
    # GitHub Pages uses this for old deep links that have no matching file.
    (output / "404.html").write_text(redirect_page(Path("index.html")))
    print(f"Built {output}: {len(pages)} page redirects, 404 fallback and original assets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "_site")
    parser.add_argument("--output", type=Path, default=ROOT / "_legacy_site")
    args = parser.parse_args()
    build(args.site, args.output)


if __name__ == "__main__":
    main()
