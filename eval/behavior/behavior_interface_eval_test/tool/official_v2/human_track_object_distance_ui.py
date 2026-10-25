"""Test-local Human Involve UI extension for named distance tracking."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, Response


_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_SCRIPT_NAME = "track_object_distance_human_ui.js"
_STYLE_NAME = "track_object_distance_human_ui.css"
_SCRIPT_URL = f"/__official__/assets/{_SCRIPT_NAME}"
_STYLE_URL = f"/__official__/assets/{_STYLE_NAME}"
_MOVE_SCRIPT_NAME = "move_tracked_point_human_ui.js"
_MOVE_STYLE_NAME = "move_tracked_point_human_ui.css"
# Keep the route path stable for direct callers, but version the injected URL
# so a browser tab cannot retain the pre-expression validator indefinitely.
_MOVE_ASSET_VERSION = "v17"
_MOVE_SCRIPT_PATH = f"/__official__/assets/{_MOVE_SCRIPT_NAME}"
_MOVE_STYLE_PATH = f"/__official__/assets/{_MOVE_STYLE_NAME}"
_MOVE_SCRIPT_URL = f"{_MOVE_SCRIPT_PATH}?{_MOVE_ASSET_VERSION}"
_MOVE_STYLE_URL = f"{_MOVE_STYLE_PATH}?{_MOVE_ASSET_VERSION}"


def _asset_response(name: str, mimetype: str) -> Response:
    source = (_ASSET_DIR / name).read_text(encoding="utf-8")
    return Response(
        source,
        mimetype=mimetype,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


def install_track_object_distance_human_ui(app: Flask) -> None:
    """Inject tracked-point controls into only the official test interface."""

    @app.get(_SCRIPT_URL, endpoint="official_track_object_distance_ui_js")
    def official_track_object_distance_ui_js() -> Response:
        return _asset_response(_SCRIPT_NAME, "application/javascript")

    @app.get(_STYLE_URL, endpoint="official_track_object_distance_ui_css")
    def official_track_object_distance_ui_css() -> Response:
        return _asset_response(_STYLE_NAME, "text/css")

    @app.get(_MOVE_SCRIPT_PATH, endpoint="official_move_tracked_point_ui_js")
    def official_move_tracked_point_ui_js() -> Response:
        return _asset_response(_MOVE_SCRIPT_NAME, "application/javascript")

    @app.get(_MOVE_STYLE_PATH, endpoint="official_move_tracked_point_ui_css")
    def official_move_tracked_point_ui_css() -> Response:
        return _asset_response(_MOVE_STYLE_NAME, "text/css")

    original_index = app.view_functions.get("index")
    if original_index is None:
        return

    def official_index():
        response = app.make_response(original_index())
        if response.status_code != 200 or not response.is_json:
            source = response.get_data(as_text=True)
            if _STYLE_URL not in source and "</head>" in source:
                source = source.replace(
                    "</head>",
                    f'<link rel="stylesheet" href="{_STYLE_URL}" />\n</head>',
                    1,
                )
            if _MOVE_STYLE_URL not in source and "</head>" in source:
                source = source.replace(
                    "</head>",
                    f'<link rel="stylesheet" href="{_MOVE_STYLE_URL}" />\n</head>',
                    1,
                )
            if _SCRIPT_URL not in source and "</body>" in source:
                source = source.replace(
                    "</body>",
                    f'<script src="{_SCRIPT_URL}"></script>\n</body>',
                    1,
                )
            if _MOVE_SCRIPT_URL not in source and "</body>" in source:
                source = source.replace(
                    "</body>",
                    f'<script src="{_MOVE_SCRIPT_URL}"></script>\n</body>',
                    1,
                )
            response.set_data(source)
            response.headers.pop("Content-Length", None)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    app.view_functions["index"] = official_index


__all__ = ["install_track_object_distance_human_ui"]
