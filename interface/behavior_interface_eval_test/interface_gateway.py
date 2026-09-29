"""Public Behavior Interface gateway for the evaluator-separated test stack.

This process deliberately does not import OmniGibson or behavior_interface.
It exposes the existing browser and Human/Model HTTP contract by forwarding
requests to an evaluator host bound to a private loopback port.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable
from urllib.parse import urljoin

import requests
from flask import Flask, Response, jsonify, request, stream_with_context


_REQUEST_HEADER_DENYLIST = {
    "connection",
    "content-length",
    "host",
    "transfer-encoding",
}
_RESPONSE_HEADER_DENYLIST = {
    "connection",
    "content-encoding",
    "content-length",
    "transfer-encoding",
}
_STREAM_PREFIXES = ("/video/",)


def _filtered_request_headers() -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _REQUEST_HEADER_DENYLIST
    }


def _filtered_response_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    out = [
        (key, value)
        for key, value in headers
        if key.lower() not in _RESPONSE_HEADER_DENYLIST
    ]
    out.append(("X-Behavior-Interface-Layer", "evaluator-separated-test"))
    return out


def _is_stream_request(path: str) -> bool:
    if path.startswith(_STREAM_PREFIXES):
        return True
    accept = request.headers.get("Accept", "").lower()
    return "multipart/x-mixed-replace" in accept or "text/event-stream" in accept


def build_app(
    evaluator_url: str,
    *,
    connect_timeout_s: float = 5.0,
    read_timeout_s: float = 600.0,
) -> Flask:
    app = Flask(__name__)
    session = requests.Session()
    upstream = evaluator_url.rstrip("/") + "/"
    started_ts = time.time()

    def evaluator_request(path: str, *, stream: bool):
        url = urljoin(upstream, path.lstrip("/"))
        timeout = (connect_timeout_s, None if stream else read_timeout_s)
        return session.request(
            method=request.method,
            url=url,
            params=list(request.args.items(multi=True)),
            data=request.get_data(cache=False),
            headers=_filtered_request_headers(),
            allow_redirects=False,
            stream=True,
            timeout=timeout,
        )

    @app.get("/__gateway__/healthz")
    def gateway_healthz():
        try:
            response = session.get(
                urljoin(upstream, "api/state"),
                timeout=(connect_timeout_s, min(read_timeout_s, 10.0)),
            )
            payload = response.json() if response.ok else {}
            return jsonify(
                {
                    "ok": response.ok,
                    "layer": "behavior-interface-gateway",
                    "evaluator_url": upstream.rstrip("/"),
                    "evaluator_status": response.status_code,
                    "task": payload.get("task"),
                    "scene": payload.get("scene"),
                    "tick": payload.get("tick"),
                    "uptime_s": round(time.time() - started_ts, 3),
                }
            ), 200 if response.ok else 503
        except Exception as exc:
            return jsonify(
                {
                    "ok": False,
                    "layer": "behavior-interface-gateway",
                    "evaluator_url": upstream.rstrip("/"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ), 503

    @app.get("/__gateway__/architecture")
    def gateway_architecture():
        return jsonify(
            {
                "ok": True,
                "layers": [
                    "OmniGibson Evaluator Host",
                    "Behavior Interface Gateway",
                    "Human/Model Policy Client",
                ],
                "evaluator_url": upstream.rstrip("/"),
                "simulator_imported": False,
                "compatibility_mode": True,
            }
        )

    @app.route(
        "/",
        defaults={"path": ""},
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    @app.route(
        "/<path:path>",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    def proxy(path: str):
        proxy_path = "/" + path
        stream = _is_stream_request(proxy_path)
        try:
            response = evaluator_request(proxy_path, stream=stream)
        except requests.RequestException as exc:
            return jsonify(
                {
                    "ok": False,
                    "error": "evaluator host unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "evaluator_url": upstream.rstrip("/"),
                }
            ), 502

        headers = _filtered_response_headers(response.headers.items())
        if stream:
            def generate():
                try:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            yield chunk
                finally:
                    response.close()

            return Response(
                stream_with_context(generate()),
                status=response.status_code,
                headers=headers,
            )

        try:
            body = response.content
        finally:
            response.close()
        return Response(body, status=response.status_code, headers=headers)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Public gateway for the evaluator-separated BEHAVIOR Interface test."
    )
    parser.add_argument(
        "--evaluator-url",
        default=os.environ.get("BEHAVIOR_EVAL_TEST_EVALUATOR_URL", "http://127.0.0.1:18080"),
    )
    parser.add_argument("--host", default=os.environ.get("BEHAVIOR_EVAL_TEST_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("BEHAVIOR_EVAL_TEST_PORT", "15060")),
    )
    parser.add_argument("--connect-timeout-s", type=float, default=5.0)
    parser.add_argument("--read-timeout-s", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(
        args.evaluator_url,
        connect_timeout_s=args.connect_timeout_s,
        read_timeout_s=args.read_timeout_s,
    )
    print(
        "Behavior Interface Gateway "
        f"public=http://{args.host}:{args.port} evaluator={args.evaluator_url}",
        flush=True,
    )
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
