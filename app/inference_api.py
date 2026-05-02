#!/usr/bin/env python3
"""Flask inference API: `POST /analyze` on port 5001 (GPU server).

Development: use `--mock` for a deterministic JSON response without weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from flask import Flask, Response, jsonify, request

from app.api_client import mock_analysis_result


def create_app(*, mock: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["MOCK"] = mock
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

    @app.route("/health", methods=["GET"])
    def health() -> Response:
        return Response("ok\n", mimetype="text/plain")

    @app.route("/analyze", methods=["POST"])
    def analyze() -> tuple[Any, int]:
        if app.config["MOCK"]:
            return jsonify(mock_analysis_result()), 200

        try:
            from src.pipeline import Pipeline
        except ImportError as e:  # pragma: no cover - env-specific
            return jsonify({"error": f"server_unavailable: {e}"}), 503

        raw = request.get_data()
        if not raw:
            return jsonify({"error": "empty_body"}), 400

        tmp_dir = Path(__import__("tempfile").mkdtemp(prefix="df_infer_"))
        tmp_path = tmp_dir / "upload.mp4"
        tmp_path.write_bytes(raw)
        out: dict[str, Any] | None = None
        err: tuple[Any, int] | None = None
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = Pipeline(device=device)
            pipe.load_models()
            out = pipe.run_on_video(tmp_path)

        except FileNotFoundError as e:
            err = (jsonify({"error": str(e)}), 503)
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc()
            err = (jsonify({"error": str(e)}), 500)
        finally:
            import shutil
            import torch

            shutil.rmtree(tmp_dir, ignore_errors=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if err is not None:
            return err
        if out is None:
            return jsonify({"error": "inference_failed"}), 500
        return jsonify(out), 200

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Flask inference API for deepfake pipeline.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--mock", action="store_true", help="Return canned JSON; no models required.")
    args = p.parse_args()
    app = create_app(mock=args.mock)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
