from __future__ import annotations

import atexit
import os
from typing import Mapping, Optional

from flask import Flask, jsonify, render_template, request, send_file

from ..io.fs_utils import atomic_write_text
from ..runtime.startup import DEFAULT_HOST, DEFAULT_PORT
from ..services.translation_runner import TranslationOptions, parse_url_list_text
from .jobs import TranslationJobManager


def create_app(
    *,
    job_base_dir: Optional[str] = None,
    write_text=atomic_write_text,
    max_workers: int = 2,
) -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
    )
    manager = TranslationJobManager(
        base_dir=job_base_dir or os.path.join(os.getcwd(), ".tmp", "web-jobs"),
        write_text=write_text,
        max_workers=max_workers,
    )
    app.extensions["translation_job_manager"] = manager

    @atexit.register
    def _shutdown_jobs() -> None:
        manager.close()

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            defaults={
                "output_format": "readable",
                "concurrency": 2,
                "timeout": 30,
                "max_chunk_chars": 5000,
                "snapdown_to_mermaid": True,
                "host": DEFAULT_HOST,
                "port": DEFAULT_PORT,
            },
        )

    @app.post("/api/jobs/url")
    def create_url_job():
        payload = request.get_json(silent=True) or request.form
        url = _get_str_value(payload, "url")
        if not url:
            return _json_error("url is required", status=400)
        try:
            job = manager.create_single_job(url, _parse_translation_options(payload))
        except ValueError as exc:
            return _json_error(str(exc), status=400)
        return jsonify(job), 202

    @app.post("/api/jobs/url-file")
    def create_url_file_job():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return _json_error("url.txt file is required", status=400)
        if not upload.filename.lower().endswith(".txt"):
            return _json_error("only .txt files are supported", status=400)
        try:
            content = upload.read().decode("utf-8-sig")
        except UnicodeDecodeError:
            return _json_error("url.txt must be UTF-8 encoded", status=400)
        try:
            urls = parse_url_list_text(content, source_label=upload.filename)
            job = manager.create_batch_job(
                urls,
                _parse_translation_options(request.form),
                source_name=upload.filename,
            )
        except ValueError as exc:
            return _json_error(str(exc), status=400)
        return jsonify(job), 202

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        payload = manager.get_job(job_id)
        if payload is None:
            return _json_error("job not found", status=404)
        return jsonify(payload)

    @app.get("/api/jobs/<job_id>/download")
    def download_job(job_id: str):
        download = manager.get_download(job_id)
        if download is None:
            return _json_error("job download is not ready", status=404)
        path, download_name = download
        return send_file(path, as_attachment=True, download_name=download_name)

    return app


def serve_app(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    job_base_dir: Optional[str] = None,
    max_workers: int = 2,
) -> None:
    app = create_app(job_base_dir=job_base_dir, max_workers=max_workers)
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


def _parse_translation_options(payload: Mapping[str, object]) -> TranslationOptions:
    output_format = _get_str_value(payload, "output_format", default="readable")
    if output_format not in {"readable", "analysis"}:
        raise ValueError("output_format must be 'readable' or 'analysis'")
    concurrency = _get_int_value(payload, "concurrency", default=2, minimum=1, maximum=8)
    timeout = _get_float_value(payload, "timeout", default=30.0, minimum=1.0, maximum=300.0)
    max_chunk_chars = _get_int_value(
        payload,
        "max_chunk_chars",
        default=5000,
        minimum=500,
        maximum=20000,
    )
    snapdown_to_mermaid = _get_bool_value(
        payload,
        "snapdown_to_mermaid",
        default=True,
    )
    return TranslationOptions(
        timeout=timeout,
        max_chunk_chars=max_chunk_chars,
        concurrency=concurrency,
        snapdown_to_mermaid=snapdown_to_mermaid,
        output_format=output_format,
    )


def _get_str_value(
    payload: Mapping[str, object],
    key: str,
    *,
    default: str = "",
) -> str:
    raw = payload.get(key, default)
    return str(raw or "").strip()


def _get_int_value(
    payload: Mapping[str, object],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = payload.get(key, default)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _get_float_value(
    payload: Mapping[str, object],
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = payload.get(key, default)
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _get_bool_value(
    payload: Mapping[str, object],
    key: str,
    *,
    default: bool,
) -> bool:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean value")


def _json_error(message: str, *, status: int):
    return jsonify({"status": "error", "message": message}), status
