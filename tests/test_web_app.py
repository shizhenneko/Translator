from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from translator.web.app import create_app


@pytest.fixture
def web_app(tmp_path: Path):
    app = create_app(job_base_dir=str(tmp_path / "jobs"), max_workers=1)
    yield app
    app.extensions["translation_job_manager"].close()


def test_index_page_renders(web_app) -> None:
    client = web_app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert "url.txt" in response.get_data(as_text=True)
    assert "网页转中文 Markdown" in response.get_data(as_text=True)


def test_create_url_job_requires_url(web_app) -> None:
    client = web_app.test_client()

    response = client.post("/api/jobs/url", json={})

    assert response.status_code == 400
    assert response.get_json()["message"] == "url is required"


def test_upload_job_rejects_non_txt_file(web_app) -> None:
    client = web_app.test_client()

    response = client.post(
        "/api/jobs/url-file",
        data={"file": (io.BytesIO(b"https://example.com"), "urls.csv")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["message"] == "only .txt files are supported"


def test_single_url_job_can_be_downloaded(
    monkeypatch: pytest.MonkeyPatch,
    web_app,
) -> None:
    def fake_translate_document(**kwargs: object) -> str:
        source_value = str(kwargs["source_value"])
        out_path = str(kwargs["out_path"])
        write_text = kwargs["write_text"]
        write_text(out_path, f"# {source_value}\n")
        return f"# {source_value}\n"

    monkeypatch.setattr("translator.pipeline.translate_document", fake_translate_document)
    client = web_app.test_client()

    response = client.post(
        "/api/jobs/url",
        json={"url": "https://example.com/article", "concurrency": 1},
    )

    assert response.status_code == 202
    payload = response.get_json()
    job_id = payload["job_id"]

    deadline = time.time() + 5
    while time.time() < deadline:
        polled = client.get(f"/api/jobs/{job_id}")
        job = polled.get_json()
        if job["status"] in {"success", "failed"}:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("job did not finish in time")

    assert job["status"] == "success"
    assert job["download_url"] == f"/api/jobs/{job_id}/download"

    download = client.get(job["download_url"])
    assert download.status_code == 200
    assert b"https://example.com/article" in download.data
