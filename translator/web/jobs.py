from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..services.translation_runner import (
    TranslationOptions,
    build_batch_out_path,
    build_single_out_path,
    normalize_urls,
    require_out_dir,
    translate_url_to_path,
)


def _utc_iso(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


@dataclass
class JobArtifact:
    name: str
    path: str
    size: int

    def to_payload(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "size": self.size,
        }


@dataclass
class JobItem:
    url: str
    status: str = "queued"
    output_name: Optional[str] = None
    error: Optional[str] = None

    def to_payload(self) -> Dict[str, object]:
        return {
            "url": self.url,
            "status": self.status,
            "output_name": self.output_name,
            "error": self.error,
        }


@dataclass
class JobRecord:
    job_id: str
    mode: str
    source_name: str
    status: str
    created_at: float
    updated_at: float
    total_count: int
    completed_count: int = 0
    successful_count: int = 0
    failed_count: int = 0
    error: Optional[str] = None
    download_name: Optional[str] = None
    download_path: Optional[str] = None
    files: List[JobArtifact] = field(default_factory=list)
    items: List[JobItem] = field(default_factory=list)

    def to_payload(self) -> Dict[str, object]:
        return {
            "job_id": self.job_id,
            "mode": self.mode,
            "source_name": self.source_name,
            "status": self.status,
            "created_at": _utc_iso(self.created_at),
            "updated_at": _utc_iso(self.updated_at),
            "total_count": self.total_count,
            "completed_count": self.completed_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "has_failures": self.failed_count > 0,
            "error": self.error,
            "download_name": self.download_name,
            "files": [artifact.to_payload() for artifact in self.files],
            "items": [item.to_payload() for item in self.items],
        }


class TranslationJobManager:
    def __init__(
        self,
        *,
        base_dir: str,
        write_text,
        ttl_seconds: int = 24 * 60 * 60,
        max_workers: int = 2,
    ) -> None:
        self._base_dir = Path(require_out_dir(base_dir))
        self._write_text = write_text
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="translator-web",
        )
        self.cleanup_expired()

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def create_single_job(self, url: str, options: TranslationOptions) -> Dict[str, object]:
        normalized_url = normalize_urls([url])[0]
        record = self._create_job(
            mode="single",
            source_name=normalized_url,
            total_count=1,
            items=[JobItem(url=normalized_url)],
        )
        self._executor.submit(self._run_single_job, record.job_id, normalized_url, options)
        return self.get_job(record.job_id) or record.to_payload()

    def create_batch_job(
        self,
        urls: Sequence[str],
        options: TranslationOptions,
        *,
        source_name: str,
    ) -> Dict[str, object]:
        normalized_urls = normalize_urls(urls)
        record = self._create_job(
            mode="batch",
            source_name=source_name,
            total_count=len(normalized_urls),
            items=[JobItem(url=url) for url in normalized_urls],
        )
        self._executor.submit(
            self._run_batch_job,
            record.job_id,
            normalized_urls,
            options,
        )
        return self.get_job(record.job_id) or record.to_payload()

    def get_job(self, job_id: str) -> Optional[Dict[str, object]]:
        self.cleanup_expired()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            payload = record.to_payload()
            if record.download_path:
                payload["download_url"] = f"/api/jobs/{job_id}/download"
            else:
                payload["download_url"] = None
            return payload

    def get_download(self, job_id: str) -> Optional[Tuple[str, str]]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or not record.download_path or not record.download_name:
                return None
            return record.download_path, record.download_name

    def cleanup_expired(self) -> None:
        now = time.time()
        expired_job_ids: List[str] = []
        active_job_ids: set[str] = set()

        with self._lock:
            for job_id, record in self._jobs.items():
                if record.status in {"queued", "running"}:
                    active_job_ids.add(job_id)
                    continue
                if now - record.updated_at > self._ttl_seconds:
                    expired_job_ids.append(job_id)
                else:
                    active_job_ids.add(job_id)
            for job_id in expired_job_ids:
                self._jobs.pop(job_id, None)

        for job_id in expired_job_ids:
            self._remove_job_dir(job_id)

        for child in self._base_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name in active_job_ids:
                continue
            try:
                age_seconds = now - child.stat().st_mtime
            except OSError:
                continue
            if age_seconds > self._ttl_seconds:
                shutil.rmtree(child, ignore_errors=True)

    def _create_job(
        self,
        *,
        mode: str,
        source_name: str,
        total_count: int,
        items: List[JobItem],
    ) -> JobRecord:
        now = time.time()
        record = JobRecord(
            job_id=uuid.uuid4().hex[:12],
            mode=mode,
            source_name=source_name,
            status="queued",
            created_at=now,
            updated_at=now,
            total_count=total_count,
            items=items,
        )
        job_dir = self._job_dir(record.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[record.job_id] = record
        return record

    def _run_single_job(
        self,
        job_id: str,
        url: str,
        options: TranslationOptions,
    ) -> None:
        self._update_job_status(job_id, "running")
        out_path = build_single_out_path(str(self._job_dir(job_id)), url)
        self._mark_item_running(job_id, 0, os.path.basename(out_path))
        try:
            _ = translate_url_to_path(
                url=url,
                out_path=out_path,
                options=options,
                write_text=self._write_text,
            )
        except Exception as exc:
            self._finish_single_failure(job_id, str(exc))
            return

        artifact = self._artifact_from_path(out_path)
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "success"
            record.updated_at = time.time()
            record.completed_count = 1
            record.successful_count = 1
            record.files = [artifact]
            record.download_name = artifact.name
            record.download_path = artifact.path
            record.items[0].status = "success"
            record.items[0].error = None

    def _run_batch_job(
        self,
        job_id: str,
        urls: Sequence[str],
        options: TranslationOptions,
    ) -> None:
        self._update_job_status(job_id, "running")
        job_dir = self._job_dir(job_id)
        used_names: set[str] = set()
        artifacts: List[JobArtifact] = []

        for index, url in enumerate(urls, start=1):
            out_path = build_batch_out_path(str(job_dir), url, index, used_names)
            self._mark_item_running(job_id, index - 1, os.path.basename(out_path))
            try:
                _ = translate_url_to_path(
                    url=url,
                    out_path=out_path,
                    options=options,
                    write_text=self._write_text,
                )
                artifact = self._artifact_from_path(out_path)
                artifacts.append(artifact)
                self._finish_batch_item(job_id, index - 1, success=True, error=None)
            except Exception as exc:
                self._finish_batch_item(job_id, index - 1, success=False, error=str(exc))

        if not artifacts:
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                record.status = "failed"
                record.updated_at = time.time()
                record.error = "all URLs failed"
            return

        zip_name = f"translated-markdown-{job_id}.zip"
        zip_path = job_dir / zip_name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for artifact in artifacts:
                archive.write(artifact.path, arcname=artifact.name)

        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "success"
            record.updated_at = time.time()
            record.files = artifacts
            record.download_name = zip_name
            record.download_path = str(zip_path)
            record.error = None

    def _finish_single_failure(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = "failed"
            record.updated_at = time.time()
            record.completed_count = 1
            record.failed_count = 1
            record.error = error
            record.items[0].status = "failed"
            record.items[0].error = error

    def _finish_batch_item(
        self,
        job_id: str,
        item_index: int,
        *,
        success: bool,
        error: Optional[str],
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            item = record.items[item_index]
            item.status = "success" if success else "failed"
            item.error = error
            record.completed_count += 1
            if success:
                record.successful_count += 1
            else:
                record.failed_count += 1
            record.updated_at = time.time()

    def _mark_item_running(
        self,
        job_id: str,
        item_index: int,
        output_name: str,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            item = record.items[item_index]
            item.status = "running"
            item.output_name = output_name
            item.error = None
            record.updated_at = time.time()

    def _update_job_status(self, job_id: str, status: str) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = status
            record.updated_at = time.time()

    def _artifact_from_path(self, path: str) -> JobArtifact:
        stat = os.stat(path)
        return JobArtifact(name=os.path.basename(path), path=path, size=stat.st_size)

    def _job_dir(self, job_id: str) -> Path:
        return self._base_dir / job_id

    def _remove_job_dir(self, job_id: str) -> None:
        shutil.rmtree(self._job_dir(job_id), ignore_errors=True)
