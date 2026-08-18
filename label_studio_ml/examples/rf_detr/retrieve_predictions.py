#!/usr/bin/env python3
"""Retrieve RF-DETR predictions for every task via the Label Studio API.

Do not use Data Manager → Retrieve Predictions for hundreds of images.
That action is one HTTP request through nginx (90s) and uWSGI harakiri (91s).
Label Studio only saves predictions after the whole queryset finishes, so a
timeout drops all work.

This script predicts and saves one task at a time, so it can be re-run safely.

  export LABEL_STUDIO_URL=http://10.242.200.200:18080
  export LABEL_STUDIO_API_KEY=<token from Account & Settings → Access Token>
  export ML_BACKEND_URL=http://10.242.200.200:9095
  export PROJECT_ID=1
  python3 retrieve_predictions.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, Optional

LS_URL = os.environ.get("LABEL_STUDIO_URL", "http://10.242.200.200:18080").rstrip("/")
API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "")
ML_URL = os.environ.get("ML_BACKEND_URL", "http://10.242.200.200:9095").rstrip("/")
PROJECT_ID = int(os.environ.get("PROJECT_ID", "1"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "RFDETRBackend-v0.0.1")
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "100"))


def _request(
    url: str,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    timeout: int = 120,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc


def _iter_tasks(token: str) -> Iterable[Dict[str, Any]]:
    page = 1
    while True:
        qs = urllib.parse.urlencode(
            {"project": PROJECT_ID, "page": page, "page_size": PAGE_SIZE}
        )
        data = _request(f"{LS_URL}/api/tasks?{qs}", token=token)
        if isinstance(data, list):
            tasks = data
            next_url = None
        else:
            tasks = data.get("tasks") or data.get("results") or []
            next_url = data.get("next")
        for task in tasks:
            yield task
        if next_url:
            page += 1
            continue
        if len(tasks) < PAGE_SIZE:
            break
        page += 1


def _predicted_task_ids(token: str) -> set:
    ids = set()
    page = 1
    while True:
        qs = urllib.parse.urlencode(
            {"project": PROJECT_ID, "page": page, "page_size": PAGE_SIZE}
        )
        data = _request(f"{LS_URL}/api/predictions?{qs}", token=token)
        if isinstance(data, list):
            preds = data
        else:
            preds = data.get("results") or []
        for pred in preds:
            if pred.get("model_version") == MODEL_VERSION:
                ids.add(pred.get("task"))
        if isinstance(data, list) or not data.get("next"):
            if len(preds) < PAGE_SIZE:
                break
        page += 1
    return ids


def main() -> int:
    if not API_KEY:
        print(
            "Set LABEL_STUDIO_API_KEY to your Label Studio access token "
            "(Account & Settings → Access Token).",
            file=sys.stderr,
        )
        return 1

    project = _request(f"{LS_URL}/api/projects/{PROJECT_ID}/", token=API_KEY)
    label_config = project.get("label_config") or ""
    print(f"Project {PROJECT_ID}: {project.get('title')}")
    print(f"ML backend: {ML_URL}")
    print(f"Model version: {MODEL_VERSION}")

    tasks = list(_iter_tasks(API_KEY))
    already = _predicted_task_ids(API_KEY)
    print(f"Loaded {len(tasks)} tasks, {len(already)} already have {MODEL_VERSION}")

    ok = skipped = failed = 0
    for i, task in enumerate(tasks, 1):
        task_id = task["id"]
        if task_id in already:
            skipped += 1
            print(f"[{i}/{len(tasks)}] task {task_id}: skip (already predicted)")
            continue

        payload = {
            "tasks": [task],
            "project": str(PROJECT_ID),
            "label_config": label_config,
        }
        try:
            ml = _request(f"{ML_URL}/predict", method="POST", payload=payload, timeout=180)
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(tasks)}] task {task_id}: ML error: {exc}")
            continue

        results = (ml or {}).get("results") or []
        if not results:
            failed += 1
            print(f"[{i}/{len(tasks)}] task {task_id}: empty ML response")
            continue

        pred = results[0]
        body = {
            "task": task_id,
            "result": pred.get("result") or [],
            "score": pred.get("score"),
            "model_version": pred.get("model_version") or MODEL_VERSION,
        }
        try:
            _request(f"{LS_URL}/api/predictions", method="POST", payload=body, token=API_KEY)
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(tasks)}] task {task_id}: save error: {exc}")
            continue

        n_boxes = len(body["result"])
        ok += 1
        print(f"[{i}/{len(tasks)}] task {task_id}: saved {n_boxes} box(es)")

    print(f"Done. saved={ok} skipped={skipped} failed={failed} total={len(tasks)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
