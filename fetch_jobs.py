#!/usr/bin/env python3
"""
fetch_jobs.py — daily job fetcher for Job_Fetch_Design.

Reads a list of job sources from sources.yaml, calls all of them in
parallel, normalizes each source's response into a common schema, and
merges the results into a single deduped data/jobs.json.

Designed to run once a day from cron/launchd. A source that errors out
(network issue, API down, missing key) is logged and skipped — it never
takes down the whole run.

Usage:
    python fetch_jobs.py
    python fetch_jobs.py --config sources.yaml --output data/jobs.json
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up a local .env if present; no-op otherwise
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "sources.yaml"
DEFAULT_OUTPUT = BASE_DIR / "data" / "jobs.json"
LOG_DIR = BASE_DIR / "logs"

REQUEST_TIMEOUT = 20
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2
USER_AGENT = "job-fetch-design-bot/1.0 (personal use, daily fetch)"


class SourceSkipped(Exception):
    """Raised when a source can't run (e.g. missing API key) — not an error, just skip it."""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_id(source, external_id, url):
    """Stable dedup id for a job: prefer source+external_id, fall back to a hash of the URL."""
    if external_id:
        return f"{source}:{external_id}"
    digest = hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:16]
    return f"{source}:url:{digest}"


def normalize(source, external_id, title, company, location, remote, url,
              tags=None, posted_at=None, salary=None, description=None):
    return {
        "id": make_id(source, external_id, url),
        "source": source,
        "title": (title or "").strip() or None,
        "company": (company or "").strip() or None,
        "location": location or None,
        "remote": remote,
        "url": url or None,
        "tags": tags or [],
        "posted_at": posted_at,
        "salary": salary,
        "description": description,
    }


def salary_range(lo, hi):
    if lo and hi:
        return f"{lo}-{hi}"
    return lo or hi or None


def epoch_to_iso(epoch_seconds):
    if not epoch_seconds:
        return None
    try:
        return datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc).isoformat(timespec="seconds")
    except (ValueError, TypeError, OSError):
        return None


def get_with_retry(session, url, **kwargs):
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    kwargs.setdefault("headers", {})
    kwargs["headers"].setdefault("User-Agent", USER_AGENT)
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc


def post_with_retry(session, url, **kwargs):
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    kwargs.setdefault("headers", {})
    kwargs["headers"].setdefault("User-Agent", USER_AGENT)
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(url, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc


# ---------------------------------------------------------------------------
# Per-source parsers. Each takes (source_cfg, session) and returns a list of
# normalized job dicts. Keep these defensive (.get() with fallbacks) so one
# unexpected field never crashes the whole run — that's what the per-item
# try/except in run_source() is for too.
# ---------------------------------------------------------------------------

def fetch_remoteok(cfg, session):
    resp = get_with_retry(session, cfg["base_url"], params=cfg.get("params"))
    data = resp.json()
    jobs = []
    for item in data:
        if not isinstance(item, dict) or "id" not in item:
            continue  # first element is a legal/attribution notice, not a job
        jobs.append(normalize(
            source=cfg["name"],
            external_id=str(item.get("id")),
            title=item.get("position"),
            company=item.get("company"),
            location=item.get("location") or "Remote",
            remote=True,
            url=item.get("url") or item.get("apply_url"),
            tags=item.get("tags") or [],
            posted_at=item.get("date"),
            salary=salary_range(item.get("salary_min"), item.get("salary_max")),
            description=item.get("description"),
        ))
    return jobs


def fetch_arbeitnow(cfg, session):
    resp = get_with_retry(session, cfg["base_url"], params=cfg.get("params"))
    data = resp.json()
    jobs = []
    for item in data.get("data", []):
        jobs.append(normalize(
            source=cfg["name"],
            external_id=item.get("slug"),
            title=item.get("title"),
            company=item.get("company_name"),
            location=item.get("location"),
            remote=item.get("remote"),
            url=item.get("url"),
            tags=item.get("tags") or [],
            posted_at=epoch_to_iso(item.get("created_at")),
            salary=None,
            description=item.get("description"),
        ))
    return jobs


def fetch_himalayas(cfg, session):
    resp = get_with_retry(session, cfg["base_url"], params=cfg.get("params"))
    data = resp.json()
    jobs = []
    for item in data.get("jobs", []):
        locations = item.get("locationRestrictions") or []
        location = "; ".join(locations) if locations else "Remote (no restriction)"
        jobs.append(normalize(
            source=cfg["name"],
            external_id=str(item.get("guid") or ""),
            title=item.get("title"),
            company=item.get("companyName"),
            location=location,
            remote=True,
            url=item.get("applicationLink") or item.get("guid"),
            tags=item.get("categories") or [],
            posted_at=item.get("pubDate"),
            salary=salary_range(item.get("minSalary"), item.get("maxSalary")),
            description=item.get("description") or item.get("excerpt"),
        ))
    return jobs


def fetch_remotive(cfg, session):
    resp = get_with_retry(session, cfg["base_url"], params=cfg.get("params"))
    data = resp.json()
    jobs = []
    for item in data.get("jobs", []):
        jobs.append(normalize(
            source=cfg["name"],
            external_id=str(item.get("id")),
            title=item.get("title"),
            company=item.get("company_name"),
            location=item.get("candidate_required_location"),
            remote=True,
            url=item.get("url"),
            tags=item.get("tags") or [],
            posted_at=item.get("publication_date"),
            salary=item.get("salary") or None,
            description=item.get("description"),
        ))
    return jobs


def fetch_adzuna(cfg, session):
    app_id = os.environ.get(cfg.get("api_id_env", ""))
    app_key = os.environ.get(cfg.get("api_key_env", ""))
    if not app_id or not app_key:
        raise SourceSkipped(f"missing {cfg.get('api_id_env')}/{cfg.get('api_key_env')} env vars")
    params = dict(cfg.get("params") or {})
    params["app_id"] = app_id
    params["app_key"] = app_key
    resp = get_with_retry(session, cfg["base_url"], params=params)
    data = resp.json()
    jobs = []
    for item in data.get("results", []):
        category = item.get("category") or {}
        jobs.append(normalize(
            source=cfg["name"],
            external_id=str(item.get("id")),
            title=item.get("title"),
            company=(item.get("company") or {}).get("display_name"),
            location=(item.get("location") or {}).get("display_name"),
            remote=None,
            url=item.get("redirect_url"),
            tags=[category["label"]] if category.get("label") else [],
            posted_at=item.get("created"),
            salary=salary_range(item.get("salary_min"), item.get("salary_max")),
            description=item.get("description"),
        ))
    return jobs


def fetch_findwork(cfg, session):
    api_key = os.environ.get(cfg.get("api_key_env", ""))
    if not api_key:
        raise SourceSkipped(f"missing {cfg.get('api_key_env')} env var")
    headers = {"Authorization": f"Token {api_key}"}
    resp = get_with_retry(session, cfg["base_url"], headers=headers, params=cfg.get("params"))
    data = resp.json()
    jobs = []
    for item in data.get("results", []):
        jobs.append(normalize(
            source=cfg["name"],
            external_id=str(item.get("id")),
            title=item.get("role"),
            company=item.get("company_name"),
            location=item.get("location"),
            remote=item.get("remote"),
            url=item.get("url"),
            tags=item.get("keywords") or [],
            posted_at=item.get("date_posted"),
            salary=None,
            description=item.get("text"),
        ))
    return jobs


def fetch_jooble(cfg, session):
    api_key = os.environ.get(cfg.get("api_key_env", ""))
    if not api_key:
        raise SourceSkipped(f"missing {cfg.get('api_key_env')} env var")
    url = cfg["base_url"].rstrip("/") + "/" + api_key
    headers = {"Content-Type": "application/json"}
    resp = post_with_retry(session, url, headers=headers, json=cfg.get("params") or {})
    data = resp.json()
    jobs = []
    for item in data.get("jobs", []):
        jobs.append(normalize(
            source=cfg["name"],
            external_id=str(item.get("id") or ""),
            title=item.get("title"),
            company=item.get("company"),
            location=item.get("location"),
            remote=None,
            url=item.get("link"),
            tags=[item.get("type")] if item.get("type") else [],
            posted_at=item.get("updated"),
            salary=item.get("salary") or None,
            description=item.get("snippet"),
        ))
    return jobs


def fetch_usajobs(cfg, session):
    api_key = os.environ.get(cfg.get("api_key_env", ""))
    email = os.environ.get(cfg.get("api_email_env", ""))
    if not api_key or not email:
        raise SourceSkipped(f"missing {cfg.get('api_key_env')}/{cfg.get('api_email_env')} env vars")
    headers = {
        "Host": "data.usajobs.gov",
        "User-Agent": email,
        "Authorization-Key": api_key,
    }
    resp = get_with_retry(session, cfg["base_url"], headers=headers, params=cfg.get("params"))
    data = resp.json()
    jobs = []
    items = data.get("SearchResult", {}).get("SearchResultItems", [])
    for item in items:
        d = item.get("MatchedObjectDescriptor", {})
        locations = d.get("PositionLocation") or []
        remuneration = d.get("PositionRemuneration") or [{}]
        jobs.append(normalize(
            source=cfg["name"],
            external_id=item.get("MatchedObjectId"),
            title=d.get("PositionTitle"),
            company=d.get("OrganizationName"),
            location="; ".join(l.get("LocationName", "") for l in locations) or None,
            remote=None,
            url=d.get("PositionURI"),
            tags=[],
            posted_at=d.get("PublicationStartDate"),
            salary=salary_range(
                remuneration[0].get("MinimumRange") if remuneration else None,
                remuneration[0].get("MaximumRange") if remuneration else None,
            ),
            description=(d.get("UserArea", {}).get("Details", {}) or {}).get("JobSummary"),
        ))
    return jobs


PARSERS = {
    "remoteok": fetch_remoteok,
    "arbeitnow": fetch_arbeitnow,
    "himalayas": fetch_himalayas,
    "remotive": fetch_remotive,
    "adzuna": fetch_adzuna,
    "findwork": fetch_findwork,
    "jooble": fetch_jooble,
    "usajobs": fetch_usajobs,
}


def run_source(cfg):
    """Runs one source end to end, returns (name, jobs, status, detail)."""
    name = cfg["name"]
    parser = PARSERS.get(cfg["type"])
    if parser is None:
        return name, [], "error", f"no parser registered for type '{cfg['type']}'"

    session = requests.Session()
    try:
        jobs = parser(cfg, session)
        valid_jobs = []
        for job in jobs:
            try:
                # cheap sanity check per item so one malformed record doesn't
                # get silently written with junk data
                if job.get("title") and job.get("url"):
                    valid_jobs.append(job)
            except Exception:
                continue
        return name, valid_jobs, "ok", f"{len(valid_jobs)} jobs"
    except SourceSkipped as exc:
        return name, [], "skipped", str(exc)
    except requests.RequestException as exc:
        return name, [], "error", f"request failed: {exc}"
    except Exception as exc:  # noqa: BLE001 - a source blowing up must not kill the run
        return name, [], "error", f"unexpected error: {exc}"
    finally:
        session.close()


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    sources = cfg.get("sources", [])
    return [s for s in sources if s.get("enabled", True)]


def load_existing(path):
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("jobs", {})
    except (json.JSONDecodeError, OSError):
        logging.warning("could not read existing %s, starting fresh", path)
        return {}


def merge_jobs(existing, fetched, run_time):
    """Merge freshly fetched jobs into the existing store, deduped by id.
    New jobs get first_seen == last_seen == now; jobs seen before keep their
    original first_seen and get last_seen refreshed plus updated fields."""
    new_count = 0
    updated_count = 0
    for job in fetched:
        job_id = job["id"]
        if job_id in existing:
            job["first_seen"] = existing[job_id].get("first_seen", run_time)
            updated_count += 1
        else:
            job["first_seen"] = run_time
            new_count += 1
        job["last_seen"] = run_time
        existing[job_id] = job
    return existing, new_count, updated_count


def write_output(path, jobs_by_id, run_time):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_run": run_time,
        "job_count": len(jobs_by_id),
        "jobs": jobs_by_id,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)  # atomic on POSIX — never leaves jobs.json half-written


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"fetch_{datetime.now().strftime('%Y-%m-%d')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch tech job listings from configured sources.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to sources.yaml")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="path to jobs.json")
    parser.add_argument("--workers", type=int, default=8, help="max parallel source fetches")
    args = parser.parse_args()

    setup_logging()
    run_time = now_iso()
    output_path = Path(args.output)

    sources = load_config(args.config)
    if not sources:
        logging.error("no enabled sources found in %s", args.config)
        sys.exit(1)

    logging.info("starting fetch: %d enabled sources", len(sources))

    all_jobs = []
    results_summary = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_source, cfg): cfg["name"] for cfg in sources}
        for future in as_completed(futures):
            name, jobs, status, detail = future.result()
            results_summary.append((name, status, detail))
            if status == "ok":
                logging.info("[%s] ok — %s", name, detail)
                all_jobs.extend(jobs)
            elif status == "skipped":
                logging.info("[%s] skipped — %s", name, detail)
            else:
                logging.error("[%s] error — %s", name, detail)

    existing = load_existing(output_path)
    merged, new_count, updated_count = merge_jobs(existing, all_jobs, run_time)
    write_output(output_path, merged, run_time)

    ok = sum(1 for _, s, _ in results_summary if s == "ok")
    skipped = sum(1 for _, s, _ in results_summary if s == "skipped")
    errored = sum(1 for _, s, _ in results_summary if s == "error")
    logging.info(
        "done: %d sources ok, %d skipped, %d errored | %d new jobs, %d updated | %d total in %s",
        ok, skipped, errored, new_count, updated_count, len(merged), output_path,
    )


if __name__ == "__main__":
    main()
