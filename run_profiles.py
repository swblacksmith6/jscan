#!/usr/bin/env python3
"""Run the job matcher for every enabled profile in profiles.yaml."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILES = BASE_DIR / "profiles.yaml"


def resolve_path(value, base=BASE_DIR):
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def load_profiles(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    defaults = data.get("defaults") or {}
    profiles = data.get("profiles") or []
    return defaults, profiles


def merged_profile(defaults, profile):
    merged = dict(defaults)
    merged.update(profile)
    return merged


def add_optional(cmd, flag, value):
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def build_match_command(python, profile):
    cmd = [
        str(python),
        str(BASE_DIR / "match_jobs.py"),
        "--resume",
        str(profile["resume_path"]),
        "--jobs",
        str(profile["jobs_path"]),
        "--html",
        str(profile["html_path"]),
        "--max-age-hours",
        str(profile.get("max_age_hours", 24)),
        "--top",
        str(profile.get("top", 10)),
        "--shortlist",
        str(profile.get("shortlist", 20)),
        "--provider",
        str(profile.get("provider", "gmail")),
    ]
    add_optional(cmd, "--model", profile.get("model"))
    add_optional(cmd, "--to", profile.get("to"))
    add_optional(cmd, "--from", profile.get("from"))
    add_optional(cmd, "--subject", profile.get("subject"))
    if profile.get("agent"):
        cmd.append("--agent")
    if profile.get("no_llm"):
        cmd.append("--no-llm")
    if profile.get("verbose"):
        cmd.append("--verbose")
    if profile.get("summary_only", True):
        cmd.append("--summary-only")
    return cmd


def run(cmd, label):
    print(f"===== {label} =====", flush=True)
    proc = subprocess.run(cmd, cwd=BASE_DIR)
    if proc.returncode != 0:
        print(f"{label} failed with exit code {proc.returncode}", flush=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Run job matches for enabled profiles.")
    ap.add_argument("--profiles", default=str(DEFAULT_PROFILES), help="path to profiles.yaml")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--skip-fetch", action="store_true", help="do not refresh jobs before matching")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = ap.parse_args()

    profile_path = resolve_path(args.profiles)
    defaults, profiles = load_profiles(profile_path)
    enabled = [merged_profile(defaults, p) for p in profiles if p.get("enabled", True)]

    if not enabled:
        print(f"No enabled profiles found in {profile_path}")
        return 1

    jobs_path = resolve_path(defaults.get("jobs", "data/jobs.json"))
    if not args.skip_fetch:
        fetch_cmd = [str(args.python), str(BASE_DIR / "fetch_jobs.py"), "--output", str(jobs_path)]
        if args.dry_run:
            print("DRY RUN fetching latest jobs:")
            print(" ".join(fetch_cmd))
        else:
            rc = run(fetch_cmd, "fetching latest jobs")
            if rc != 0:
                return rc

    failures = 0
    for profile in enabled:
        profile_id = profile.get("id") or profile.get("name") or "profile"
        resume_path = resolve_path(profile.get("resume"))
        if not resume_path or not resume_path.exists():
            print(f"Skipping {profile_id}: resume not found at {resume_path}", flush=True)
            failures += 1
            continue

        html_path = resolve_path(profile.get("html"))
        if html_path is None:
            html_dir = resolve_path(profile.get("html_dir", "reports"))
            html_path = html_dir / f"{profile_id}_matches.html"
        if not args.dry_run:
            html_path.parent.mkdir(parents=True, exist_ok=True)

        profile["resume_path"] = resume_path
        profile["jobs_path"] = resolve_path(profile.get("jobs", jobs_path))
        profile["html_path"] = html_path

        cmd = build_match_command(args.python, profile)
        if args.dry_run:
            print(f"DRY RUN matching profile {profile_id}:")
            print(" ".join(cmd))
        else:
            rc = run(cmd, f"matching profile {profile_id}")
            if rc != 0:
                failures += 1

    if failures:
        print(f"Completed with {failures} profile failure(s).", flush=True)
        return 1
    print("Completed all enabled profiles.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
