#!/usr/bin/env python3
"""
match_jobs.py — match a resume against fetched jobs with a local Ollama model.

Default (fast) mode:
    1. Pre-rank all jobs in data/jobs.json against the resume using an
       IDF-weighted keyword overlap (pure Python, no model call). This turns
       hundreds of postings into a small, high-signal shortlist instantly.
    2. Score the shortlist in a SINGLE LLM call: the model returns, per job,
       only a job_id + fit score + one-line reason. Titles, companies, and
       URLs are then joined back from the real job records, so the model
       cannot hallucinate a job or a link.
    3. Print the ranked matches and write an email-ready matches.html.

Agent mode (--agent):
    Drives the same jobs as a LangChain "deep agent" (deepagents) that decides
    its own searches. Slower, but fully agentic. Kept for reference.

Usage:
    python match_jobs.py                      # fast mode, top 10, writes matches.html
    python match_jobs.py --top 15 --shortlist 25
    python match_jobs.py --html out.html --open
    python match_jobs.py --agent              # original deep-agent mode
"""

import argparse
import html as html_lib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESUME = BASE_DIR / "input_resume.md"
DEFAULT_JOBS = BASE_DIR / "data" / "jobs.json"
DEFAULT_HTML = BASE_DIR / "matches.html"

# Load secrets (RESEND_API_KEY, etc.) from the project's .env, same as
# fetch_jobs.py. Anchored to this file's directory so it works from any cwd.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

MODEL = "nemotron-3.5-lightning:30b-mlx"
NUM_CTX = 32768

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9\+#]{3,}")

# Generic words that carry no job-matching signal (IDF handles most common
# words too, but these show up in almost every resume/posting).
STOPWORDS = {
    "and", "or", "the", "for", "with", "role", "roles", "job", "jobs", "the",
    "experience", "senior", "junior", "lead", "team", "teams", "years", "year",
    "management", "assessment", "review", "using", "based", "strong", "work",
    "working", "across", "including", "within", "new", "support", "business",
    "process", "processes", "company", "time", "skills", "tools", "data",
    "will", "you", "your", "our", "are", "who", "this", "that", "from", "have",
    "been", "was", "were", "has", "had", "not", "but", "all", "any", "can",
    "led", "drove", "managed", "performed", "ability", "responsibilities",
    "requirements", "such", "into", "per", "via", "etc", "including", "level",
}


def strip_html(text):
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = (
        text.replace("&amp;", "&").replace("&nbsp;", " ").replace("&lt;", "<")
        .replace("&gt;", ">").replace("&#39;", "'").replace("&quot;", '"')
    )
    return _WS_RE.sub(" ", text).strip()


def tokenize(text):
    """Lowercase content words, stopwords removed."""
    return [w for w in _WORD_RE.findall((text or "").lower()) if w not in STOPWORDS]


def load_jobs(path):
    data = json.loads(Path(path).read_text())
    jobs = list(data.get("jobs", {}).values())
    for j in jobs:
        title = j.get("title") or ""
        tags = " ".join(j.get("tags") or [])
        desc = strip_html(j.get("description") or "")
        j["_desc_clean"] = desc
        j["_title_c"] = Counter(tokenize(title))
        j["_tags_c"] = Counter(tokenize(tags))
        j["_body_c"] = Counter(tokenize(desc))
        # term-frequency vector with fields weighted: title >> tags > body
        combined = Counter()
        for w, c in j["_title_c"].items():
            combined[w] += c * 4
        for w, c in j["_tags_c"].items():
            combined[w] += c * 2
        for w, c in j["_body_c"].items():
            combined[w] += c
        j["_combined_c"] = combined
    return jobs


def compute_idf(jobs):
    """Inverse document frequency per token across the job corpus, so
    distinctive skills (burp, stride, sbom, kubernetes) outweigh common words."""
    n = len(jobs) or 1
    df = {}
    for j in jobs:
        for w in j["_combined_c"]:
            df[w] = df.get(w, 0) + 1
    return {w: math.log(n / (1 + c)) + 1.0 for w, c in df.items()}


def _tfidf_vec(counter, idf):
    return {w: c * idf.get(w, 0.0) for w, c in counter.items() if idf.get(w, 0.0)}


def _cosine(a, b):
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(w, 0.0) for w, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (na * nb)


# --------------------------------------------------------------------------
# Recency filtering (--max-age-hours)
# --------------------------------------------------------------------------

def parse_dt(s):
    """Parse an ISO-8601 string (or epoch seconds) to an aware UTC datetime."""
    if not s:
        return None
    s = str(s).strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    if s.isdigit():
        try:
            return datetime.fromtimestamp(int(s), tz=timezone.utc)
        except (ValueError, OSError):
            return None
    return None


def job_recency(job):
    """Freshest available timestamp for a job: the newest of posted_at,
    first_seen, last_seen (first_seen/last_seen are always set by the fetcher)."""
    cands = [parse_dt(job.get(k)) for k in ("posted_at", "first_seen", "last_seen")]
    cands = [c for c in cands if c]
    return max(cands) if cands else None


def filter_recent(jobs, max_age_hours, now=None):
    """Keep only jobs whose freshest timestamp is within max_age_hours. Jobs
    with no parseable date are dropped. A falsy/zero window is a no-op."""
    if not max_age_hours or max_age_hours <= 0:
        return jobs
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)
    kept = []
    for j in jobs:
        dt = job_recency(j)
        if dt and dt >= cutoff:
            kept.append(j)
    return kept


def extract_candidate_name(resume_text):
    for line in resume_text.splitlines():
        s = line.strip().strip("*").strip()
        if s:
            return s
    return "Candidate"


def prerank(jobs, resume_text, idf, shortlist):
    """Rank jobs by TF-IDF cosine similarity to the resume. Cosine length-
    normalizes, so a verbose posting can't win just by being long, and shared
    distinctive skills (burp, stride, sbom, kubernetes) drive the score."""
    resume_vec = _tfidf_vec(Counter(tokenize(resume_text)), idf)
    scored = []
    for j in jobs:
        sim = _cosine(resume_vec, _tfidf_vec(j["_combined_c"], idf))
        if sim > 0:
            scored.append((sim, j))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [j for _, j in scored[:shortlist]]


# --------------------------------------------------------------------------
# One-pass LLM scoring
# --------------------------------------------------------------------------

class Match(BaseModel):
    job_id: str = Field(description="The job_id, copied exactly from the shortlist")
    fit_score: float = Field(description="How well the job fits the candidate, 0-10")
    reason: str = Field(description="One sentence on why it fits (or doesn't)")


class Ranking(BaseModel):
    matches: list[Match] = Field(description="Jobs ranked best-fit first")


SCORE_SYSTEM = (
    "You are a strict technical recruiter scoring how well each job fits a "
    "candidate. Judge the JOB'S CORE FUNCTION and seniority against the "
    "candidate's actual target role, NOT shared keywords or tools.\n\n"
    "A shared tool or buzzword (Kubernetes, AWS, AI, Claude, Python, 'security' "
    "appearing once) is NOT a match when the job's actual function is a "
    "different discipline (marketing, sales, content, product management, "
    "recruiting, generic software engineering). Penalize keyword-only overlap "
    "hard.\n\n"
    "Scoring rubric (0-10):\n"
    "  8-10: same role family AND seniority as the candidate's target "
    "(e.g. application/product security engineer, security architect, AppSec).\n"
    "  5-7 : adjacent role that genuinely uses the candidate's core skills "
    "day-to-day (e.g. DevSecOps, cloud/platform security, security-focused SRE).\n"
    "  2-4 : different job function that only shares tools or buzzwords.\n"
    "  0-1 : unrelated (sales, marketing, content, non-security PM, HR).\n\n"
    "Be conservative: when unsure, score lower. It is fine to return fewer than "
    "requested if only a few are genuinely relevant; do not pad the list with "
    "weak keyword matches. Only use jobs from the shortlist, copy each job_id "
    "exactly, rank best-fit first, and give each a fit_score and a one-sentence "
    "reason naming the core-function match or mismatch."
)


def shortlist_block(jobs):
    lines = []
    for j in jobs:
        desc = j["_desc_clean"]
        if len(desc) > 300:
            desc = desc[:300] + "..."
        lines.append(
            f"- job_id: {j['id']}\n"
            f"  title: {j.get('title')}\n"
            f"  company: {j.get('company') or 'n/a'}\n"
            f"  location: {j.get('location') or 'n/a'} | remote={bool(j.get('remote'))}\n"
            f"  tags: {', '.join(j.get('tags') or []) or 'none'}\n"
            f"  description: {desc or 'n/a'}"
        )
    return "\n".join(lines)


def _extract_json(text):
    """Best-effort JSON out of a possibly-noisy model reply (never raises)."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, k = text.find(open_c), text.rfind(close_c)
        if 0 <= i < k:
            try:
                return json.loads(text[i:k + 1])
            except Exception:
                continue
    return None


def score_matches(model, resume_text, shortlist, top, verbose=False):
    """Single schema-constrained LLM call. Returns [] on any failure so the
    caller can fall back to the (already strong) cosine ordering."""
    prompt = (
        f"CANDIDATE RESUME:\n---\n{resume_text}\n---\n\n"
        f"SHORTLIST OF {len(shortlist)} CANDIDATE JOBS:\n{shortlist_block(shortlist)}\n\n"
        f"Return the best {top} jobs for this candidate, ranked best-fit first, "
        f'as JSON: {{"matches":[{{"job_id":"...","fit_score":0-10,"reason":"..."}}]}}. '
        f"Copy each job_id exactly from the shortlist."
    )
    messages = [("system", SCORE_SYSTEM), ("human", prompt)]
    if verbose:
        approx = len(SCORE_SYSTEM) + len(prompt)
        print(f"  [llm] request: {len(shortlist)} jobs, ~{approx} chars (~{approx // 4} tokens), model={getattr(model, 'model', '?')}")
    t0 = time.time()
    try:
        # Ollama constrained decoding to the Ranking schema -> valid JSON.
        resp = model.bind(format=Ranking.model_json_schema()).invoke(messages)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        if verbose:
            preview = " ".join(text.split())[:280]
            print(f"  [llm] reply in {time.time() - t0:.1f}s, {len(text)} chars: {preview}")
        data = _extract_json(text)
        if isinstance(data, dict):
            matches = data.get("matches") or []
        elif isinstance(data, list):
            matches = data
        else:
            matches = []
    except Exception as exc:
        print(f"  [llm] scoring failed after {time.time() - t0:.1f}s: {exc}; falling back to keyword ranking")
        return []
    if verbose:
        print(f"  [llm] parsed {len(matches)} matches from reply")
    return matches[:top]


def join_matches(matches, by_id, by_suffix):
    """Attach real job records to model output; drop unrecognized ids and any
    job the model listed more than once (keep its first/highest-ranked spot)."""
    out, seen = [], set()
    for m in matches:
        jid = str(m.get("job_id", "")).strip()
        job = by_id.get(jid) or by_suffix.get(re.split(r"[:/]", jid)[-1])
        if not job or job["id"] in seen:
            continue
        seen.add(job["id"])
        out.append({
            "job": job,
            "fit_score": m.get("fit_score"),
            "reason": (m.get("reason") or "").strip(),
        })
    return out


# --------------------------------------------------------------------------
# HTML output (email-friendly: inline styles, simple table)
# --------------------------------------------------------------------------

def esc(s):
    return html_lib.escape(str(s if s is not None else ""))


def build_html(matches, candidate_name, out_path):
    rows = []
    for i, m in enumerate(matches, 1):
        j = m["job"]
        title = esc(j.get("title") or "Untitled role")
        url = j.get("url") or "#"
        company = esc(j.get("company") or "n/a")
        loc = esc(j.get("location") or ("Remote" if j.get("remote") else "n/a"))
        remote = "Remote" if j.get("remote") else ""
        loc_disp = f"{loc}{' · Remote' if remote and remote.lower() not in loc.lower() else ''}"
        salary = esc(j.get("salary") or "")
        score = m.get("fit_score")
        score_disp = f"{score:g}" if isinstance(score, (int, float)) else esc(score)
        reason = esc(m.get("reason") or "")
        rows.append(f"""
      <tr>
        <td style="padding:12px 10px;border-bottom:1px solid #eee;vertical-align:top;color:#888;font-weight:700;">{i}</td>
        <td style="padding:12px 10px;border-bottom:1px solid #eee;vertical-align:top;">
          <a href="{esc(url)}" target="_blank" rel="noopener noreferrer" style="color:#1a56db;text-decoration:none;font-weight:600;font-size:15px;">{title}</a>
          <div style="color:#555;font-size:13px;margin-top:2px;">{company} &nbsp;·&nbsp; {loc_disp}{(' &nbsp;·&nbsp; ' + salary) if salary else ''}</div>
          <div style="color:#444;font-size:13px;margin-top:6px;">{reason}</div>
        </td>
        <td style="padding:12px 10px;border-bottom:1px solid #eee;vertical-align:top;text-align:center;">
          <span style="display:inline-block;background:#e8f0fe;color:#1a56db;font-weight:700;font-size:13px;padding:4px 10px;border-radius:12px;">{score_disp}/10</span>
        </td>
      </tr>""")

    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f5f6f8;">
  <div style="max-width:680px;margin:0 auto;padding:24px 16px;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1a1a;">
    <h1 style="font-size:20px;margin:0 0 4px;">Job matches for {esc(candidate_name)}</h1>
    <p style="color:#666;font-size:13px;margin:0 0 18px;">Top {len(matches)} matches · generated {date.today().isoformat()}</p>
    <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="border-collapse:collapse;background:#fff;border:1px solid #eee;border-radius:8px;overflow:hidden;">
      <tr style="background:#fafafa;">
        <th style="padding:10px;text-align:left;font-size:12px;color:#888;border-bottom:1px solid #eee;">#</th>
        <th style="padding:10px;text-align:left;font-size:12px;color:#888;border-bottom:1px solid #eee;">Role</th>
        <th style="padding:10px;text-align:center;font-size:12px;color:#888;border-bottom:1px solid #eee;">Fit</th>
      </tr>{''.join(rows)}
    </table>
    <p style="color:#999;font-size:12px;margin-top:16px;">Links go directly to each job's application page.</p>
  </div>
</body>
</html>"""


# --------------------------------------------------------------------------
# Email delivery (Gmail SMTP by default, Resend optional)
# --------------------------------------------------------------------------

# Default provider. Gmail sends from your own account over SMTP using an app
# password, so it can email anyone with no domain setup. Override per-run with
# --provider resend.
DEFAULT_PROVIDER = "gmail"

# Gmail SMTP. Set GMAIL_ADDRESS (the sending account) and GMAIL_APP_PASSWORD
# (a 16-char Google app password, not your normal password) in .env.
GMAIL_SMTP_SERVER = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587


def send_via_gmail(to, subject, html_body, sender=None):
    """Send an HTML email through Gmail's SMTP using an app password. Reads
    GMAIL_ADDRESS / GMAIL_APP_PASSWORD from the environment. Returns a dict with
    an 'id' key (Gmail has no message id here, so it's a marker) on success and
    raises RuntimeError on failure so callers can surface it."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    address = sender or os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not address:
        raise RuntimeError("GMAIL_ADDRESS is not set in the environment.")
    if not password:
        raise RuntimeError("GMAIL_APP_PASSWORD is not set in the environment.")

    recipients = [to] if isinstance(to, str) else list(to)
    message = MIMEMultipart("alternative")
    message["From"] = address
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(GMAIL_SMTP_SERVER, GMAIL_SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(address, password)
            server.sendmail(address, recipients, message.as_string())
    except smtplib.SMTPAuthenticationError as exc:
        raise RuntimeError(
            f"Gmail login failed for {address} (check GMAIL_APP_PASSWORD): {exc}"
        ) from None
    except (smtplib.SMTPException, OSError) as exc:
        raise RuntimeError(f"Gmail send failed: {exc}") from None
    return {"id": "gmail"}


def send_email_report(to, subject, html_body, sender=None, provider=DEFAULT_PROVIDER):
    """Deliver the report via the chosen provider (Gmail by default). Returns the
    provider's response dict (has an 'id' on success)."""
    if provider == "resend":
        return send_via_resend(to, subject, html_body, sender or DEFAULT_FROM)
    return send_via_gmail(to, subject, html_body, sender)


# --------------------------------------------------------------------------
# Email via Resend (optional; use --provider resend)
# --------------------------------------------------------------------------

RESEND_ENDPOINT = "https://api.resend.com/emails"
# Resend's shared test sender: works with any account and needs no verified
# domain, but only delivers to the email you signed up to Resend with. Use a
# verified domain sender (e.g. jobs@yourdomain.com) to email anyone.
DEFAULT_FROM = "onboarding@resend.dev"


def send_via_resend(to, subject, html_body, sender=DEFAULT_FROM):
    """POST an HTML email to the Resend API. Returns the parsed JSON response
    (contains an 'id' on success). Raises RuntimeError with the API message on
    failure so callers can surface it."""
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        raise RuntimeError("RESEND_API_KEY is not set in the environment.")
    recipients = [to] if isinstance(to, str) else list(to)
    payload = json.dumps({
        "from": sender,
        "to": recipients,
        "subject": subject,
        "html": html_body,
    }).encode("utf-8")
    req = urllib.request.Request(
        RESEND_ENDPOINT, data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # A non-default User-Agent is required: Cloudflare (in front of the
            # Resend API) blocks the default "Python-urllib/x" agent with a 403
            # "error code: 1010".
            "User-Agent": "jscan-match-jobs/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Resend API error {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Resend: {exc.reason}") from None


# --------------------------------------------------------------------------
# Fetching (runs fetch_jobs.py and reloads the result)
# --------------------------------------------------------------------------

def run_fetch(jobs_path):
    """Run fetch_jobs.py to refresh jobs_path, then load and return the jobs.
    Runs it as a subprocess so its parsers/logging are reused as-is."""
    import subprocess
    import sys
    print("Fetching latest jobs from all sources...")
    try:
        proc = subprocess.run(
            [sys.executable, str(BASE_DIR / "fetch_jobs.py"), "--output", str(jobs_path)],
            capture_output=True, text=True, timeout=600,
        )
        summary = next((l for l in reversed(proc.stdout.splitlines()) if l.strip()), "")
        if summary:
            print(f"  {summary}")
        if proc.returncode != 0:
            print(f"  fetch_jobs.py exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    except Exception as exc:
        print(f"  fetch failed ({exc}); using whatever is already in {jobs_path}")
    return load_jobs(jobs_path)


# --------------------------------------------------------------------------
# Agent mode (original deep agent, kept behind --agent)
# --------------------------------------------------------------------------

def run_agent_mode(resume_text, jobs, args, subject):
    from langchain_core.tools import tool
    from deepagents import create_deep_agent
    from langgraph.errors import GraphRecursionError

    model_name = args.model
    top = args.top
    to = args.to
    sender = args.sender
    provider = args.provider
    jobs_path = args.jobs
    max_age = args.max_age_hours

    # Mutable index so the fetch_jobs tool can swap in fresh listings and every
    # other tool sees them immediately. Recency filter is applied here so both
    # the seed jobs and any freshly fetched ones are limited the same way.
    state = {}

    def reindex(js):
        js = filter_recent(js, max_age)
        state["jobs"] = js
        state["by_id"] = {j["id"]: j for j in js}
        bs = {}
        for j in js:
            bs.setdefault(j["id"].split(":", 1)[-1], j)
        state["by_suffix"] = bs

    reindex(jobs)

    def resolve(job_id):
        raw = (job_id or "").strip()
        for cand in (raw, raw.replace("/", ":", 1)):
            if cand in state["by_id"]:
                return state["by_id"][cand]
        return state["by_suffix"].get(re.split(r"[:/]", raw)[-1])

    @tool
    def fetch_jobs(confirm: bool = True) -> str:
        """Fetch the latest job listings from all configured sources into
        data/jobs.json and load them. Call this ONCE at the very start so you
        are matching against fresh jobs."""
        js = run_fetch(jobs_path)
        reindex(js)
        window = f" within the last {max_age}h" if max_age else ""
        return (f"Refreshed listings: {len(state['jobs'])} jobs now available{window}. "
                f"Use search_jobs next to find matches.")

    @tool
    def search_jobs(keywords: str, limit: int = 20) -> str:
        """Search jobs by comma-separated keywords (OR-matched, ranked)."""
        words = [w for w in tokenize(keywords) if w]
        if not words:
            return "No usable keywords."
        limit = max(1, min(int(limit), 30))
        scored = []
        for j in state["jobs"]:
            s, matched = 0, []
            for w in set(words):
                if w in j["_title_c"]:
                    s += 5; matched.append(w)
                elif w in j["_tags_c"]:
                    s += 3; matched.append(w)
                elif w in j["_body_c"]:
                    s += 1; matched.append(w)
            if s:
                scored.append((s, j, matched))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return "No jobs matched."
        return "\n".join(
            f"[{j['id']}] {j.get('title')} — {j.get('company') or 'n/a'} | "
            f"{j.get('location') or 'n/a'} | remote={bool(j.get('remote'))} | matched: {', '.join(mt)}"
            for _, j, mt in scored[:limit]
        )

    @tool
    def get_job_details(job_id: str) -> str:
        """Full details for one job_id."""
        j = resolve(job_id)
        if not j:
            return f"No job found with id '{job_id}'."
        d = j["_desc_clean"]
        d = d[:1800] + " ... [truncated]" if len(d) > 1800 else d
        return (f"id: {j['id']}\ntitle: {j.get('title')}\ncompany: {j.get('company')}\n"
                f"location: {j.get('location')}\nremote: {bool(j.get('remote'))}\n"
                f"salary: {j.get('salary') or 'not listed'}\nurl: {j.get('url')}\n"
                f"description:\n{d}")

    @tool
    def send_email(subject: str, html_body: str) -> str:
        """Email the final job-match report as an HTML email via Gmail.

        subject: the email subject line.
        html_body: the full report as an HTML fragment (use <h2>, <ul>/<li>,
            and <a href> links for each job). Call this ONLY once, at the very
            end, after you have finalized the matches.
        The recipient is fixed by the operator; you cannot change it.
        """
        if not to:
            return "No recipient configured (run with --to). Do not call send_email."
        try:
            resp = send_email_report(to, subject, html_body, sender, provider)
            return f"Email sent to {to} (id: {resp.get('id', 'ok')})."
        except Exception as exc:
            return f"Failed to send email: {exc}"

    tools = [fetch_jobs, search_jobs, get_job_details]
    send_instr = ""
    if to:
        tools.append(send_email)
        send_instr = (
            " Then email the finalized report to the configured recipient by "
            "calling send_email exactly once with a clear subject and an HTML "
            "body (each job as a linked <a href> item). Do not call send_email "
            "until the matches are final."
        )

    model = ChatOllama(model=model_name, temperature=0, num_ctx=NUM_CTX)
    agent = create_deep_agent(
        model=model, tools=tools,
        system_prompt=(
            "You run the whole job pipeline for one candidate. Steps, in order:\n"
            "1. Call fetch_jobs ONCE to pull the latest listings.\n"
            "2. Call search_jobs with skills from the resume; inspect promising "
            "hits with get_job_details.\n"
            "3. Judge fit by the job's core function and seniority, not shared "
            "keywords; a role in a different discipline (sales, marketing, "
            "product management) that merely shares a tool is a weak match.\n"
            f"4. Output the top {top} matches with title, company, url, a fit "
            "score /10, and a one-line reason. Never invent a job or a URL.\n"
            "EFFICIENCY: never call a tool with arguments you have already used, "
            "and never repeat a search you have already run. A handful of "
            "searches (at most ~4) and a few get_job_details calls is enough — "
            "then finalize. Do not keep searching once you have good candidates."
            + send_instr
        ),
    )

    inputs = {"messages": [{"role": "user", "content": (
        f"Resume:\n{resume_text}\n\nRun the full pipeline: fetch the latest "
        f"jobs, then report the top {top} matches.")}]}
    config = {"recursion_limit": args.recursion_limit}

    # Stream so the tool/LLM activity is visible as it happens.
    seen, final_text = 0, ""
    try:
        for upd in agent.stream(inputs, config, stream_mode="values"):
            msgs = upd.get("messages", [])
            for msg in msgs[seen:]:
                kind = getattr(msg, "type", "")
                calls = getattr(msg, "tool_calls", None)
                if calls:
                    for c in calls:
                        a = json.dumps(c.get("args", {}))
                        print(f"  -> {c['name']}({a[:200]})", flush=True)
                elif kind == "tool":
                    prev = " ".join((msg.content or "").split())
                    print(f"  <- {getattr(msg, 'name', 'tool')}: {prev[:160]}", flush=True)
                elif kind == "ai" and msg.content:
                    final_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if args.verbose:
                        print(f"  [ai] {' '.join(final_text.split())[:200]}", flush=True)
            seen = len(msgs)
    except GraphRecursionError:
        print(f"\nAgent hit its {args.recursion_limit}-step limit without finishing "
              f"(it was looping). Falling back to fast one-pass scoring.\n")
        fast_match(resume_text, state.get("jobs", jobs), args, subject)
        return

    print("\n" + "=" * 66)
    print("AGENT RESULT")
    print("=" * 66)
    print(final_text)


# --------------------------------------------------------------------------

def fast_match(resume_text, jobs, args, subject):
    """Fast pipeline: cosine pre-rank -> one LLM scoring pass -> print, HTML,
    and optional email. Also used as the agent-mode fallback."""
    idf = compute_idf(jobs)
    shortlist = prerank(jobs, resume_text, idf, args.shortlist)

    by_id = {j["id"]: j for j in jobs}
    by_suffix = {}
    for j in jobs:
        by_suffix.setdefault(j["id"].split(":", 1)[-1], j)

    matches = []
    if not args.no_llm and shortlist:
        print(f"Pre-ranked to {len(shortlist)} candidates. Scoring in one pass with '{args.model}'...")
        # reasoning=False suppresses the model's <think> output: faster + clean JSON.
        model = ChatOllama(model=args.model, temperature=0, num_ctx=NUM_CTX, reasoning=False)
        raw_matches = score_matches(model, resume_text, shortlist, args.top, verbose=args.verbose)
        matches = join_matches(raw_matches, by_id, by_suffix)
    if not matches:  # --no-llm, no jobs, or the model returned nothing usable
        if not shortlist:
            print("No jobs to match — check --max-age-hours or run a fetch (--fetch).")
        elif not args.no_llm:
            print("Using keyword pre-ranking order (no usable LLM output).")
        else:
            print(f"Pre-ranked to {len(shortlist)} candidates (keyword ranking, no LLM).")
        matches = [{"job": j, "fit_score": None, "reason": ""} for j in shortlist[:args.top]]

    name = extract_candidate_name(resume_text)
    print("\n" + "=" * 66)
    print(f"Found {len(matches)} job match(es) — {name}")
    print("=" * 66)
    if not args.summary_only:
        for i, m in enumerate(matches, 1):
            j = m["job"]
            sc = m.get("fit_score")
            sc = f"{sc:g}/10" if isinstance(sc, (int, float)) else "n/a"
            print(f"{i:2}. [{sc:>5}] {j.get('title')} — {j.get('company') or 'n/a'}")
            if m.get("reason"):
                print(f"        {m['reason']}")
            print(f"        {j.get('url')}")

    out_path = Path(args.html)
    html_doc = build_html(matches, name, out_path)
    out_path.write_text(html_doc)
    print(f"\nWrote {out_path}")

    if args.to and matches:
        frm = args.sender or (os.environ.get("GMAIL_ADDRESS") if args.provider == "gmail" else DEFAULT_FROM)
        print(f"Sending to {args.to} via {args.provider} (from {frm})...")
        try:
            resp = send_email_report(args.to, subject, html_doc, args.sender, args.provider)
            print(f"Sent. id: {resp.get('id', 'ok')}")
        except Exception as exc:
            print(f"Email send failed: {exc}")

    if args.open:
        import subprocess
        subprocess.run(["open", str(out_path)], check=False)


def main():
    ap = argparse.ArgumentParser(description="Match a resume against fetched jobs with a local model.")
    ap.add_argument("--resume", default=str(DEFAULT_RESUME))
    ap.add_argument("--jobs", default=str(DEFAULT_JOBS))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--top", type=int, default=10, help="how many matches to report")
    ap.add_argument("--shortlist", type=int, default=20, help="jobs pre-ranked into the single LLM pass")
    ap.add_argument("--max-age-hours", type=float, default=24.0, metavar="N",
                    help="only match jobs seen/posted within the last N hours (default 24; use 0 to disable)")
    ap.add_argument("--html", default=str(DEFAULT_HTML), help="output HTML path")
    ap.add_argument("--open", action="store_true", help="open the HTML when done")
    ap.add_argument("--no-llm", action="store_true", help="skip the model; use keyword ranking only (instant)")
    ap.add_argument("--fetch", action="store_true", help="refresh data/jobs.json via fetch_jobs.py before matching")
    ap.add_argument("--agent", action="store_true", help="deep-agent mode: fetch + match (+ email) in one agent run")
    ap.add_argument("--recursion-limit", type=int, default=60,
                    help="max agent steps before falling back to fast scoring (--agent)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="log LLM requests/replies and agent tool calls")
    ap.add_argument("--summary-only", action="store_true",
                    help="print only the number of matches, not every match")
    ap.add_argument("--to", help="email the HTML report to this address (Gmail by default; needs GMAIL_ADDRESS + GMAIL_APP_PASSWORD)")
    ap.add_argument("--send-only", action="store_true",
                    help="just email the existing --html file via --to; no fetch, match, or model")
    ap.add_argument("--provider", choices=["gmail", "resend"], default=DEFAULT_PROVIDER,
                    help=f"email provider for --to (default {DEFAULT_PROVIDER})")
    ap.add_argument("--from", dest="sender", default=None,
                    help="sender address for --to (default: GMAIL_ADDRESS for gmail, "
                         f"{DEFAULT_FROM} for resend)")
    ap.add_argument("--subject", help="email subject (default: 'Job matches — <date>')")
    args = ap.parse_args()

    subject = args.subject or f"Job matches — {date.today().isoformat()}"

    if args.send_only:
        if not args.to:
            print("--send-only needs --to (the recipient).")
            return
        path = Path(args.html)
        if not path.exists():
            print(f"No HTML at {path} — run a match first, or pass --html <file>.")
            return
        frm = args.sender or (os.environ.get("GMAIL_ADDRESS") if args.provider == "gmail" else DEFAULT_FROM)
        print(f"Sending {path} to {args.to} via {args.provider} (from {frm})...")
        try:
            resp = send_email_report(args.to, subject, path.read_text(), args.sender, args.provider)
            print(f"Sent. id: {resp.get('id', 'ok')}")
        except Exception as exc:
            print(f"Email send failed: {exc}")
        return

    resume_text = Path(args.resume).read_text()

    if args.agent:
        # The agent refreshes jobs itself via its fetch_jobs tool, so just seed
        # it with whatever is on disk (empty is fine on a first run).
        jobs = load_jobs(args.jobs) if Path(args.jobs).exists() else []
        run_agent_mode(resume_text, jobs, args, subject)
        return

    jobs = run_fetch(args.jobs) if args.fetch else load_jobs(args.jobs)
    if args.max_age_hours:
        before = len(jobs)
        jobs = filter_recent(jobs, args.max_age_hours)
        print(f"Loaded {before} jobs; {len(jobs)} within the last {args.max_age_hours:g}h.")
    else:
        print(f"Loaded {len(jobs)} jobs from {args.jobs}")

    fast_match(resume_text, jobs, args, subject)


if __name__ == "__main__":
    main()
