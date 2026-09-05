# jscan




A small, local-first job pipeline:

1. **`fetch_jobs.py`** — pulls tech job listings from several free job APIs into a
   single deduped `data/jobs.json`.
2. **`match_jobs.py`** — ranks those jobs against your resume using a local
   [Ollama](https://ollama.com) model, writes an email-ready `matches.html`, and
   can email it via Gmail (or [Resend](https://resend.com)).

Everything runs on your machine. No resume or job data leaves it, except the
optional email send.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # for fetch_jobs.py
.venv/bin/pip install deepagents langchain-ollama   # for match_jobs.py
```

You also need [Ollama](https://ollama.com) running with a model pulled. The
default model tag is `nemotron-3.5-lightning:30b-mlx`; override with `--model`.

```bash
ollama list          # confirm your model tag
ollama serve         # if it isn't already running
```

## 1. Fetch jobs — `fetch_jobs.py`

Reads sources from `sources.yaml`, calls them in parallel, normalizes each into a
common schema, and merges into `data/jobs.json` (deduped, with `first_seen` /
`last_seen` timestamps). A source that errors or is missing its API key is logged
and skipped — it never takes down the run. Designed to run once a day from
cron/launchd.

```bash
.venv/bin/python fetch_jobs.py
.venv/bin/python fetch_jobs.py --config sources.yaml --output data/jobs.json --workers 8
```

| Option | Default | Description |
|---|---|---|
| `--config` | `sources.yaml` | Job sources to fetch |
| `--output` | `data/jobs.json` | Where to write the merged jobs |
| `--workers` | `8` | Max parallel source fetches |

Sources that need a key (Adzuna, Findwork, Jooble India, USAJOBS) are enabled
by setting the env vars named in `sources.yaml` (e.g. `ADZUNA_APP_ID` /
`ADZUNA_APP_KEY` or `JOOBLE_IN_API_KEY`); the free no-signup sources (RemoteOK,
Arbeitnow, Himalayas, Remotive) work as-is.

The config includes India-focused API sources:

- `adzuna_in` uses Adzuna's India endpoint.
- `jooble_in` uses Jooble India's REST API and is skipped until
  `JOOBLE_IN_API_KEY` is set.

## 2. Match your resume — `match_jobs.py`

Reads a resume (`input_resume.md`), scores the jobs in `data/jobs.json` against
it, prints a ranked list, and writes `matches.html`.

**How it works (default / fast mode):**

1. **Pre-rank (instant, no model):** TF-IDF cosine similarity between the resume
   and every job. This is length-normalized, so a verbose posting can't win just
   by being long, and distinctive skills (burp, stride, sbom, kubernetes) drive
   the score. Cuts hundreds of jobs down to a high-signal shortlist.
2. **Score (one LLM call):** the shortlist goes to the local model in a single
   schema-constrained pass. The model returns only `job_id + fit_score + reason`;
   titles, companies, and links are joined back from the real records, so it
   **cannot hallucinate a job or a URL**. The prompt scores on the job's core
   function, not shared keywords, so marketing/sales/PM roles that merely share a
   buzzword are pushed down.

If the model returns nothing usable, it falls back to the cosine ordering, so the
tool never crashes or emits junk.

```bash
# Fast mode: top 10, writes matches.html
.venv/bin/python match_jobs.py

# Instant, no model at all (cosine ranking only)
.venv/bin/python match_jobs.py --no-llm

# Refresh jobs first (runs fetch_jobs.py), then match
.venv/bin/python match_jobs.py --fetch

# Tune counts and open the result
.venv/bin/python match_jobs.py --top 15 --shortlist 25 --open

# Email the HTML report via Gmail
.venv/bin/python match_jobs.py --to you@example.com

# Deep-agent mode: ONE run does the whole pipeline — fetch, match, and email
.venv/bin/python match_jobs.py --agent --to you@example.com
```

| Option | Default | Description |
|---|---|---|
| `--resume` | `input_resume.md` | Resume to match against |
| `--jobs` | `data/jobs.json` | Jobs file from `fetch_jobs.py` |
| `--model` | `nemotron-3.5-lightning:30b-mlx` | Ollama model tag |
| `--top` | `10` | How many matches to report |
| `--shortlist` | `20` | Jobs pre-ranked into the single LLM pass |
| `--max-age-hours` | `24` | Only match jobs whose freshest timestamp is within the last N hours (`0` disables) |
| `--html` | `matches.html` | Output HTML path |
| `--open` | off | Open the HTML when done (macOS `open`) |
| `--no-llm` | off | Skip the model; use keyword ranking only (instant) |
| `--fetch` | off | Run `fetch_jobs.py` to refresh `data/jobs.json` before matching (fast mode) |
| `--agent` | off | Deep-agent mode: fetch + match (+ email) in one agent run |
| `--recursion-limit` | `60` | Max agent steps before falling back to fast scoring (`--agent`) |
| `-v`, `--verbose` | off | Log LLM requests/replies and agent tool calls |
| `--to` | – | Email the HTML report to this address (Gmail by default) |
| `--send-only` | off | Just email the existing `--html` file via `--to`; no fetch/match/model |
| `--provider` | `gmail` | Email provider for `--to`: `gmail` or `resend` |
| `--from` | `GMAIL_ADDRESS` | Sender address for `--to` (`onboarding@resend.dev` for resend) |
| `--subject` | `Job matches — <date>` | Email subject |

### Recency filter (`--max-age-hours`)

Matching is limited to jobs from the **last 24 hours by default** — a job's age
is its freshest timestamp among `posted_at`, `first_seen`, and `last_seen`. The
job store itself is never pruned (it keeps growing), so this filter is what keeps
each run focused on fresh listings. Widen, narrow, or disable it:

```bash
.venv/bin/python match_jobs.py --fetch --to you@example.com          # last 24h (default)
.venv/bin/python match_jobs.py --max-age-hours 72                     # last 3 days
.venv/bin/python match_jobs.py --max-age-hours 0                      # no filter (all jobs)
```

Works in both fast and agent mode.

### Logging and the agent step limit

`--verbose` prints what the model is doing: in fast mode, the `[llm]` request
size, reply time, a reply preview, and how many matches parsed; in agent mode,
every `-> tool(args)` call and `<- tool: result` as it happens (agent tool calls
are shown even without `--verbose`).

Local models sometimes loop on near-identical searches in agent mode. If the
agent exceeds `--recursion-limit` steps, it no longer crashes — it prints a
notice and **falls back to the fast one-pass scoring** over the fetched jobs, so
you still get a ranked report (and email, if `--to` is set). For a local model,
fast mode (`--fetch`) is the more reliable path; agent mode is the flashier one.

### Modes at a glance

| Mode | Command | Speed | Output |
|---|---|---|---|
| Instant | `--no-llm` | Instant | Cosine ranking, no scores/reasons |
| Fast (default) | *(none)* | ~seconds warm | One LLM pass: scores + reasons |
| Agent | `--agent` | Minutes | Runs the whole pipeline itself: fetches, searches, matches, and (with `--to`) emails |

In `--agent` mode the agent has its own `fetch_jobs`, `search_jobs`,
`get_job_details`, and (when `--to` is set) `send_email` tools, and is
instructed to fetch fresh listings first — so a single command does everything.
Fast mode is faster and more predictable; agent mode is the "one cool run that
does it all."

## Emailing the report

`matches.html` uses inline styles and a simple table, and every job link opens in
a new tab. Two ways to send it:

- **Copy-paste into Gmail:** open `matches.html` in a browser, select all
  (Cmd+A), copy, and paste into the Gmail compose window. Gmail keeps the
  rendered formatting and links. (Pasting the raw HTML *source* does not work —
  it must be the rendered page.)
- **Send via Gmail (`--to`, the default):** the report is sent from your own
  Gmail account over SMTP using an [app password](https://myaccount.google.com/apppasswords).
  Set two vars in your shell **or** in a `.env` file in the project root (both
  `fetch_jobs.py` and `match_jobs.py` load `.env` automatically — this needs
  `python-dotenv`, in `requirements.txt`):

```bash
# in .env (or export in your shell):
GMAIL_ADDRESS=you@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx   # 16-char Google app password, not your login password

.venv/bin/python match_jobs.py --to anyone@example.com

# send the existing matches.html only — no fetch/match/model:
.venv/bin/python match_jobs.py --send-only --to anyone@example.com
```

Gmail sends from your own address, so it can email **anyone** with no domain
setup. The `--from` sender defaults to `GMAIL_ADDRESS`.

- **Send via Resend (`--provider resend`):** set `RESEND_API_KEY` (shell or
  `.env`) and pass `--provider resend`.

```bash
export RESEND_API_KEY=re_xxx   # or put it in .env
.venv/bin/python match_jobs.py --to you@example.com --provider resend
```

  The default `--from onboarding@resend.dev` is Resend's shared test address: it
  needs no domain setup **but only delivers to the email you registered with
  Resend.** To email any other address, verify a domain in Resend and pass
  `--from you@yourdomain.com`.

In `--agent` mode with `--to` set, the agent is given a `send_email` tool and
sends the finalized report itself (via Gmail unless `--provider resend`).

## Files

| Path | What it is |
|---|---|
| `fetch_jobs.py` | Daily job fetcher |
| `sources.yaml` | Job sources and their API/config |
| `match_jobs.py` | Resume-to-job matcher + emailer |
| `input_resume.md` | Your resume (Markdown) |
| `data/jobs.json` | Fetched, deduped jobs |
| `matches.html` | Generated match report |
| `logs/` | Per-day fetch logs |

## Daily use

```bash
# Two explicit steps (predictable):
.venv/bin/python fetch_jobs.py                          # refresh jobs
.venv/bin/python match_jobs.py --to you@example.com     # match + email

# Or fetch + match + email in one fast-mode command:
.venv/bin/python match_jobs.py --fetch --to you@example.com

# Or let the agent do the whole thing:
.venv/bin/python match_jobs.py --agent --to you@example.com
```

Wire any of these into cron/launchd for a daily job digest in your inbox.
