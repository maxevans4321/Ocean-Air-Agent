# Ocean Air demo

A frontend for `agent.py`, deployed as a static site (`/public`) plus a
single Vercel serverless function (`/api/index.py`) that wraps `agent.py`'s
existing functions — `call_router`, `call_planner`, `call_executor`,
`call_validator` — without modifying that file at all.

## Structure

```
api/
  index.py      <- serverless function, exposes POST /api/chat
  agent.py      <- your agent, unchanged
public/
  index.html    <- the chat UI, served statically
vercel.json     <- routes /api/* to the function; /public is auto-served
requirements.txt
.env.example
```

**Important — stateless by design:** Vercel functions don't persist memory
between requests, so conversation history is held in the browser (in
`index.html`'s JS) and sent with every request, not stored server-side.

## 1. Push to GitHub

```bash
cd ocean-air-demo
git init
git add .
git commit -m "Ocean Air demo"
gh repo create ocean-air-demo --private --source=. --push
```

(Or create the repo on github.com first and `git remote add origin <url>`,
then `git push -u origin main`.)

Double check `.env` is NOT in the repo — `.gitignore` already excludes it,
but worth a `git status` check before your first commit.

## 2. Deploy on Vercel

- Go to vercel.com → **Add New Project** → import the GitHub repo
- Vercel will detect `vercel.json` and the `/api` + `/public` structure automatically
- Before deploying, add your environment variable:
  **Project Settings → Environment Variables** → add `ANTHROPIC_API_KEY`
  with your real key. Never put it in the repo itself.
- Deploy

Vercel will give you a live URL (e.g. `ocean-air-demo.vercel.app`) with
the chat UI at the root and the API at `/api/chat`.

## Local development (optional, before deploying)

```bash
pip install -r requirements.txt
cp .env.example .env   # add your real ANTHROPIC_API_KEY
uvicorn api.index:app --reload --app-dir .
```

Serve `public/index.html` with any static server, or open it directly —
just note the frontend calls a relative `/api/chat` path, so for local
testing it's easiest to run everything behind one origin (e.g. `vercel dev`,
if you have the Vercel CLI installed, reproduces the deployed setup
locally including routing).

## What you get

- A clean chat interface branded for Ocean Air
- A **live "Agent reasoning" panel** showing, per message: the router's
  classification, the planner's decision, the executor's draft reply, and
  the validator's approve/block decision
- A "Hide reasoning" toggle and a "Reset conversation" button
