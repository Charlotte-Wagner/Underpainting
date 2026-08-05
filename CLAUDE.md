# Conventions for this repo

This file describes how Underpainting is built, for both future AI-assisted sessions and
human reviewers. It records what's intentional so a reviewer can tell a deliberate choice
from an oversight.

## Structure

- `app.py` — Streamlit UI and orchestration only. If a function doesn't need `st.*`
  calls, it doesn't belong in this file.
- `imaging.py` (and future modules like it) — pure numpy in, numpy out, zero Streamlit
  imports. This is what makes the core logic checkable by hand against a small array
  instead of only through the browser. New image-math functions follow this pattern:
  their own module, no framework dependency.
- `docs/` — architecture and design notes, kept short on purpose.
- `.streamlit/secrets.toml` — local-only, gitignored, never committed. `.env.example`
  documents the variable name it needs without containing a real value.

## Conventions

- **Comments explain why, not what.** `# Hardcoded on purpose. A curated pair of studies
  is a better demo than a slider that lets a visitor find the ugliest output the app can
  produce.` is the target — not a restatement of the line below it.
- **No tunable UI in v1.** Level counts, cluster counts, and similar knobs are curated
  constants, not sliders. Don't add configurability speculatively.
- **The Anthropic call is always gated behind an explicit user action.** Never call the
  model automatically on upload or rerun — Streamlit reruns the whole script on every
  interaction, so "automatic" means "unbounded spend on a public page."
- **Secrets never go in code, ever, not even temporarily to test something.** Read them
  from `st.secrets`. If a real key ever ends up in a commit, it's compromised the moment
  it's pushed — rotate it at the provider, deleting the file afterward is not sufficient.

## Testing approach

No automated test framework yet. Core image-math functions are written to be run against
a tiny hand-built array (see README's "Running the checks") — that's a deliberate
substitute for a full suite while the project is this small, not a placeholder for one
that quietly never arrives. If a real test suite gets added, it belongs alongside the
module it tests and should keep using the same small-array style of check.

## Scope

v1 is fixed: upload a photo, get a value study, a palette with paint names, an animated
stage progression, and a written step-by-step. No user accounts, no database, no mobile
app, no payments, no live camera. New feature ideas are noted (locally, not in this repo)
rather than started as a branch — a small finished tool beats an ambitious unfinished one.

## Git workflow

- Work on a feature branch; open a pull request into `main` rather than committing
  straight to `main`. `main` stays deployable.
- Keep commits small and single-purpose — one session's worth of work, one commit
  (matches this repo's existing history).
- Write commit messages that say what changed and, where it's not obvious, why.

## Never

- Never commit `.env`, `.streamlit/secrets.toml`, or any file containing a real API key.
- Never commit the private planning docs (`BUILD-PLAN.md`, `v2.md`, `debug-log.md`, and
  similar) — they're gitignored intentionally and stay local.
- Never write a README or resume claim about this app that isn't literally true of the
  code as it stands.
