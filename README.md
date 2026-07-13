# Multi-Policy Compliance with Mellea

A tourist information agent (built with [Mellea](https://github.com/generative-computing/mellea)) that enforces
region-specific content policies at request time, plus tooling to embed, index, and compare those policies, and a
minimal web front end for both.

## What's in this repo

| Path | What it does |
|---|---|
| `tourist_assistant_tools.py` | The agent's `@tool`-decorated functions (attractions, weather, currency, translation, travel blogs, emergency services, emergency-contact notification). Most call real, free, key-less public APIs (OpenStreetMap, Open-Meteo, Frankfurter, MyMemory). |
| `tourist_assistant_hooks.py` | Mellea plugin hooks: a tool allow-list, PII redaction on tool output, audit logging, a travel-distress detector, and `find_policy_conflicts` — embeds the incoming request and blocks it if it's semantically close to any indexed policy's `reply_cannot_contain` chunks. |
| `tourist_assistant.py` | Runs the agent standalone (a few example goals) via Mellea's `react()` loop. |
| `index_policies.py` | Chunks policy YAML files, embeds each chunk with `sentence-transformers`, and upserts them into a local Milvus Lite vector DB. Re-indexing a policy only replaces its own rows. |
| `compare_policies_embeddings_multi.py` | Reads precomputed vectors back out of Milvus (no embedding model needed at compare time) and generates a semantic conflict/agreement report between two or more policies, either by explicit file paths or by `--place <name>`. |
| `policy_locations.yaml` | Maps a deployment place (e.g. `us`, `eu`) to the policy labels enforced there. Editing this never requires re-indexing. |
| `policies/` | Example policy YAML files (alcohol content policies, financial crime, private information, cybersecurity). |
| `webapp/` | A minimal two-page Flask front end: send a query to the tourist agent, or view the policy comparison report for a place. |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/), running locally, with the `llama3.1` model pulled:
  ```
  ollama pull llama3.1
  ```

## Setup

```bash
uv venv
uv pip install --python .venv/bin/python \
    mellea sentence-transformers "pymilvus[milvus_lite]" flask markdown pyyaml httpx
```

## 1. Index the policies into the vector DB

```bash
uv run python index_policies.py policies/
```

This creates `policy_embeddings.db` (a local Milvus Lite database — just a folder on disk, no server to run)
and a `policy_chunks` collection. Re-run this against a single changed policy file at any time; it only
replaces that policy's rows and leaves the rest of the collection untouched.

## 2. Configure which policies apply where

Edit `policy_locations.yaml`:

```yaml
locations:
  us:
    - alcohol_prohibited
    - alcohol_off_topic
  eu:
    - alcohol_consumption_permissive
    - alcohol_anti_coercion
    - alcohol_off_topic
```

Labels are the policy YAML filename stems. This file is the only thing that changes when a policy's
applicability changes — it never triggers re-indexing.

## 3. Run the agent

Standalone (a few example goals, run once and exit):

```bash
uv run python tourist_assistant.py
```

Compare policies for a place from the command line:

```bash
uv run python compare_policies_embeddings_multi.py --place us
```

This prints the report and saves it to `policy_comparison_report.md`.

## 4. Run the web front end

```bash
uv run python webapp/app.py
```

Then open http://127.0.0.1:5001/:

- **`/`** — send a free-form query to the tourist assistant agent. If the request semantically matches
  something an indexed policy forbids, the block reason (policy, risk, matched text, similarity score) is
  shown instead of a response.
- **`/policies`** — pick a place from the `policy_locations.yaml` dropdown and view its rendered
  comparison report (topic-similarity matrix, conflicts, agreements).

## Notes

- Milvus Lite only allows one process to hold the `policy_embeddings.db` file at a time. Stop the web app
  (or any other script using it) before running `index_policies.py` or the CLI comparator against the same
  file.
- `tourist_assistant_tools.py` calls real third-party APIs (OpenStreetMap Nominatim/Overpass, Open-Meteo,
  Frankfurter, MyMemory, DuckDuckGo's HTML results) — expect occasional rate limiting or transient failures
  from those services.
