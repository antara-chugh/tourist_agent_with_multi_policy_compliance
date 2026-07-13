"""
Very basic two-page front end:
  - "/"          Send a query to the tourist assistant agent and see its response.
  - "/policies"  Pick a deployment place and view its policy comparison report.

Run (from anywhere):
    uv run python webapp/app.py
Then open http://127.0.0.1:5001/
"""

import asyncio
import itertools
import logging
import sys
from pathlib import Path

# The agent/comparator modules live at the project root, one level up from
# this file, so make them importable regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import markdown
from flask import Flask, render_template, request
from pymilvus import MilvusClient

import compare_policies_embeddings_multi as cpe
from mellea.plugins import PluginViolationError
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from mellea.stdlib.session import start_session
from tourist_assistant_hooks import tool_sanitizer, user_input_safety
from tourist_assistant_tools import (
    convert_currency,
    find_travel_blogs,
    get_attractions,
    get_emergency_services,
    get_local_customs_and_etiquette,
    get_local_time,
    get_weather_forecast,
    notify_emergency_contact,
    translate_phrase,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("app")

app = Flask(__name__)

TOOLS = [
    get_attractions,
    get_emergency_services,
    get_weather_forecast,
    get_local_time,
    convert_currency,
    translate_phrase,
    get_local_customs_and_etiquette,
    notify_emergency_contact,
    find_travel_blogs,
]

MILVUS_DB = str(PROJECT_ROOT / "policy_embeddings.db")
MILVUS_COLLECTION = "policy_chunks"
LOCATIONS_CONFIG = str(PROJECT_ROOT / "policy_locations.yaml")

# One session is reused across requests so the model backend isn't
# reconnected on every query — fine for this single-process demo app.
_session = start_session(model_id="llama3.1", ctx=ChatContext(), plugins=[tool_sanitizer, user_input_safety])


@app.route("/", methods=["GET", "POST"])
def chat():
    query = ""
    answer = None
    error = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            error = "Please enter a query."
        else:
            try:
                out, _ = asyncio.run(react(
                    goal=query,
                    context=_session.ctx,
                    backend=_session.backend,
                    tools=TOOLS,
                    loop_budget=2,
                ))
                answer = str(out)
            except PluginViolationError as exc:
                log.warning("Request blocked by policy: [%s] %s", exc.code, exc.reason)
                error = f"Blocked by policy check ({exc.code}): {exc.reason}"
            except Exception as exc:
                log.exception("Agent query failed")
                error = f"Request blocked or failed: {exc}"

    return render_template("chat.html", query=query, answer=answer, error=error)


def _run_comparison(place: str, threshold: float = 0.6) -> str:
    location_map = cpe.load_location_config(LOCATIONS_CONFIG)
    labels = location_map.get(place)
    if not labels:
        raise ValueError(f"Unknown place '{place}'.")

    client = MilvusClient(uri=MILVUS_DB)
    try:
        if not client.has_collection(MILVUS_COLLECTION):
            raise RuntimeError("No policies indexed yet. Run index_policies.py first.")

        rows_by_label = cpe.fetch_policy_rows(client, MILVUS_COLLECTION, labels)
    finally:
        client.close()

    missing = [label for label, rows in rows_by_label.items() if not rows]
    if missing:
        raise RuntimeError(f"Not indexed yet: {', '.join(missing)}. Run index_policies.py on them first.")

    indices = [cpe.build_policy_index_from_rows(label, rows_by_label[label]) for label in labels]
    pair_results = [
        cpe.compare_pair(idx1, idx2, threshold)
        for idx1, idx2 in itertools.combinations(indices, 2)
    ]
    return cpe.generate_report(indices, pair_results, threshold)


@app.route("/policies", methods=["GET", "POST"])
def policies():
    try:
        location_map = cpe.load_location_config(LOCATIONS_CONFIG)
    except OSError as exc:
        return render_template("policies.html", places=[], selected_place=None,
                                report=None, error=f"Couldn't load locations config: {exc}")

    places = sorted(location_map)
    selected_place = request.values.get("place") or (places[0] if places else None)
    report_html = None
    error = None

    if request.method == "POST" and selected_place:
        try:
            report_md = _run_comparison(selected_place)
            report_html = markdown.markdown(report_md, extensions=["tables"])
        except Exception as exc:
            error = str(exc)

    return render_template("policies.html", places=places, selected_place=selected_place,
                            report_html=report_html, error=error)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
