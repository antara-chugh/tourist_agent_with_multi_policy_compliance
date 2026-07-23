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
import yaml
from flask import Flask, abort, render_template, request, send_from_directory
from pymilvus import MilvusClient

import compare_policies_embeddings_multi as cpe
import critique_enforcement
import eval as eval_
from mellea.backends import ModelOption
from mellea.plugins import PluginViolationError, register, unregister
from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from mellea.stdlib.session import start_session
from tourist_assistant_hooks import find_policy_conflicts, model_output_safety, tool_sanitizer, user_input_safety
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

# Emptied out temporarily to test the critique-and-revise pipeline in
# isolation from react()'s tool-calling loop: with no tools to call, the
# model can only respond with `final_answer`, which sidesteps both the
# "multiple tools were called with 'final'" assertion (a model bundling a
# regular tool call with the finalizer in one turn) and loop-budget
# exhaustion from multi-step tool use. Restore the list below to bring
# tool use back.
TOOLS = [
    # get_attractions,
    # get_emergency_services,
    # get_weather_forecast,
    # get_local_time,
    # convert_currency,
    # translate_phrase,
    # get_local_customs_and_etiquette,
    # notify_emergency_contact,
    # find_travel_blogs,
]

MILVUS_DB = str(PROJECT_ROOT / "policy_embeddings.db")
MILVUS_COLLECTION = "policy_chunks"
LOCATIONS_CONFIG = str(PROJECT_ROOT / "policy_locations.yaml")

POLICIES_DIR = PROJECT_ROOT / "policies"
# Policies appended to the system prompt for the "base + policies in prompt"
# enforcement mode below — prompt-based compliance, as opposed to the
# hook-based checks in find_policy_conflicts/model_output_safety.
PROMPT_POLICY_LABELS = [
    "alcohol_off_topic",
    "cybersecurity_risks",
    "financial_crimes_and_illegal_trading",
    "private_information",
]


def _format_policy_for_prompt(policy: dict) -> str:
    lines = [f"Policy: {policy.get('risk_group', '')} — {policy.get('description', '')}"]
    for risk in policy.get("risks", []) or []:
        lines.append(f"- {risk.get('risk', '')}: {risk.get('description', '')}")
        rules = risk.get("policy", {}) or {}
        for item in rules.get("reply_cannot_contain", []) or []:
            lines.append(f"    Must NOT include: {item}")
        for item in rules.get("reply_may_contain", []) or []:
            lines.append(f"    May include: {item}")
    return "\n".join(lines)


def _build_policies_prompt_text(labels: list[str]) -> str:
    sections = []
    for label in labels:
        with open(POLICIES_DIR / f"{label}.yaml") as f:
            sections.append(_format_policy_for_prompt(yaml.safe_load(f)))
    return (
        "In addition to your other instructions, you must comply with the "
        "following policies. Do not produce content described as \"Must NOT "
        "include\" under any matching risk; content described as \"May "
        "include\" is acceptable.\n\n" + "\n\n".join(sections)
    )


POLICIES_PROMPT_TEXT = _build_policies_prompt_text(PROMPT_POLICY_LABELS)

# Production systems put the agent's role, tone, and behavior in a system
# prompt rather than leaving it implicit — until now nothing here set one
# (mellea doesn't inject a default), so the model's only signal about its
# role came from the tourist-specific tool names/descriptions. This makes
# it a general-purpose assistant instead: it isn't told it's a tourist
# assistant, and it's explicitly told to ask rather than guess when a
# request is ambiguous.
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Read each request carefully and respond "
    "concisely. If a request is ambiguous, is missing information "
    "you'd need to answer well, or could reasonably be interpreted more than one "
    "way, ask the user a clarifying question instead of guessing. "
)

# react()'s tool-calling loop asserts that the finalizer tool is called
# alone in its turn (mellea/stdlib/frameworks/react.py) — a model that
# bundles a regular tool call together with the finalizer in one turn trips
# that assertion and aborts the whole call. A tight budget seems to push
# local/smaller models toward trying to wrap up in fewer turns, which makes
# that more likely; a looser budget gives the model more room to finish
# cleanly. Raised from 2 -> 4; tune further if "multiple tools were called
# with 'final'" or "could not complete react loop" keep showing up, at the
# cost of slower requests (each turn is a full generation call).
REACT_LOOP_BUDGET = 4

# One session is reused across requests so the model backend isn't
# reconnected on every query — fine for this single-process demo app.
# Registered by default: the full current hard-block behavior
# (find_policy_conflicts on the request, model_output_safety on the
# response). react()'s returned (updated) context is discarded below
# (`out, _ = ...`) rather than written back to `_session.ctx`, so — by
# construction, not by anything special done here — every call already
# starts from the same context state; that's what makes it safe to run
# several passes back-to-back for the comparison modes without one pass's
# turn leaking into the next.
#
# mellea's plugin registry dispatches by hook type globally — a
# "session-scoped" registration only controls *bulk cleanup*, not which
# session a firing hook applies to (see mellea/plugins/manager.invoke_hook:
# it has no per-session filtering) — so there's no way to make a hook fire
# for only some requests while leaving a second, differently-configured
# session "open" at the same time. Instead, `_run_open_pass` below
# unregisters find_policy_conflicts + model_output_safety for the duration
# of one request and re-registers them right after. Safe under Flask's
# default single-threaded dev server, where requests are handled one at a
# time. detect_travel_distress and the tool-security hooks (tool_sanitizer)
# are left out of this toggle entirely and stay active in every mode —
# they're not part of what's being compared here.
# Keep AGENT_MODEL_ID and
# critique_enforcement.DEFAULT_AGENT_MODEL_ID in sync — revision is always
# supposed to be done by the same model that produced the original
# response (see critique_enforcement.py's module docstring).
AGENT_MODEL_ID = "gemma4:31b-cloud"

_session = start_session(
    model_id=AGENT_MODEL_ID, ctx=ChatContext(),
    model_options={ModelOption.SYSTEM_PROMPT: SYSTEM_PROMPT},
    plugins=[tool_sanitizer],
)

ENFORCEMENT_MODES = {
    "base": "Base model (no enforcement)",
    "policies_in_prompt": "Base model + policies appended to prompt",
    "block": "Hard block (current policy hook)",
    "critique_self": "Critique & revise — self-critique",
    "critique_separate": "Critique & revise — separate critique model",
    "critique_both": "Critique & revise — compare self vs. separate",
    "all_five": "Compare all 5 (base, policies-in-prompt, hard block, self-critique, separate-critique)",
}
DEFAULT_ENFORCEMENT_MODE = "block"


async def _run_open_pass(query: str) -> tuple[str | None, str | None]:
    
    try:
        out, _ = await react(
            goal=query, context=_session.ctx, backend=_session.backend, tools=TOOLS, loop_budget=REACT_LOOP_BUDGET,
        )
        return str(out), None
    except Exception as exc:
        log.exception("Open-pass agent query failed")
        return None, f"Request failed: {exc}"


async def _run_policies_in_prompt_pass(query: str) -> tuple[str | None, str | None]:
    """Run the agent with no enforcement hooks active, but with the selected
    policies (PROMPT_POLICY_LABELS) appended to the system prompt for just
    this pass — prompt-based compliance instead of hook-based enforcement."""
    try:
        out, _ = await react(
            goal=query, context=_session.ctx, backend=_session.backend, tools=TOOLS, loop_budget=REACT_LOOP_BUDGET,
            model_options={ModelOption.SYSTEM_PROMPT: SYSTEM_PROMPT + "\n\n" + POLICIES_PROMPT_TEXT},
        )
        return str(out), None
    except Exception as exc:
        log.exception("Policies-in-prompt agent query failed")
        return None, f"Request failed: {exc}"


async def _run_block_pass(query: str) -> tuple[str | None, str | None]:
    """Run the agent with the current hard-block hooks fully active — the
    existing default ("block") behavior."""
    register(find_policy_conflicts)
    register(model_output_safety)
    try:
        out, _ = await react(
            goal=query, context=_session.ctx, backend=_session.backend, tools=TOOLS, loop_budget=REACT_LOOP_BUDGET,
        )
        return str(out), None
    except PluginViolationError as exc:
        log.warning("Request blocked by policy: [%s] %s", exc.code, exc.reason)
        return None, f"Blocked by policy check ({exc.code}): {exc.reason}"
    except Exception as exc:
        log.exception("Agent query failed")
        return None, f"Request failed: {exc}"
    finally: 
        unregister(find_policy_conflicts)
        unregister(model_output_safety)


def _critique_result(label: str, answer: str | None, error: str | None, critique_run: dict | None) -> dict:
    return {"label": label, "answer": answer, "error": error, "critique_run": critique_run}


async def _run_chat_turn(query: str, enforcement_mode: str) -> dict:
    """Run the agent under the selected enforcement configuration. Returns
    {"results": [...]} — one entry per intervention shown (more than one
    for the comparison modes: critique_both, all_five)."""
    if enforcement_mode == "base":
        answer, error = await _run_open_pass(query)
        return {"results": [_critique_result(ENFORCEMENT_MODES["base"], answer, error, None)]}

    if enforcement_mode == "policies_in_prompt":
        answer, error = await _run_policies_in_prompt_pass(query)
        return {"results": [_critique_result(ENFORCEMENT_MODES["policies_in_prompt"], answer, error, None)]}

    if enforcement_mode == "block":
        answer, error = await _run_block_pass(query)
        return {"results": [_critique_result(ENFORCEMENT_MODES["block"], answer, error, None)]}

    if enforcement_mode in ("critique_self", "critique_separate", "critique_both"):
        answer, error = await _run_open_pass(query)
        critique_modes = {
            "critique_self": ["self"],
            "critique_separate": ["separate"],
            "critique_both": ["self", "separate"],
        }[enforcement_mode]
        results = []
        for mode in critique_modes:
            critique_run = (
                await critique_enforcement.critique_and_revise(query, answer, mode=mode, agent_model_id=AGENT_MODEL_ID)
                if answer else None
            )
            results.append(_critique_result(f"Critique & revise — {mode}-critique", answer, error, critique_run))
        return {"results": results}

    if enforcement_mode == "all_five":
        base_answer, base_error = await _run_open_pass(query)
        policies_in_prompt_answer, policies_in_prompt_error = await _run_policies_in_prompt_pass(query)
        block_answer, block_error = await _run_block_pass(query)
        results = [
            _critique_result(ENFORCEMENT_MODES["base"], base_answer, base_error, None),
            _critique_result(ENFORCEMENT_MODES["policies_in_prompt"], policies_in_prompt_answer, policies_in_prompt_error, None),
            _critique_result(ENFORCEMENT_MODES["block"], block_answer, block_error, None),
        ]
        for mode in ("self", "separate"):
            critique_run = (
                await critique_enforcement.critique_and_revise(
                    query, base_answer, mode=mode, agent_model_id=AGENT_MODEL_ID,
                )
                if base_answer else None
            )
            results.append(_critique_result(f"Critique & revise — {mode}-critique", base_answer, base_error, critique_run))
        return {"results": results}

    raise ValueError(f"Unknown enforcement_mode: {enforcement_mode!r}")


@app.route("/", methods=["GET", "POST"])
def chat():
    query = ""
    error = None
    results = []
    enforcement_mode = request.values.get("enforcement_mode", DEFAULT_ENFORCEMENT_MODE)
    if enforcement_mode not in ENFORCEMENT_MODES:
        enforcement_mode = DEFAULT_ENFORCEMENT_MODE

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if not query:
            error = "Please enter a query."
        else:
            try:
                turn = asyncio.run(_run_chat_turn(query, enforcement_mode))
                results = turn["results"]
            except Exception as exc:
                log.exception("Chat turn failed")
                error = f"Request failed: {exc}"

    return render_template(
        "chat.html", query=query, error=error,
        enforcement_modes=ENFORCEMENT_MODES, enforcement_mode=enforcement_mode,
        results=results,
    )


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


# Metric series charted per evaluation type — key must match a field in each
# threshold row (see eval.eval_prompts / eval.eval_QA_pairs docstrings).
PROMPTS_CHART_SERIES = [
    ("block_fired_rate", "Block fired rate"),
    ("any_hit_match_rate", "Any-hit match rate"),
    ("top_hit_match_rate", "Top-hit match rate"),
]
QA_CLASSIFICATION_SERIES = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("safety_accuracy", "Safety accuracy"),
]
QA_MATCH_SERIES = [
    ("any_hit_match_rate", "Any-hit match rate"),
    ("top_hit_match_rate", "Top-hit match rate"),
]

# Only these exact filenames may be downloaded via /test/download/<name> —
# an allowlist keeps that route from being an arbitrary-file-read primitive.
_DOWNLOADABLE_CSVS = {"eval_prompts_samples.csv", "eval_qa_pairs_samples.csv", "eval_qa_pairs_errors.csv"}


@app.route("/test", methods=["GET", "POST"])
def test():
    max_rows = request.form.get("max_rows", eval_.DEFAULT_MAX_ROWS, type=int)

    result = None
    charts = {}
    error = None

    if request.method == "POST":
        try:
            result = eval_.run_eval(
                split=eval_.DATASET_SPLIT, max_rows=max_rows,
                policy_db=MILVUS_DB, policy_collection=MILVUS_COLLECTION,
                output_dir=eval_.DEFAULT_OUTPUT_DIR,
            )
            for policy, policy_result in result["prompts"].items():
                charts[f"prompts||{policy}"] = eval_.render_threshold_chart(
                    policy_result["thresholds"], f"chart-prompts-{policy}", PROMPTS_CHART_SERIES
                )
            for policy, policy_result in result["qa_pairs"].items():
                charts[f"qa_classification||{policy}"] = eval_.render_threshold_chart(
                    policy_result["thresholds"], f"chart-qa-class-{policy}", QA_CLASSIFICATION_SERIES
                )
                charts[f"qa_match||{policy}"] = eval_.render_threshold_chart(
                    policy_result["thresholds"], f"chart-qa-match-{policy}", QA_MATCH_SERIES
                )
        except Exception as exc:
            log.exception("Evaluation failed")
            error = f"Evaluation failed: {exc}"

    return render_template(
        "test.html",
        max_rows=max_rows,
        dataset_name=eval_.DATASET_NAME,
        dataset_split=eval_.DATASET_SPLIT,
        hook_type=eval_.HOOK_TYPE,
        operating_threshold=eval_.OPERATING_THRESHOLD,
        result=result,
        charts=charts,
        error=error,
    )


@app.route("/test/download/<path:filename>")
def test_download(filename):
    if filename not in _DOWNLOADABLE_CSVS:
        abort(404)
    return send_from_directory(eval_.DEFAULT_OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
