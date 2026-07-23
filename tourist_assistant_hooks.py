import logging
import re
from pathlib import Path

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from mellea import start_session
from mellea.backends import ModelOption, tool
from mellea.plugins import (
    HookType,
    PluginMode,
    PluginResult,
    PluginSet,
    PluginViolationError,
    block,
    hook,
)
from mellea.stdlib.functional import _call_tools
from mellea.stdlib.requirements import uses_tool
from tourist_assistant_tools import get_attractions, get_emergency_services, get_weather_forecast, get_local_time, get_local_customs_and_etiquette, convert_currency, translate_phrase, find_travel_blogs, _run_notification_obligation



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("fancy_logger").setLevel(logging.ERROR)
log = logging.getLogger("tool_hooks")

_HTTP_TIMEOUT = 10.0
# Nominatim's usage policy requires a descriptive User-Agent and no more than ~1 req/sec.
_USER_AGENT = "tourist-assistant-example/1.0 (contact: example@example.com)"

# ---------------------------------------------------------------------------
# Plugin 1 — Tool allow list (enforce)
#
# Only tools explicitly listed in ALLOWED_TOOLS may be called. Any tool call
# for an unlisted tool is blocked before it reaches the function.
# ---------------------------------------------------------------------------

ALLOWED_TOOLS: frozenset[str] = frozenset({
    "get_attractions",
    "get_emergency_services",
    "get_weather_forecast",
    "get_local_time",
    "convert_currency",
    "translate_phrase",
    "get_local_customs_and_etiquette",
    "notify_emergency_contact",
    "find_travel_blogs",
})


@hook(HookType.TOOL_PRE_INVOKE, mode=PluginMode.CONCURRENT, priority=5)
async def enforce_tool_allowlist(payload, _):
    """Block any tool not on the explicit allow list."""
    tool_name = payload.model_tool_call.name
    if tool_name not in ALLOWED_TOOLS:
        log.warning(
            "[allowlist] BLOCKED tool=%r — not in allowed set %s",
            tool_name,
            sorted(ALLOWED_TOOLS),
        )
        return block(
            f"Tool '{tool_name}' is not permitted",
            code="TOOL_NOT_ALLOWED",
            details={"tool": tool_name, "allowed": sorted(ALLOWED_TOOLS)},
        )
    log.info("[allowlist] permitted tool=%r", tool_name)


# ---------------------------------------------------------------------------
# Plugin 2 — Tool PII detector
#
# Ensures sensitive information (passport numbers, credit cards, emails,
# phone numbers) is not passed into LLM context from tool output.
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b[A-Z]{1,2}[0-9]{6,9}\b"),  # passport-number-like tokens
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),  # credit-card-like sequences
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
]

@hook(HookType.TOOL_POST_INVOKE, mode=PluginMode.SEQUENTIAL, priority=5)
async def redact_pii_from_tool_output(payload, _) -> PluginResult | None:
    """Scrub passport/credit-card/email/phone from tool output before it enters context."""
    raw: str = str(payload.tool_output or "")
    redacted = raw
    for pattern in _PII_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    if redacted == raw:
        return None
    log.warning("[pii] Redacted PII from output of tool=%r", payload.model_tool_call.name)
    modified = payload.model_copy(update={"tool_output": redacted})
    return PluginResult(continue_processing=True, modified_payload=modified)


# ---------------------------------------------------------------------------
# Plugin 3 — Tool audit logger (fire-and-forget)
#
# Records every tool invocation outcome for audit purposes. Uses
# fire_and_forget so it never adds latency to the main execution path.
# ---------------------------------------------------------------------------


@hook(HookType.TOOL_POST_INVOKE, mode=PluginMode.FIRE_AND_FORGET)
async def audit_tool_calls(payload, _):
    """Log the result of every tool call for audit purposes."""
    status = "OK" if payload.success else "ERROR"
    tool_name = payload.model_tool_call.name
    log.info(
        "[audit] tool=%r status=%s latency=%dms args=%s",
        tool_name,
        status,
        payload.execution_time_ms,
        payload.model_tool_call.args,
    )
    if not payload.success and payload.error is not None:
        log.error("[audit] tool=%r error=%r", tool_name, str(payload.error))


# ---------------------------------------------------------------------------
# Plugin 4 — Detect travel-distress message
#
# Intercepts every user message and checks for signs the traveler may be in
# danger, robbed, or lost. On detection, blocks and sends messages to the
# traveler's emergency contact and home embassy.
# ---------------------------------------------------------------------------
_DISTRESS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(robbed|mugged|stolen (my|from me))\b", re.IGNORECASE),
    re.compile(r"\b(lost my passport|passport (was )?stolen)\b", re.IGNORECASE),
    re.compile(r"\b(i('m| am) (lost|scared|in danger|being followed))\b", re.IGNORECASE),
    re.compile(r"\b(someone is following me|feel unsafe)\b", re.IGNORECASE),
    re.compile(r"\b(need help now|emergency|call the police)\b", re.IGNORECASE),
]

@hook(HookType.GENERATION_PRE_CALL, mode=PluginMode.SEQUENTIAL, priority=1)
async def detect_travel_distress(payload, ctx):
    raw_context = payload.context
    context = ""

    if hasattr(raw_context, 'last_turn'):
        last = raw_context.last_turn()
        model_input = last.model_input if last else None
        """
        Note: Parsing context is reliant on knowledge that the user message/goal is encoded in the
        .goal attribute of the model input, accessed via the last turn of the context. The goal and
        tools from the react request are stored as "ReactInitiator" objects in context.
        """
        if model_input is not None:
            if hasattr(model_input, 'goal'):
                context += (str(model_input.goal))
                print(f"Extracted goal from context: {model_input.goal}")

    if not any(p.search(context) for p in _DISTRESS_PATTERNS):
        return None
    log.warning("[distress] Flagged message — triggering obligation pipeline.")
    _run_notification_obligation("username", context)
    return block(
        f"A travel-distress message was detected, message sent to emergency contact and embassy",
        code="USER_UNSAFE",
        details={"message": context},
    )

_POLICY_DB = str(Path(__file__).resolve().parent / "policy_embeddings.db")
_POLICY_COLLECTION = "policy_chunks"
_POLICY_CONFLICT_THRESHOLD = 0.5

_embedding_model: SentenceTransformer | None = None
_milvus_client: MilvusClient | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazily load the same model index_policies.py embeds with — embeddings
    from different models live in different vector spaces and aren't comparable."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_milvus_client() -> MilvusClient | None:
    """Lazily open (and cache) a connection to the policy vector DB. Returns
    None if nothing has been indexed yet, so the hook can no-op gracefully."""
    global _milvus_client
    if _milvus_client is None:
        client = MilvusClient(uri=_POLICY_DB)
        if not client.has_collection(_POLICY_COLLECTION):
            client.close()
            return None
        client.load_collection(_POLICY_COLLECTION)
        _milvus_client = client
    return _milvus_client


def _hit_fields(hit: dict) -> dict:
    # MilvusClient.search() nests output_fields under "entity" in some
    # pymilvus versions and flattens them into the hit dict in others.
    return hit.get("entity", hit)


def _check_text_for_policy_conflicts(text: str, source: str) -> PluginResult | None:
    """Embed `text` and check it against every indexed policy's reply_cannot_contain
    chunks. Returns a block() PluginResult if it's semantically close to something
    a policy forbids, else None. `source` (e.g. "request"/"response") is only used
    in logging/reason text so pre-call vs post-call hits are distinguishable."""
    if not text.strip():
        return None

    client = _get_milvus_client()
    if client is None:
        log.warning("[policy_conflicts] no policy_chunks collection found — run index_policies.py first")
        return None

    query_vector = _get_embedding_model().encode(text).tolist()
    hits = client.search(
        collection_name=_POLICY_COLLECTION,
        data=[query_vector],
        filter='chunk_type == "cannot_contain"',
        limit=5,
        output_fields=["text", "risk_name", "risk_id", "policy_label"],
    )[0]

    # For metric_type="COSINE", Milvus returns *distance* (1 - cosine similarity),
    # not similarity — smaller means more similar, so convert back before thresholding.
    scored = [(1 - hit["distance"], hit) for hit in hits]
    matches = [(similarity, hit) for similarity, hit in scored if similarity > _POLICY_CONFLICT_THRESHOLD]
    if not matches:
        return None

    log.warning(
        "[policy_conflicts] %s matched %d forbidden policy chunk(s) above threshold %.2f",
        source, len(matches), _POLICY_CONFLICT_THRESHOLD,
    )
    # PluginViolationError only carries `reason`/`code` out of the plugin
    # pipeline (the `details` dict below is dropped by mellea's plugin
    # manager when it raises the exception), so name the violated policy
    # directly in the reason text — that's what ends up on the frontend.
    top_similarity, top_hit = matches[0]
    top_fields = _hit_fields(top_hit)
    reason = (
        f"The {source} conflicts with policy '{top_fields['policy_label']}' "
        f"(risk {top_fields['risk_id']} – {top_fields['risk_name']}): "
        f"forbids \"{top_fields['text']}\" (similarity {top_similarity:.2f})"
    )
    return block(
        reason,
        code="POLICY_CONFLICT",
        details={
            "text": text,
            "hits": [
                {
                    "similarity": round(similarity, 3),
                    "policy_label": _hit_fields(hit)["policy_label"],
                    "risk_name": _hit_fields(hit)["risk_name"],
                    "risk_id": _hit_fields(hit)["risk_id"],
                    "text": _hit_fields(hit)["text"],
                }
                for similarity, hit in matches
            ],
        },
    )


@hook(HookType.GENERATION_PRE_CALL, mode=PluginMode.SEQUENTIAL, priority=2)
async def find_policy_conflicts(payload, ctx):
    """Embed the user's request and check it against every indexed policy's
    reply_cannot_contain chunks. If it's semantically close to something a
    policy forbids, block and report which policy/risk it hit."""
    raw_context = payload.context
    context = ""

    if hasattr(raw_context, 'last_turn'):
        last = raw_context.last_turn()
        model_input = last.model_input if last else None
        """
        Note: Parsing context is reliant on knowledge that the user message/goal is encoded in the
        .goal attribute of the model input, accessed via the last turn of the context. The goal and
        tools from the react request are stored as "ReactInitiator" objects in context.
        """
        if model_input is not None:
            if hasattr(model_input, 'goal'):
                context += (str(model_input.goal))

    return _check_text_for_policy_conflicts(context, source="request")


@hook(HookType.GENERATION_POST_CALL, mode=PluginMode.SEQUENTIAL, priority=2)
async def find_policy_conflicts_model_response(payload, ctx):
    """Embed the model's fully-generated response and check it against every
    indexed policy's reply_cannot_contain chunks. If the response is
    semantically close to something a policy forbids, block it from being
    returned and report which policy/risk it hit.

    Per mellea's GenerationPostCallPayload (mellea/plugins/hooks/generation.py),
    this hook fires once the model output thunk is fully computed, so
    `model_output.value` is guaranteed to already hold the generated text —
    unlike GENERATION_PRE_CALL, there's no `.context`/`.action` here to read
    a user goal from.
    """
    model_output = payload.model_output
    response_text = getattr(model_output, "value", None) or ""

    return _check_text_for_policy_conflicts(response_text, source="response")

# ---------------------------------------------------------------------------
# Compose into PluginSets for clean session-scoped registration
# ---------------------------------------------------------------------------

user_input_safety = PluginSet("user-input-safety", [detect_travel_distress, find_policy_conflicts])
model_output_safety = PluginSet("model_output_verification", [find_policy_conflicts_model_response])

tool_security = PluginSet("tool-security", [enforce_tool_allowlist])

tool_sanitizer = PluginSet("tool-sanitizer", [redact_pii_from_tool_output, audit_tool_calls])


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _run_scenario(name: str, fn) -> None:
    """Run a scenario function, logging any PluginViolationError without halting."""
    log.info("=== %s ===", name)
    try:
        fn()
    except PluginViolationError as e:
        log.warning(
            "Execution blocked on %s: [%s] %s (plugin=%s)",
            e.hook_type,
            e.code,
            e.reason,
            e.plugin_name,
        )
    log.info("")


def scenario_1_allowed_tool(all_tools):
    """Scenario 1: allowed tool call (get_attractions)."""
    with start_session(model_id="llama3.1", plugins=[tool_security]) as m:
        result = m.instruct(
            description="What are some good attractions to visit in Kyoto?",
            requirements=[uses_tool("get_attractions")],
            model_options={ModelOption.TOOLS: all_tools},
            tool_calls=True,
        )
        tool_outputs = _call_tools(result, m.backend)
        if tool_outputs:
            log.info("Tool returned: %s", tool_outputs[0].content)
        else:
            log.error("Expected tool call but none were executed")


def scenario_2_blocked_tool(all_tools):
    """Scenario 2: blocked tool call (search_web not on allow list)."""
    with start_session(model_id="llama3.1", plugins=[tool_security]) as m:
        result = m.instruct(
            description="Search the web for the latest travel news.",
            requirements=[uses_tool("search_web")],
            model_options={ModelOption.TOOLS: all_tools},
            tool_calls=True,
        )
        tool_outputs = _call_tools(result, m.backend)
        if not tool_outputs:
            log.info("Tool call was blocked — outputs list is empty, as expected")
        else:
            log.warning("Expected tool to be blocked but it executed: %s", tool_outputs)


# ---------------------------------------------------------------------------
# Main — scenarios
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("--- Tool hook plugins example ---")
    log.info("")

    all_tools = [
        get_attractions,
        get_emergency_services,
        get_weather_forecast,
        get_local_time,
        convert_currency,
        translate_phrase,
        get_local_customs_and_etiquette,
        find_travel_blogs,
    ]

    _run_scenario(
        "Scenario 1: allowed tool — get_attractions",
        lambda: scenario_1_allowed_tool(all_tools),
    )
    _run_scenario(
        "Scenario 2: blocked tool — search_web not on allow list",
        lambda: scenario_2_blocked_tool(all_tools),
    )
