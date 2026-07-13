import asyncio
import logging

from mellea.stdlib.context import ChatContext
from mellea.stdlib.frameworks.react import react
from mellea.stdlib.session import start_session
from tourist_assistant_tools import (
    convert_currency,
    get_attractions,
    get_emergency_services,
    get_local_customs_and_etiquette,
    get_local_time,
    get_weather_forecast,
    notify_emergency_contact,
    translate_phrase,
)
from tourist_assistant_hooks import (
    user_input_safety, 
    tool_security, 
    tool_sanitizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("agent")

# ---------------------------------------------------------------------------
# Tools
#
# @tool already returns MelleaTool instances — pass them directly to react().
# ---------------------------------------------------------------------------

tools = [
    get_attractions,
    get_emergency_services,
    get_weather_forecast,
    get_local_time,
    convert_currency,
    translate_phrase,
    get_local_customs_and_etiquette,
    notify_emergency_contact,
    tool_security,
]


# ---------------------------------------------------------------------------
# Agent goals to test
# ---------------------------------------------------------------------------

GOALS = [
    # Normal informational query — should use get_weather_forecast and get_attractions
    "What's the weather going to be like in Kyoto for the next 3 days, and what "
    "are some good attractions to visit while I'm there?",

    # Multi-tool practical query — should use convert_currency and get_emergency_services
    "I need to convert 200 USD to Japanese yen, and find the nearest pharmacy near Shibuya, Tokyo.",

    # Distress trigger — detect_travel_distress hook should fire before the LLM sees this,
    # send dummy emails to the emergency contact and embassy, and block the request.
    "Someone just stole my bag near the Colosseum and I don't know what to do, I'm scared.",
]


async def run_goal(goals: list[str], username: str = "demo_user") -> None:

    # agent_plugins includes:
    #   - detect_travel_distress  (GENERATION_PRE_CALL, priority 1)
    #   - enforce_tool_allowlist  (TOOL_PRE_INVOKE)
    #   - redact_pii              (TOOL_POST_INVOKE)
    #   - audit_tool_calls        (TOOL_POST_INVOKE, fire-and-forget)
    with start_session(model_id="llama3.1", ctx=ChatContext(), plugins=[tool_sanitizer, user_input_safety]) as m:
        for goal in goals:
            log.info("=" * 60)
            log.info("GOAL: %s", goal)
            log.info("=" * 60)

            try:
                out, _ = await react(
                    goal=goal,
                    context=m.ctx,
                    backend=m.backend,
                    tools=tools,
                    loop_budget=2,
                )
                log.info("RESPONSE:\n%s", out)
            except Exception as exc:
                # PluginViolationError surfaces here if a hook blocks the request.
                log.warning("Request blocked or failed: %s", exc)

            log.info("")


async def main() -> None:
    await run_goal(GOALS)


if __name__ == "__main__":
    asyncio.run(main())
