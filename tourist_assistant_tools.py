# pytest: ollama, e2e
#
# Tool hook plugins for the example tourist information assistant agent. This example includes:
# - A tool allow list plugin that blocks any tool calls for tools not explicitly allowed.
# - A tool output sanitizer that redacts potential PII (passport numbers, credit cards,
#   emails, phone numbers) from tool outputs before they enter context.
# - A tool audit logger that records every tool call outcome for auditing purposes.
# - A travel-distress message detector that blocks and triggers notifications if a user
#   message indicates they may be in danger, robbed, or lost (e.g. lost passport, theft).
#
# Unlike the mental-health example, most tools here call real, free, key-less public
# APIs (OpenStreetMap Nominatim/Overpass, Open-Meteo, Frankfurter, MyMemory) instead of
# returning canned data, so they behave like a real production tourist assistant would.
#
# Run:
#   uv run python tourist_assistant_tools.py

from dataclasses import dataclass
import logging
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

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
# Shared helpers — real geocoding / POI lookups against OpenStreetMap
# ---------------------------------------------------------------------------

def _geocode_city(city: str) -> tuple[float, float, str] | None:
    """Resolve a city name to (lat, lon, display_name) via OpenStreetMap Nominatim."""
    try:
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "json", "limit": 1},
            headers={"User-Agent": _USER_AGENT},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        top = results[0]
        return float(top["lat"]), float(top["lon"]), top.get("display_name", city)
    except (httpx.HTTPError, KeyError, ValueError, IndexError) as exc:
        log.error("[geocode] failed for city=%r: %s", city, exc)
        return None


_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

_POI_CATEGORY_TAGS: dict[str, str] = {
    "attractions": '["tourism"~"attraction|museum|viewpoint|artwork|gallery"]',
    "hospital": '["amenity"="hospital"]',
    "police": '["amenity"="police"]',
    "embassy": '["diplomatic"="embassy"]',
    "pharmacy": '["amenity"="pharmacy"]',
    "restaurant": '["amenity"="restaurant"]',
    "atm": '["amenity"="atm"]',
}


def _find_nearby_places(city: str, category: str, radius_km: float, max_results: int) -> str:
    """Query OpenStreetMap Overpass for real POIs of `category` near `city`."""
    geo = _geocode_city(city)
    if geo is None:
        return f"Could not find location data for '{city}'."
    lat, lon, display_name = geo

    tag_filter = _POI_CATEGORY_TAGS.get(category.strip().lower())
    if tag_filter is None:
        known = ", ".join(_POI_CATEGORY_TAGS)
        return f"Unknown category '{category}'. Known categories: {known}."

    radius_m = int(max(0.1, radius_km) * 1000)
    fetch_limit = min(max_results * 4, 50)
    query = (
        f"[out:json][timeout:25];"
        f"(node{tag_filter}(around:{radius_m},{lat},{lon});"
        f"way{tag_filter}(around:{radius_m},{lat},{lon}););"
        f"out center {fetch_limit};"
    )
    try:
        resp = httpx.post(
            _OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": _USER_AGENT},
            timeout=25.0,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except (httpx.HTTPError, ValueError) as exc:
        log.error("[places] Overpass query failed for city=%r category=%r: %s", city, category, exc)
        return f"Place search failed for '{city}': {exc}"

    named = [e for e in elements if e.get("tags", {}).get("name")]
    if not named:
        return f"No '{category}' results found within {radius_km:.1f}km of {display_name}."

    lines = [f"{category.title()} near {display_name} (within {radius_km:.1f}km):"]
    for e in named[:max_results]:
        tags = e["tags"]
        name = tags["name"]
        addr_parts = [tags.get(k) for k in ("addr:housenumber", "addr:street", "addr:city")]
        addr = " ".join(p for p in addr_parts if p)
        lines.append(f"  • {name}" + (f" — {addr}" if addr else ""))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_attractions(city: str, radius_km: float = 5.0, max_results: int = 5) -> str:
    """Find real tourist attractions (museums, viewpoints, galleries, landmarks) near a city,
    using live OpenStreetMap data.

    Args:
        city: City to search near (e.g. 'Kyoto', 'Paris', 'Rome').
        radius_km: Search radius in kilometers from the city center.
        max_results: Maximum number of results to return.
    """
    return _find_nearby_places(city, "attractions", radius_km, max_results)


@tool
def get_emergency_services(city: str, category: str = "hospital", radius_km: float = 5.0, max_results: int = 5) -> str:
    """Find real nearby emergency or essential services using live OpenStreetMap data.

    Args:
        city: City to search near (e.g. 'Kyoto', 'Paris', 'Rome').
        category: One of: hospital, police, embassy, pharmacy, restaurant, atm.
        radius_km: Search radius in kilometers from the city center.
        max_results: Maximum number of results to return.
    """
    return _find_nearby_places(city, category, radius_km, max_results)


@tool
def get_weather_forecast(city: str, days: int = 3) -> str:
    """Get a live multi-day weather forecast for a city.

    Args:
        city: City name to look up (e.g. 'Kyoto', 'Paris').
        days: Number of forecast days to return (1-7).
    """
    days = max(1, min(days, 7))
    geo = _geocode_city(city)
    if geo is None:
        return f"Could not find location data for '{city}'."
    lat, lon, display_name = geo
    try:
        resp = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": days,
            },
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        daily = resp.json()["daily"]
        lines = [f"Forecast for {display_name}:"]
        for i, date in enumerate(daily["time"]):
            lines.append(
                f"  {date}: {daily['temperature_2m_min'][i]:.0f}-{daily['temperature_2m_max'][i]:.0f}°C, "
                f"{daily['precipitation_probability_max'][i]}% chance of rain"
            )
        return "\n".join(lines)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.error("[weather] failed for city=%r: %s", city, exc)
        return f"Weather lookup failed for '{city}': {exc}"


@tool
def get_local_time(city: str) -> str:
    """Get the current local date and time for a city.

    Args:
        city: City name to look up (e.g. 'Tokyo', 'New York').
    """
    geo = _geocode_city(city)
    if geo is None:
        return f"Could not find location data for '{city}'."
    lat, lon, display_name = geo
    try:
        resp = httpx.get(
            "https://timeapi.io/api/time/current/coordinate",
            params={"latitude": lat, "longitude": lon},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return f"Local time in {display_name}: {data['dateTime']} ({data.get('timeZone', 'unknown timezone')})"
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.error("[local_time] failed for city=%r: %s", city, exc)
        return f"Local time lookup failed for '{city}': {exc}"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies using live ECB exchange rates.

    Args:
        amount: Amount to convert.
        from_currency: Source currency code (e.g. 'USD', 'EUR', 'JPY').
        to_currency: Target currency code (e.g. 'USD', 'EUR', 'JPY').
    """
    src = from_currency.strip().upper()
    dst = to_currency.strip().upper()
    try:
        resp = httpx.get(
            "https://api.frankfurter.app/latest",
            params={"amount": amount, "from": src, "to": dst},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        converted = data["rates"].get(dst)
        if converted is None:
            return f"Could not convert {src} to {dst}. Check that both are valid currency codes."
        return f"{amount:.2f} {src} = {converted:.2f} {dst} (rate date: {data['date']})"
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.error("[currency] conversion failed %s->%s: %s", src, dst, exc)
        return f"Currency conversion failed: {exc}"


@tool
def translate_phrase(phrase: str, target_language: str, source_language: str = "en") -> str:
    """Translate a short phrase for use while traveling, using a live translation service.

    Args:
        phrase: The phrase to translate.
        target_language: Target language code (e.g. 'ja', 'fr', 'it', 'es').
        source_language: Source language code of the phrase (default 'en').
    """
    try:
        resp = httpx.get(
            "https://api.mymemory.translated.net/get",
            params={"q": phrase, "langpair": f"{source_language}|{target_language}"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        translated = resp.json()["responseData"]["translatedText"]
        return f"'{phrase}' -> '{translated}' ({source_language} -> {target_language})"
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.error("[translate] failed phrase=%r: %s", phrase, exc)
        return f"Translation failed: {exc}"


@tool
def search_web(query: str, max_results: int = 5) -> list[str]:
    """Search the web for information.

    Args:
        query: Search query
        max_results: Maximum number of results to return
    """
    return [f"Result {i + 1} for '{query}'" for i in range(max_results)]


_DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"

_DDG_RESULT_PATTERN = re.compile(
    r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_HTML_TAG_PATTERN = re.compile(r"<.*?>")


def _clean_ddg_redirect_url(url: str) -> str:
    """DuckDuckGo's HTML results wrap outbound links in a `/l/?uddg=...` redirect; unwrap it."""
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        if target:
            return unquote(target)
    return url


@tool
def find_travel_blogs(place: str, max_results: int = 5) -> str:
    """Search the live web for real travel blog posts and itineraries about a place.

    Args:
        place: A city, region, or country to find travel blogs about (e.g. 'Kyoto', 'Tuscany').
        max_results: Maximum number of blog results to return.
    """
    # There's no free official web-search API, so this scrapes DuckDuckGo's
    # key-less HTML results endpoint. Fragile by nature — their markup can change.
    query = f"{place} travel blog itinerary"
    try:
        resp = httpx.post(
            _DUCKDUCKGO_HTML_URL,
            data={"q": query},
            headers={"User-Agent": _USER_AGENT},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.error("[travel_blogs] search failed for place=%r: %s", place, exc)
        return f"Travel blog search failed for '{place}': {exc}"

    matches = _DDG_RESULT_PATTERN.findall(resp.text)
    if not matches:
        return f"No travel blog results found for '{place}'."

    lines = [f"Travel blogs and itineraries for {place}:"]
    for url, title in matches[:max_results]:
        clean_title = _HTML_TAG_PATTERN.sub("", title).strip()
        clean_url = _clean_ddg_redirect_url(url)
        lines.append(f"  • {clean_title}\n    {clean_url}")
    return "\n".join(lines)


_CUSTOMS_AND_ETIQUETTE: dict[str, list[str]] = {
    "japan": [
        "Remove shoes before entering homes, temples, and some restaurants.",
        "Tipping is not customary and can be considered rude.",
        "Avoid eating or drinking while walking in public.",
        "Bowing is a common greeting; a slight nod is fine for visitors.",
    ],
    "france": [
        "Always greet with 'Bonjour' before starting a conversation, even in shops.",
        "Tipping is not required — a service charge is usually included.",
        "Keep your voice down in restaurants and on public transport.",
    ],
    "italy": [
        "Cover shoulders and knees when visiting churches.",
        "Cappuccino is typically a morning drink, not ordered after meals.",
        "Tipping is appreciated but not mandatory; rounding up is common.",
    ],
}

@tool
def get_local_customs_and_etiquette(country: str) -> str:
    """Return a list of cultural customs and etiquette tips for a country.

    Args:
        country: Country to look up customs for (e.g. 'Japan', 'France', 'Italy').
    """
    key = country.strip().lower()
    tips = _CUSTOMS_AND_ETIQUETTE.get(key)
    if not tips:
        return (
            f"No customs guide available for '{country}' yet. "
            "General advice: observe local dress norms, learn basic greetings, "
            "and research tipping customs before you go."
        )
    return "\n".join(f"• {tip}" for tip in tips)


@dataclass(frozen=True)
class TrustedContact:
    contact_email: str
    contact_name: str
    home_embassy_email: str


# Stub — replace with a real DB lookup (see user_system_information.py).
_TRUSTED_CONTACTS: dict[str, TrustedContact] = {
    "demo_user": TrustedContact(
        contact_email="family@example.com",
        contact_name="Alex (emergency contact)",
        home_embassy_email="embassy@example.gov",
    ),
}


def get_trusted_contact(username: str) -> TrustedContact | None:
    return _TRUSTED_CONTACTS.get(username.strip().lower())


def _send_email(to_email: str, subject: str, body: str) -> None:
    """Stub: prints the email instead of sending it. Swap for a real mail/SMS provider."""
    print(
        f"\n[DUMMY EMAIL]\n"
        f"  To:      {to_email}\n"
        f"  Subject: {subject}\n"
        f"  Body:\n{body}\n"
        f"[END EMAIL]\n"
    )
    log.info("[dummy-smtp] would have sent email to=%r subject=%r", to_email, subject)


def _run_notification_obligation(username: str, trigger_message: str) -> str:
    contact = get_trusted_contact(username)
    if contact is None:
        log.warning("[obligation] no trusted contact for username=%r", username)
        return f"No trusted contact on file for '{username}'. No emails sent."

    subject = "Important: travel emergency alert"
    contact_body = (
        f"Dear {contact.contact_name},\n\n"
        f"{username} may be in an emergency situation while traveling and could "
        f"benefit from your support.\n\n"
        f"Message flagged: \"{trigger_message}\"\n\n"
        f"If you believe they are in immediate danger, contact local emergency "
        f"services at their destination or the nearest embassy.\n\n"
        f"— The Travel Safety Team"
    )
    embassy_body = (
        f"An automated alert has been raised for traveler '{username}':\n\n"
        f"  \"{trigger_message}\"\n\n"
        f"Please advise on next steps for consular assistance.\n\n"
        f"— Automated Alert System"
    )

    results = []
    for label, to_email, body in [
        ("emergency contact", contact.contact_email, contact_body),
        ("embassy", contact.home_embassy_email, embassy_body),
    ]:
        try:
            _send_email(to_email, subject, body)
            results.append(f"{label} notified at {to_email}")
        except Exception as exc:
            log.error("[obligation] failed to email %s=%r: %s", label, to_email, exc)
            results.append(f"{label} notification FAILED: {exc}")

    return "; ".join(results)

@tool
def notify_emergency_contact(username: str, reason: str) -> str:
    """Send an alert email to the traveler's emergency contact and home embassy.
    Use when the conversation warrants escalation beyond what has already been
    sent automatically.

    Args:
        username: The user's account name (must match the trusted contacts registry).
        reason: Plain-language description of the emergency or concern.
    """
    return _run_notification_obligation(username, reason)


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

# ---------------------------------------------------------------------------
# Compose into PluginSets for clean session-scoped registration
# ---------------------------------------------------------------------------

user_input_safety = PluginSet("user-input-safety", [detect_travel_distress])

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
        notify_emergency_contact,
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
