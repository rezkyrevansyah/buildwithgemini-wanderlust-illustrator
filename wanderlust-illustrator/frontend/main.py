"""Minimal FastAPI proxy for a deployed A2A agent (Agent Runtime, agents-cli 1.1.0+).

The browser talks ONLY to this proxy (same origin, no CORS, no GCP creds in the
browser). The proxy authenticates with Application Default Credentials and
forwards chat to the deployed agent over the A2A protocol, returning replies as
structured parts the chat UI knows how to show:

  * {"kind": "text", "text": ...}  -> a normal chat bubble
  * {"kind": "a2ui", "data": ...}  -> one A2UI message (beginRendering /
    surfaceUpdate); static/index.html renders these as a card.

Why A2A: agents-cli 1.1.0 (GA) deploys ADK agents to Agent Runtime as A2A agents
and no longer registers the reasoning-engine operation schema the old
`agent_engines.get(...).stream_query()` path relied on (operation_schemas() comes
back empty). The container serves the A2A protocol over the Agent Engine HTTP
passthrough, so this proxy fetches the agent's card and sends messages with the
a2a-sdk client (the same path `agents-cli run --mode a2a` uses). This works for
both A2A and plain ADK 1.1.0 deployments (the container serves A2A either way).

Run:
  pip install -r requirements.txt
  export AGENT_ENGINE_RESOURCE_NAME="projects/.../locations/.../reasoningEngines/..."
  export AGENT_DIRECTORY="app"   # your agent's app directory (agents-cli-manifest.yaml)
  python main.py                 # -> http://localhost:8080
"""

import json
import os
import re
import uuid

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import google.auth
import google.auth.transport.requests
import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.client import ClientConfig, ClientFactory
from a2a.types import (
    AgentCard,
    FilePart,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TextPart,
    TransportProtocol,
)
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

RESOURCE = os.environ.get(
    "AGENT_ENGINE_RESOURCE_NAME",
    "projects/494495082021/locations/us-east1/reasoningEngines/2823170926663696384",
)
# The agent's app directory (matches agent_directory in agents-cli-manifest.yaml).
AGENT_DIRECTORY = os.environ.get("AGENT_DIRECTORY", "app")
# Location is embedded in the resource name: projects/<p>/locations/<loc>/reasoningEngines/<id>.
LOCATION = RESOURCE.split("/locations/")[1].split("/")[0]

# A2A endpoint for an Agent Runtime deployment, via the Agent Engine HTTP
# passthrough. The card lives at the well-known path under this base.
A2A_BASE = (
    f"https://{LOCATION}-aiplatform.googleapis.com/reasoningEngines/v1/"
    f"{RESOURCE}/api/a2a/{AGENT_DIRECTORY}"
)
A2A_CARD_URL = f"{A2A_BASE}/.well-known/agent-card.json"

# The agent tags its A2UI data parts with this mime type.
_A2UI_MIME = "application/json+a2ui"

# ADC credentials, initialized lazily and refreshed per request (access tokens expire ~1h).
_creds = None


def _get_creds():
    global _creds
    if _creds is None:
        _creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    return _creds


def _auth_headers() -> dict[str, str]:
    creds = _get_creds()
    creds.refresh(google.auth.transport.requests.Request())
    return {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
    }


app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.exception_handler(Exception)
async def _json_errors(request: Request, exc: Exception):
    # Always return JSON so the browser never receives a plain-text 500 page
    # (which shows up in the chat as "Unexpected token 'I', "Internal S"... is
    # not valid JSON"). Any server-side failure now surfaces as a readable
    # message in the chat bubble instead.
    return JSONResponse(
        status_code=200,
        content={
            "parts": [{"kind": "text", "text": f"Error: {type(exc).__name__}: {exc}"}]
        },
    )


# Reuse ONE A2A context per user so the agent remembers the conversation.
_contexts: dict[str, str] = {}
# Cache the agent card after the first fetch.
_card: AgentCard | None = None


async def _get_card(client: httpx.AsyncClient) -> AgentCard:
    global _card
    if _card is None:
        resp = await client.get(A2A_CARD_URL)
        resp.raise_for_status()
        card = AgentCard(**resp.json())
        # Agent Runtime does not serve a public card URL, so point the client at
        # the passthrough base for message sends.
        card.url = A2A_BASE
        _card = card
    return _card


def _extract_parts(parts: list) -> list[dict]:
    """Turn A2A response parts into structured parts for the chat UI.

    Text parts pass through as {"kind": "text"}. A2UI data parts (tagged
    application/json+a2ui) become {"kind": "a2ui", "data": <message>} so the UI
    renders the card; each data part is one A2UI message (beginRendering or
    surfaceUpdate).
    """
    out: list[dict] = []
    for p in parts:
        root = getattr(p, "root", p)
        if isinstance(root, TextPart) and getattr(root, "text", None):
            txt = root.text
            # If the model or runtime wrapped A2UI into <a2a_datapart_json> inside a text part
            if "<a2a_datapart_json>" in txt:
                matches = re.findall(r"<a2a_datapart_json>(.*?)</a2a_datapart_json>", txt, re.DOTALL)
                for m in matches:
                    try:
                        parsed = json.loads(m.strip())
                        data_body = parsed.get("data") if isinstance(parsed, dict) else parsed
                        out.append({"kind": "a2ui", "data": data_body})
                    except Exception:
                        pass
                clean_txt = re.sub(r"<a2a_datapart_json>.*?</a2a_datapart_json>", "", txt, flags=re.DOTALL).strip()
                if clean_txt:
                    out.append({"kind": "text", "text": clean_txt})
            else:
                out.append({"kind": "text", "text": txt})
        elif getattr(root, "data", None) is not None:
            raw_data = root.data
            meta = getattr(root, "metadata", None) or {}
            # Handle nested {"kind": "data", "metadata": {"mimeType": ...}, "data": ...}
            if (
                isinstance(raw_data, dict)
                and "data" in raw_data
                and "metadata" in raw_data
            ):
                meta = raw_data.get("metadata") or {}
                raw_data = raw_data.get("data")

            mime = meta.get("mimeType") if isinstance(meta, dict) else None
            if mime == _A2UI_MIME:
                out.append({"kind": "a2ui", "data": raw_data})
        elif isinstance(root, FilePart):
            uri = getattr(getattr(root, "file", None), "uri", None)
            if uri:
                out.append({"kind": "text", "text": uri})
    return out


@app.post("/chat")
async def chat(req: Request):
    try:
        body = await req.json()
    except Exception:
        body = {}
    message = body.get("message", "")
    user_id = body.get("user_id") or "web-user"
    parts: list[dict] = []

    async with httpx.AsyncClient(headers=_auth_headers(), timeout=120) as client:
        card = await _get_card(client)
        factory = ClientFactory(
            ClientConfig(
                supported_transports=[
                    TransportProtocol.jsonrpc,
                    TransportProtocol.http_json,
                ],
                httpx_client=client,
            )
        )
        a2a_client = factory.create(card)

        msg = Message(
            message_id=str(uuid.uuid4()),
            role=Role.user,
            parts=[Part(root=TextPart(text=message))],
            context_id=_contexts.get(user_id),
        )

        last_task = None
        got_artifact_update = False
        async for event in a2a_client.send_message(msg):
            if not isinstance(event, tuple):
                continue
            task, update = event
            if task is not None:
                last_task = task
                if getattr(task, "context_id", None):
                    _contexts[user_id] = task.context_id
            if isinstance(update, TaskArtifactUpdateEvent):
                got_artifact_update = True
                parts.extend(_extract_parts(update.artifact.parts))
            elif hasattr(update, "status"):
                status_obj = getattr(update, "status", None)
                msg_obj = getattr(status_obj, "message", None)
                if msg_obj and getattr(msg_obj, "parts", None):
                    parts.extend(_extract_parts(msg_obj.parts))

        # Fallback: pull parts from final task artifacts
        if not got_artifact_update and last_task is not None:
            for artifact in getattr(last_task, "artifacts", None) or []:
                parts.extend(_extract_parts(artifact.parts))

        # Fallback: check status message or history on last_task
        if not parts and last_task is not None:
            status_obj = getattr(last_task, "status", None)
            msg_obj = getattr(status_obj, "message", None)
            if msg_obj and getattr(msg_obj, "parts", None):
                parts.extend(_extract_parts(msg_obj.parts))
            for hist in getattr(last_task, "history", None) or []:
                if getattr(hist, "role", None) in (Role.model, "model"):
                    parts.extend(_extract_parts(getattr(hist, "parts", None) or []))

    if not parts:
        parts = [{"kind": "text", "text": "(The agent didn't return a reply.)"}]
    return JSONResponse({"parts": parts})


# -------------------------------------------------------------
# WORLDWIDE DESTINATIONS, GEOCODING, & CHAT HISTORY APIS
# -------------------------------------------------------------
_user_histories: dict[str, list[dict]] = {}


def _flag_from_code(code: str) -> str:
    if not code or len(code) != 2:
        return "🌍"
    try:
        return "".join(chr(127397 + ord(c.upper())) for c in code)
    except Exception:
        return "🌍"


@app.get("/api/cities/search")
async def search_cities(q: str):
    """Search any city in the world using Open-Meteo Geocoding + Wikipedia summary."""
    query = (q or "").strip()
    if not query:
        return JSONResponse({"results": []})

    async with httpx.AsyncClient(timeout=8) as client:
        # 1. Open-Meteo Worldwide Geocoding API (free, open, no key)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={httpx.URL(query).raw_path.decode()}&count=6&language=en&format=json"
        try:
            resp = await client.get(geo_url, headers={"User-Agent": "WanderlustApp/1.0"})
            geo_data = resp.json()
        except Exception as e:
            return JSONResponse({"results": [], "error": str(e)})

        raw_results = geo_data.get("results") or []
        results = []

        for item in raw_results:
            name = item.get("name", "")
            country = item.get("country", "")
            country_code = item.get("country_code", "")
            flag = _flag_from_code(country_code)
            admin1 = item.get("admin1", "")
            lat = item.get("latitude")
            lon = item.get("longitude")
            pop = item.get("population")
            tz = item.get("timezone", "UTC")

            results.append({
                "name": name,
                "country": country,
                "country_code": country_code,
                "flag": flag,
                "admin1": admin1,
                "display_name": f"{flag} {name}, {country}" if country else f"{flag} {name}",
                "latitude": lat,
                "longitude": lon,
                "population": pop,
                "timezone": tz,
            })

        # 2. Enrich top result with Wikipedia summary and thumbnail photo
        if results:
            top_name = results[0]["name"]
            try:
                wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{httpx.URL(top_name).raw_path.decode()}"
                w_resp = await client.get(wiki_url, headers={"User-Agent": "WanderlustApp/1.0"})
                if w_resp.status_code == 200:
                    w_data = w_resp.json()
                    results[0]["extract"] = w_data.get("extract", "")
                    if w_data.get("thumbnail") and w_data["thumbnail"].get("source"):
                        results[0]["thumbnail"] = w_data["thumbnail"]["source"]
            except Exception:
                pass

        return JSONResponse({"results": results})


@app.get("/api/destinations/worldwide")
async def get_worldwide_destinations():
    """Curated list of 25+ iconic destinations across all continents."""
    destinations = [
        # Asia
        {"name": "Kyoto", "country": "Japan", "continent": "Asia", "flag": "🇯🇵", "vibe": "cultural", "budget": "moderate", "season": "Spring (Mar-May) / Autumn", "highlights": ["Fushimi Inari Shrine", "Arashiyama Bamboo Grove", "Gion Geisha District"]},
        {"name": "Tokyo", "country": "Japan", "continent": "Asia", "flag": "🇯🇵", "vibe": "modern", "budget": "moderate", "season": "Spring & Autumn", "highlights": ["Shibuya Crossing", "Shinjuku Nightlife", "Sensō-ji Temple"]},
        {"name": "Bali", "country": "Indonesia", "continent": "Asia", "flag": "🇮🇩", "vibe": "tropical", "budget": "budget", "season": "Apr-Oct (Dry season)", "highlights": ["Ubud Rice Terraces", "Uluwatu Temple Sunset", "Canggu Cafes"]},
        {"name": "Yogyakarta", "country": "Indonesia", "continent": "Asia", "flag": "🇮🇩", "vibe": "cultural", "budget": "budget", "season": "May-Sep", "highlights": ["Borobudur Sunrise", "Prambanan Temple", "Keraton Palace"]},
        {"name": "Singapore", "country": "Singapore", "continent": "Asia", "flag": "🇸🇬", "vibe": "futuristic", "budget": "luxury", "season": "Year-round", "highlights": ["Gardens by the Bay", "Marina Bay Sands", "Chinatown Food"]},
        {"name": "Bangkok", "country": "Thailand", "continent": "Asia", "flag": "🇹🇭", "vibe": "foodie", "budget": "budget", "season": "Nov-Feb", "highlights": ["Grand Palace", "Wat Arun Sunset", "Street Food Night Markets"]},
        {"name": "Seoul", "country": "South Korea", "continent": "Asia", "flag": "🇰🇷", "vibe": "cultural", "budget": "moderate", "season": "Sep-Nov", "highlights": ["Gyeongbokgung Palace", "Bukchon Hanok Village", "Myeongdong"]},
        {"name": "Dubai", "country": "UAE", "continent": "Asia", "flag": "🇦🇪", "vibe": "luxury", "budget": "luxury", "season": "Nov-Mar", "highlights": ["Burj Khalifa", "Desert Safari Dune Bashing", "Palm Jumeirah"]},
        # Europe
        {"name": "Amalfi Coast", "country": "Italy", "continent": "Europe", "flag": "🇮🇹", "vibe": "romantic", "budget": "luxury", "season": "May-Sep", "highlights": ["Positano Cliffside", "Capri Boat Tour", "Ravello Gardens"]},
        {"name": "Rome", "country": "Italy", "continent": "Europe", "flag": "🇮🇹", "vibe": "historic", "budget": "moderate", "season": "Apr-Jun / Sep-Oct", "highlights": ["Colosseum", "Vatican & Sistine Chapel", "Trevi Fountain"]},
        {"name": "Paris", "country": "France", "continent": "Europe", "flag": "🇫🇷", "vibe": "romantic", "budget": "moderate", "season": "Apr-Oct", "highlights": ["Eiffel Tower", "Louvre Museum", "Montmartre & Sacré-Cœur"]},
        {"name": "Santorini", "country": "Greece", "continent": "Europe", "flag": "🇬🇷", "vibe": "scenic", "budget": "luxury", "season": "May-Oct", "highlights": ["Oia Sunset Caldera", "Red Beach", "White & Blue Domes"]},
        {"name": "Barcelona", "country": "Spain", "continent": "Europe", "flag": "🇪🇸", "vibe": "coastal", "budget": "moderate", "season": "May-Oct", "highlights": ["Sagrada Família", "Park Güell", "Barceloneta Beach"]},
        {"name": "Swiss Alps (Zermatt)", "country": "Switzerland", "continent": "Europe", "flag": "🇨🇭", "vibe": "nature", "budget": "luxury", "season": "Dec-Apr (Snow) / Jun-Sep (Hike)", "highlights": ["Matterhorn Views", "Gornergrat Railway", "Alpine Ski Slopes"]},
        {"name": "Reykjavik", "country": "Iceland", "continent": "Europe", "flag": "🇮🇸", "vibe": "adventure", "budget": "luxury", "season": "Sep-Mar (Aurora) / Jun-Aug", "highlights": ["Northern Lights", "Blue Lagoon Geothermal Spa", "Golden Circle Waterfalls"]},
        {"name": "London", "country": "UK", "continent": "Europe", "flag": "🇬🇧", "vibe": "historic", "budget": "moderate", "season": "May-Sep", "highlights": ["Big Ben & Westminster", "British Museum", "Tower Bridge"]},
        # Americas
        {"name": "Banff & Lake Louise", "country": "Canada", "continent": "Americas", "flag": "🇨🇦", "vibe": "nature", "budget": "moderate", "season": "Jun-Aug / Dec-Mar", "highlights": ["Turquoise Lake Louise", "Moraine Lake", "Icefields Parkway"]},
        {"name": "New York City", "country": "USA", "continent": "Americas", "flag": "🇺🇸", "vibe": "urban", "budget": "luxury", "season": "Sep-Dec", "highlights": ["Central Park", "Times Square & Broadway", "Empire State Building"]},
        {"name": "Cusco & Machu Picchu", "country": "Peru", "continent": "Americas", "flag": "🇵🇪", "vibe": "adventure", "budget": "moderate", "season": "May-Oct (Dry season)", "highlights": ["Inca Citadel", "Sacred Valley", "Rainbow Mountain"]},
        {"name": "Rio de Janeiro", "country": "Brazil", "continent": "Americas", "flag": "🇧🇷", "vibe": "coastal", "budget": "moderate", "season": "Dec-Mar", "highlights": ["Christ the Redeemer", "Copacabana Beach", "Sugarloaf Mountain"]},
        {"name": "Maui (Hawaii)", "country": "USA", "continent": "Americas", "flag": "🇺🇸", "vibe": "tropical", "budget": "luxury", "season": "Year-round", "highlights": ["Road to Hana", "Haleakala Sunrise", "Kaanapali Snorkeling"]},
        # Africa & Oceania
        {"name": "Cairo", "country": "Egypt", "continent": "Africa", "flag": "🇪🇬", "vibe": "historic", "budget": "budget", "season": "Oct-Apr", "highlights": ["Great Pyramids of Giza", "The Sphinx", "Grand Egyptian Museum"]},
        {"name": "Cape Town", "country": "South Africa", "continent": "Africa", "flag": "🇿🇦", "vibe": "scenic", "budget": "moderate", "season": "Nov-Mar", "highlights": ["Table Mountain Cableway", "Cape Point", "Boulders Penguin Beach"]},
        {"name": "Sydney", "country": "Australia", "continent": "Oceania", "flag": "🇦🇺", "vibe": "coastal", "budget": "moderate", "season": "Sep-Nov / Mar-May", "highlights": ["Sydney Opera House", "Bondi to Coogee Walk", "Harbour Bridge Climb"]},
        {"name": "Queenstown", "country": "New Zealand", "continent": "Oceania", "flag": "🇳🇿", "vibe": "adventure", "budget": "moderate", "season": "Dec-Feb (Summer) / Jun-Aug", "highlights": ["Milford Sound Cruise", "Bungy Jumping", "Lake Wakatipu Skyline"]},
    ]
    return JSONResponse({"destinations": destinations})


@app.get("/api/history/{user_id}")
async def get_history(user_id: str):
    return JSONResponse({"messages": _user_histories.get(user_id, [])})


@app.post("/api/history/{user_id}")
async def save_history(user_id: str, req: Request):
    data = await req.json()
    messages = data.get("messages", [])
    _user_histories[user_id] = messages
    return JSONResponse({"status": "saved", "count": len(messages)})


@app.delete("/api/history/{user_id}")
async def clear_history(user_id: str):
    if user_id in _user_histories:
        del _user_histories[user_id]
    if user_id in _contexts:
        del _contexts[user_id]
    return JSONResponse({"status": "cleared"})


# Serve the chat UI (keep this mount last so /chat and /api win).
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
