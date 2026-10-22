# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import datetime
import os
import uuid
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.code_executors import AgentEngineSandboxCodeExecutor
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.cloud import firestore, storage
from google.genai import types

from .a2ui_utils import a2ui_callback

try:
    from a2ui.schema.manager import A2uiSchemaManager
    from a2ui.basic_catalog.provider import BasicCatalog
except ImportError:
    from a2ui.core.basic_catalog.v0_8 import BasicCatalog as _BC08
    from a2ui.inference_formats.direct_json import DirectJsonPromptGenerator

    class BasicCatalog:
        @staticmethod
        def get_config(version: str = "0.8"):
            return _BC08()

    class A2uiSchemaManager:
        def __init__(self, version: str = "0.8", catalogs=None):
            self.version = version
            self.catalogs = catalogs or [BasicCatalog.get_config(version)]

        def generate_system_prompt(
            self,
            role_description: str = "",
            workflow_description: str = "",
            ui_description: str = "",
            include_schema: bool = True,
            include_examples: bool = True,
        ) -> str:
            gen = DirectJsonPromptGenerator(catalogs=self.catalogs)
            base_prompt = gen.generate()
            parts = []
            if role_description:
                parts.append(f"## Role Description:\n{role_description}")
            if workflow_description:
                parts.append(f"## Workflow Description:\n{workflow_description}")
            if ui_description:
                parts.append(f"## UI Description & Guidelines:\n{ui_description}")
            parts.append(base_prompt)
            return "\n\n".join(parts)


MODEL = os.getenv("MODEL", "gemini-2.5-flash")
GCP_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-42d301455a69")
COLLECTION_NAME = os.getenv("FIRESTORE_COLLECTION", "destinations")
BUCKET_NAME = os.getenv(
    "BUCKET_NAME", "wanderlust-gallery-qwiklabs-gcp-04-42d301455a69"
)
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
AGENT_ENGINE_ID = os.getenv(
    "AGENT_ENGINE_ID",
    "projects/494495082021/locations/us-east1/reasoningEngines/1386241170555797504",
)
MEMORY_BANK_ID = os.getenv("MEMORY_BANK_ID", "1386241170555797504")

# A2UI Schema Manager setup (v0.8 with Basic Catalog)
schema_manager = A2uiSchemaManager(
    version="0.8",
    catalogs=[BasicCatalog.get_config("0.8")],
)

a2ui_system_instruction = schema_manager.generate_system_prompt(
    role_description=(
        "You are Wanderlust Illustrator, an expert travel planner agent. "
        "You help users assemble dream itineraries, explore destinations, and manage their travel dreams. "
        "IMPORTANT MEMORY & WISHLIST GUIDELINES: "
        "- Explicitly track and remember every destination, city, country, landmark, and trip idea the user mentions in their travel wishlist or bucket list. "
        "- When the user mentions wanting to visit, liking, or adding a place to their wishlist, acknowledge it and ensure it is remembered across sessions. "
        "- In new conversations, proactively recall and reference places from the user's saved wishlist to tailor itineraries, recommendations, and postcard illustrations. "
        "- Also remember general preferences (budget tier, travel vibe, travel dates, companion details). "
        "Use `list_destinations`, `get_destination`, and `add_or_update_destination` to consult and expand the catalog. "
        "Use `generate_destination_postcard` whenever the user asks to see, illustrate, or create a visual postcard of a destination. "
        "Use `generate_destination_video` whenever the user asks for a video, clip, teaser, cinematic shot, or animated view of a destination. "
        "Use `get_destination_live_weather` for real-time live climate and forecast data. "
        "You can run Python code in your sandbox for trip budgeting, currency conversions, day splits, or math calculations."
    ),
    workflow_description="Analyze the request, call relevant tools to fetch destinations, weather, or postcards, and return structured UI when appropriate.",
    ui_description=(
        "Keep every surface tiny and flat: ONE Card > ONE Column > a few Text rows. "
        "Never nest a Card inside a Card. "
        "Use ONLY these components: Card, Column, Row, Text, and Image. Do not use "
        "Table or Heading (unsupported), or Buttons, actions, or forms (they do "
        "nothing in adk web). "
        "You may include one Image component, but only when you have a public https "
        "URL for the image (for example the URL an image tool returns after uploading "
        "to a public bucket). Set the Image url to that exact https link, for example "
        '{"Image": {"url": {"literalString": "https://..."}}}. Never point an '
        "Image at a bare filename, an artifact name, or a non-http(s) path. If you do "
        "not have a public URL, add a short Text line noting the image instead. "
        "No markdown in text; use the usageHint property ('h1', 'h2', 'body') for "
        "headings and emphasis. "
        "Output ONLY the raw A2UI JSON array — no prose, and never wrap it in "
        "<a2a_datapart_json> tags or 'kind'/'data'/'metadata' objects."
    ),
    include_schema=True,
    include_examples=True,
)


class _LazyClientProxy:
    """Delays client creation until first attribute access so module import never hangs or fails."""

    def __init__(self, factory):
        self._factory = factory
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = self._factory()
        return self._client

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._get_client(), name)


# Clients initialized lazily with GCP project ID
db = _LazyClientProxy(lambda: firestore.Client(project=GCP_PROJECT_ID))
storage_client = _LazyClientProxy(lambda: storage.Client(project=GCP_PROJECT_ID))
genai_client = genai.Client(
    vertexai=True, project=GCP_PROJECT_ID, location=GOOGLE_CLOUD_LOCATION
)

# Agent Platform Sandbox Code Executor
code_executor = AgentEngineSandboxCodeExecutor(
    agent_engine_resource_name=AGENT_ENGINE_ID
)


# Memory generation callback for Vertex AI Memory Bank
async def generate_memories_callback(callback_context: CallbackContext):
    """Write: after each turn, extract and send salient facts/preferences to Memory Bank."""
    await callback_context.add_session_to_memory()
    return None


def list_destinations(
    vibe: Optional[str] = None, budget_level: Optional[str] = None
) -> List[Dict[str, Any]]:
    """List travel destinations from the catalog, optionally filtering by vibe or budget level.

    Args:
        vibe: Optional travel vibe to filter by (e.g. 'cultural', 'romantic', 'adventure', 'nature', 'coastal').
        budget_level: Optional budget tier ('moderate', 'luxury', etc.).

    Returns:
        A list of destination dictionaries with details including name, highlights, season, and recommended days.
    """
    col_ref = db.collection(COLLECTION_NAME)
    docs = col_ref.stream()
    results = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        if vibe and vibe.lower() not in [v.lower() for v in data.get("vibes", [])]:
            continue
        if (
            budget_level
            and budget_level.lower() != data.get("budget_level", "").lower()
        ):
            continue
        results.append(data)
    return results


def get_destination(destination_id: str) -> Dict[str, Any]:
    """Retrieve detailed information for a specific travel destination by its ID.

    Args:
        destination_id: The unique slug/id of the destination (e.g. 'kyoto-japan', 'amalfi-italy', 'banff-canada', 'reykjavik-iceland').

    Returns:
        A dictionary with full destination details, or an error dictionary if not found.
    """
    doc_ref = db.collection(COLLECTION_NAME).document(destination_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {"error": f"Destination '{destination_id}' not found."}
    data = doc.to_dict()
    data["id"] = doc.id
    return data


def add_or_update_destination(
    destination_id: str,
    name: str,
    country: str,
    vibes: List[str],
    budget_level: str,
    best_season: str,
    highlights: List[str],
    description: str,
    recommended_days: int = 3,
) -> Dict[str, Any]:
    """Add a new destination or update an existing one in the travel catalog.

    Args:
        destination_id: Unique slug/id for the destination (e.g. 'tokyo-japan', 'santorini-greece').
        name: Name of the destination (e.g. 'Tokyo, Japan').
        country: Country where the destination is located.
        vibes: List of travel vibes/styles (e.g. ['cultural', 'nightlife', 'foodie']).
        budget_level: Budget tier (e.g. 'budget', 'moderate', 'luxury').
        best_season: Recommended time of year to visit.
        highlights: Key attractions or activities.
        description: A vivid description of the destination.
        recommended_days: Recommended stay duration in days.

    Returns:
        A dictionary confirming the operation and containing the saved destination data.
    """
    doc_ref = db.collection(COLLECTION_NAME).document(destination_id)
    data = {
        "name": name,
        "country": country,
        "vibes": vibes,
        "budget_level": budget_level,
        "best_season": best_season,
        "highlights": highlights,
        "description": description,
        "recommended_days": recommended_days,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    doc_ref.set(data, merge=True)
    return {"status": "success", "id": destination_id, "data": data}


async def generate_destination_postcard(
    destination_name: str,
    tool_context: ToolContext,
    visual_style: str = "scenic golden hour postcard",
) -> Dict[str, str]:
    """Illustrates a travel destination postcard using AI image generation, saves it as an artifact in Playground, and uploads it to public Cloud Storage.

    Args:
        destination_name: Name of the destination to illustrate (e.g. 'Kyoto, Japan', 'Amalfi Coast, Italy', 'Banff, Canada').
        tool_context: Injected ADK ToolContext to save artifacts in Playground.
        visual_style: Artistic mood or style for the postcard (e.g. 'scenic golden hour postcard', 'vibrant watercolor landscape').

    Returns:
        A dictionary with the destination name, public image URL, and caption.
    """
    prompt = f"A beautiful travel postcard illustration of {destination_name}, {visual_style}, high quality travel photography."
    response = await genai_client.aio.models.generate_content(
        model="gemini-3.1-flash-lite-image",
        contents=prompt,
    )

    image_bytes = None
    mime_type = "image/jpeg"
    parts = getattr(response.candidates[0].content, "parts", None) or []
    for part in parts:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            image_bytes = part.inline_data.data
            mime_type = part.inline_data.mime_type or "image/jpeg"
            break

    if not image_bytes:
        return {"error": "Failed to generate image bytes."}

    # 1. Save artifact to Playground
    clean_slug = destination_name.lower().replace(" ", "_").replace(",", "")
    artifact_filename = f"{clean_slug}_postcard.jpg"
    artifact_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    await tool_context.save_artifact(filename=artifact_filename, artifact=artifact_part)

    # 2. Upload same bytes directly to Cloud Storage bucket (no local file write)
    ext = "jpg" if "jpeg" in mime_type or "jpg" in mime_type else "png"
    storage_filename = f"postcards/{uuid.uuid4().hex[:8]}_{clean_slug}.{ext}"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.upload_from_string(image_bytes, content_type=mime_type)

    public_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{storage_filename}"
    return {
        "destination": destination_name,
        "image_url": public_url,
        "caption": f"Postcard illustration of {destination_name}",
    }


async def generate_destination_video(
    destination_name: str,
    tool_context: ToolContext,
    video_prompt: str = "A scenic cinematic 4-second drone video showcasing the landscape, atmospheric lighting, high quality travel footage",
) -> Dict[str, str]:
    """Generates a short travel video for a destination using Gemini Omni (gemini-omni-flash-preview), saves it as an artifact in Playground, and uploads it to public Cloud Storage.

    Args:
        destination_name: Name of the destination (e.g. 'Kyoto, Japan', 'Amalfi Coast, Italy', 'Banff, Canada').
        tool_context: Injected ADK ToolContext to save artifacts in Playground.
        video_prompt: Creative prompt or scene description for the video clip (e.g. 'A scenic cinematic drone video at sunset', 'A slow motion walking clip through historical streets').

    Returns:
        A dictionary with the destination name, public video URL, and description.
    """
    full_prompt = f"A short video of {destination_name}. {video_prompt}"
    interaction = await genai_client.aio.interactions.create(
        model="gemini-omni-flash-preview",
        input=full_prompt,
    )

    video_bytes = None
    mime_type = "video/mp4"

    if (
        hasattr(interaction, "output_video")
        and interaction.output_video
        and getattr(interaction.output_video, "data", None)
    ):
        video_bytes = base64.b64decode(interaction.output_video.data)
        mime_type = getattr(interaction.output_video, "mime_type", None) or "video/mp4"

    if not video_bytes:
        return {"error": "Failed to generate video bytes from Omni model."}

    # 1. Save artifact to Playground
    clean_slug = destination_name.lower().replace(" ", "_").replace(",", "")
    artifact_filename = f"{clean_slug}_video.mp4"
    artifact_part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
    await tool_context.save_artifact(filename=artifact_filename, artifact=artifact_part)

    # 2. Upload same bytes directly to Cloud Storage bucket (no local file write)
    storage_filename = f"videos/{uuid.uuid4().hex[:8]}_{clean_slug}.mp4"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.upload_from_string(video_bytes, content_type=mime_type)

    public_url = f"https://storage.googleapis.com/{BUCKET_NAME}/{storage_filename}"
    return {
        "destination": destination_name,
        "video_url": public_url,
        "caption": f"Cinematic travel video of {destination_name}",
    }


def get_destination_live_weather(destination_name: str) -> Dict[str, Any]:
    """Fetch live real-time weather and forecast for any travel destination using the Open-Meteo API.

    Args:
        destination_name: The city or region name (e.g. 'Kyoto', 'Amalfi', 'Banff', 'Reykjavik', 'Paris', 'Tokyo').

    Returns:
        A dictionary with live temperature, apparent temperature, humidity, wind speed, and daily high/low.
    """
    import json
    import urllib.parse
    import urllib.request

    # 1. Geocode location name
    encoded_name = urllib.parse.quote(destination_name)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_name}&count=1&language=en&format=json"
    req = urllib.request.Request(
        geo_url, headers={"User-Agent": "WanderlustTravelAgent/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            geo_data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"Failed to geocode '{destination_name}': {str(e)}"}

    if not geo_data.get("results"):
        return {
            "error": f"Location '{destination_name}' not found in geocoding service."
        }

    loc = geo_data["results"][0]
    lat, lon = loc["latitude"], loc["longitude"]
    resolved_name = (
        f"{loc.get('name', destination_name)}, {loc.get('country', '')}".strip(", ")
    )

    # 2. Fetch live weather & forecast
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m"
        "&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
    )
    req2 = urllib.request.Request(
        weather_url, headers={"User-Agent": "WanderlustTravelAgent/1.0"}
    )

    try:
        with urllib.request.urlopen(req2, timeout=5) as resp2:
            w_data = json.loads(resp2.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"Failed to fetch weather for '{resolved_name}': {str(e)}"}

    curr = w_data.get("current", {})
    daily = w_data.get("daily", {})
    return {
        "destination": resolved_name,
        "temperature_celsius": curr.get("temperature_2m"),
        "apparent_temperature_celsius": curr.get("apparent_temperature"),
        "humidity_percent": curr.get("relative_humidity_2m"),
        "wind_speed_kmh": curr.get("wind_speed_10m"),
        "precipitation_mm": curr.get("precipitation"),
        "today_max_celsius": daily.get("temperature_2m_max", [None])[0],
        "today_min_celsius": daily.get("temperature_2m_min", [None])[0],
    }


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=a2ui_system_instruction,
    code_executor=code_executor,
    tools=[
        PreloadMemoryTool(),
        list_destinations,
        get_destination,
        add_or_update_destination,
        generate_destination_postcard,
        generate_destination_video,
        get_destination_live_weather,
    ],
    after_model_callback=a2ui_callback,
    after_agent_callback=generate_memories_callback,
)

app = App(
    root_agent=root_agent,
    name="app",
)
