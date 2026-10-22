"""Unit tests for Wanderlust Illustrator agent tools and configuration."""

from unittest.mock import MagicMock, patch

from app.agent import (
    add_or_update_destination,
    get_destination,
    get_destination_live_weather,
    list_destinations,
    root_agent,
)


def test_root_agent_configuration() -> None:
    """Test that root_agent is properly configured with model and tools."""
    assert root_agent.name == "root_agent"
    assert root_agent.model.model == "gemini-2.5-flash"
    assert root_agent.code_executor is not None

    tool_names = [
        getattr(t, "__name__", t.__class__.__name__) for t in root_agent.tools
    ]
    assert "PreloadMemoryTool" in tool_names
    assert "list_destinations" in tool_names
    assert "get_destination" in tool_names
    assert "add_or_update_destination" in tool_names
    assert "generate_destination_postcard" in tool_names
    assert "generate_destination_video" in tool_names
    assert "get_destination_live_weather" in tool_names


@patch("app.agent.db")
def test_list_destinations_with_vibe_and_budget(mock_db: MagicMock) -> None:
    """Test destination listing with filtering."""
    doc1 = MagicMock()
    doc1.id = "kyoto-japan"
    doc1.to_dict.return_value = {
        "name": "Kyoto, Japan",
        "vibes": ["cultural", "scenic"],
        "budget_level": "moderate",
    }

    doc2 = MagicMock()
    doc2.id = "amalfi-italy"
    doc2.to_dict.return_value = {
        "name": "Amalfi Coast, Italy",
        "vibes": ["romantic", "coastal"],
        "budget_level": "luxury",
    }

    mock_db.collection.return_value.stream.return_value = [doc1, doc2]

    # Filter by vibe cultural
    results = list_destinations(vibe="cultural")
    assert len(results) == 1
    assert results[0]["id"] == "kyoto-japan"

    # Filter by budget luxury
    results_luxury = list_destinations(budget_level="luxury")
    assert len(results_luxury) == 1
    assert results_luxury[0]["id"] == "amalfi-italy"


@patch("app.agent.db")
def test_get_destination_found_and_not_found(mock_db: MagicMock) -> None:
    """Test retrieving destination details."""
    mock_doc = MagicMock()
    mock_doc.exists = True
    mock_doc.id = "kyoto-japan"
    mock_doc.to_dict.return_value = {"name": "Kyoto, Japan", "country": "Japan"}

    mock_db.collection.return_value.document.return_value.get.return_value = mock_doc

    dest = get_destination("kyoto-japan")
    assert dest["id"] == "kyoto-japan"
    assert dest["name"] == "Kyoto, Japan"

    # Not found case
    mock_doc.exists = False
    not_found = get_destination("non-existent")
    assert "error" in not_found


@patch("app.agent.db")
def test_add_or_update_destination(mock_db: MagicMock) -> None:
    """Test adding or updating a destination."""
    res = add_or_update_destination(
        destination_id="bali-indonesia",
        name="Bali, Indonesia",
        country="Indonesia",
        vibes=["tropical", "beach", "cultural"],
        budget_level="moderate",
        best_season="Dry Season (May-September)",
        highlights=["Ubud Monkey Forest", "Uluwatu Temple"],
        description="Island of the Gods with lush greenery, beaches, and rich cultural heritage.",
        recommended_days=5,
    )

    assert res["status"] == "success"
    assert res["id"] == "bali-indonesia"
    assert res["data"]["country"] == "Indonesia"
    mock_db.collection.return_value.document.return_value.set.assert_called_once()


@patch("urllib.request.urlopen")
def test_get_destination_live_weather(mock_urlopen: MagicMock) -> None:
    """Test live weather tool with mock HTTP responses."""
    import json

    geo_response = MagicMock()
    geo_response.read.return_value = json.dumps(
        {
            "results": [
                {
                    "name": "Kyoto",
                    "country": "Japan",
                    "latitude": 35.0116,
                    "longitude": 135.7681,
                }
            ]
        }
    ).encode("utf-8")

    weather_response = MagicMock()
    weather_response.read.return_value = json.dumps(
        {
            "current": {
                "temperature_2m": 18.5,
                "apparent_temperature": 18.0,
                "relative_humidity_2m": 65,
                "wind_speed_10m": 12.0,
                "precipitation": 0.0,
            },
            "daily": {
                "temperature_2m_max": [22.0],
                "temperature_2m_min": [14.0],
            },
        }
    ).encode("utf-8")

    mock_urlopen.side_effect = [
        MagicMock(__enter__=MagicMock(return_value=geo_response)),
        MagicMock(__enter__=MagicMock(return_value=weather_response)),
    ]

    weather = get_destination_live_weather("Kyoto")
    assert weather["destination"] == "Kyoto, Japan"
    assert weather["temperature_celsius"] == 18.5
    assert weather["today_max_celsius"] == 22.0
