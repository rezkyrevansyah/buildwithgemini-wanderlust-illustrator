"""Seed script for populating the destinations collection in Firestore."""

import os

from dotenv import load_dotenv
from google.cloud import firestore

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-42d301455a69")
COLLECTION_NAME = os.getenv("FIRESTORE_COLLECTION", "destinations")

DESTINATIONS = [
    {
        "id": "kyoto-japan",
        "name": "Kyoto, Japan",
        "country": "Japan",
        "vibes": ["cultural", "scenic", "historic"],
        "budget_level": "moderate",
        "best_season": "Spring / Autumn",
        "highlights": ["Fushimi Inari Shrine", "Arashiyama Bamboo Grove", "Kinkaku-ji"],
        "description": "Ancient capital of Japan renowned for classical Buddhist temples, gardens, imperial palaces, and traditional wooden houses.",
        "recommended_days": 4,
    },
    {
        "id": "amalfi-italy",
        "name": "Amalfi Coast, Italy",
        "country": "Italy",
        "vibes": ["romantic", "coastal", "relaxation"],
        "budget_level": "luxury",
        "best_season": "Late Spring / Early Autumn",
        "highlights": ["Positano Cliffside", "Ravello Gardens", "Capri Boat Excursion"],
        "description": "Dramatic cliffside coastline dotted with pastel-colored fishing villages, terraced vineyards, and cliff-clinging lemon groves.",
        "recommended_days": 5,
    },
    {
        "id": "banff-canada",
        "name": "Banff & Lake Louise, Canada",
        "country": "Canada",
        "vibes": ["adventure", "nature", "outdoor"],
        "budget_level": "moderate",
        "best_season": "Summer / Winter",
        "highlights": ["Lake Louise Canoeing", "Moraine Lake Sunrise", "Banff Gondola"],
        "description": "Breathtaking turquoise glacial lakes surrounded by the majestic peaks of the Canadian Rocky Mountains.",
        "recommended_days": 4,
    },
    {
        "id": "reykjavik-iceland",
        "name": "Reykjavik & South Coast, Iceland",
        "country": "Iceland",
        "vibes": ["adventure", "dramatic", "nature"],
        "budget_level": "luxury",
        "best_season": "Winter (Aurora) / Summer (Midnight Sun)",
        "highlights": [
            "Blue Lagoon",
            "Skógafoss Waterfall",
            "Black Sand Beach Reynisfjara",
        ],
        "description": "Land of fire and ice featuring geysers, hot springs, volcanic landscapes, and Northern Lights.",
        "recommended_days": 5,
    },
]


def seed_database():
    db = firestore.Client(project=PROJECT_ID)
    print(f"Connecting to Firestore for project {PROJECT_ID}...")
    for item in DESTINATIONS:
        doc_id = item["id"]
        data = {k: v for k, v in item.items() if k != "id"}
        db.collection(COLLECTION_NAME).document(doc_id).set(data)
        print(f"Seeded destination: {item['name']} ({doc_id})")
    print(
        f"Successfully seeded {len(DESTINATIONS)} destinations into '{COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    seed_database()
