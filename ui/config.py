import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TOKEN_REFRESH_THRESHOLD_SECONDS: int = 300
TOKEN_PRICE: float = 0.5

TRAINING_COST=10
PREDICTION_COST=5
METADATA_COST=1
ASSIST_COST=2

COOLDOWN_KEY = "cooldowns"
