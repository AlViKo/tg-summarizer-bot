import os
import sys

# Ensure project root is on path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set dummy env vars before any project module is imported
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "fake-token")
os.environ.setdefault("OPENAI_API_KEY", "fake-key")
