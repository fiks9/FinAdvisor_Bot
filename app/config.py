"""
app/config.py — Centralised configuration via Pydantic v2 Settings
====================================================================
Reads all secrets/paths from .env (via python-dotenv).
If any required variable is missing the app will REFUSE to start
and print a clear validation error — no silent failures.

TODO (Крок 3): implement full Settings class with Pydantic BaseSettings
"""
