"""Compatibility entry point. Run with `uvicorn app:app --reload`."""
from backend.main import app

__all__ = ["app"]
