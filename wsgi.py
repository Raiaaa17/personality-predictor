"""
WSGI config for Personality Predictor.

This module contains the WSGI application used by the production server.
"""

from app import app

if __name__ == "__main__":
    app.run()
