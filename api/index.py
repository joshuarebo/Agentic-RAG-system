"""Vercel serverless function entry point.

Vercel's Python runtime expects an ASGI app exported from this module.
All requests are routed here via vercel.json rewrites.
"""
import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
