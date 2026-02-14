"""
AWS Lambda entry point for Clarividex.

Uses Mangum to wrap the FastAPI app for Lambda + API Gateway.
"""

from mangum import Mangum

from backend.app.main import app

handler = Mangum(app, lifespan="off")
