"""
AWS Lambda entry point for Clarividex.

Uses Mangum to wrap the FastAPI app for Lambda Function URL.
Handles both Function URL events and EventBridge warmup pings.
"""

import json

from mangum import Mangum

from backend.app.main import app

mangum_handler = Mangum(app, lifespan="off")


def handler(event, context):
    """
    Lambda handler that routes between warmup pings and real requests.

    EventBridge warmup events are detected by:
    - 'source' key set to 'aws.events' (default EventBridge format)
    - 'detail-type' key present (default EventBridge format)
    - 'CloudWatch-Warmup' user-agent (custom warmup input format)
    Function URL requests are passed through to Mangum/FastAPI.
    """
    # Detect EventBridge warmup ping (default format)
    if event.get("source") == "aws.events" or event.get("detail-type"):
        return {
            "statusCode": 200,
            "body": json.dumps({"status": "warm", "source": "warmup-ping"}),
        }

    # Detect custom warmup input (formatted as HTTP request with CloudWatch-Warmup UA)
    headers = event.get("headers", {})
    if isinstance(headers, dict) and headers.get("user-agent") == "CloudWatch-Warmup":
        return {
            "statusCode": 200,
            "body": json.dumps({"status": "warm", "source": "warmup-ping"}),
        }

    return mangum_handler(event, context)
