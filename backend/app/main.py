"""
Clarividex - Main FastAPI Application

The Clairvoyant Index - An AI-powered financial predictions platform that provides:
- Probabilistic forecasts with transparent reasoning
- Real-time market data aggregation
- Multi-source sentiment analysis
- Technical indicator calculations
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import uvicorn

from backend.app.config import settings
from backend.app.api.routes import router


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if settings.is_development else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "Starting Clarividex",
        environment=settings.app_env,
        debug=settings.app_debug,
    )

    # Log API key status
    if settings.has_anthropic_key:
        logger.info("Anthropic API key configured")
    else:
        logger.warning("Anthropic API key NOT configured - predictions will use fallback analysis")

    if settings.has_finnhub_key:
        logger.info("Finnhub API key configured")
    else:
        logger.info("Finnhub API key not configured - using free data sources only")

    # Initialize RAG index
    try:
        from backend.app.rag.service import rag_service
        rag_service.ensure_indexed()
    except Exception as e:
        logger.warning("RAG initialization failed (non-fatal)", error=str(e))

    yield

    # Shutdown
    logger.info("Shutting down Clarividex")

    # Clean up async resources
    from backend.app.services.news_service import news_service
    from backend.app.services.social_service import social_service

    await news_service.close()
    await social_service.close()


# Create FastAPI application
app = FastAPI(
    title="Clarividex",
    description="""
## Clarividex - The Clairvoyant Index

AI-Powered Market Predictions API. See tomorrow's markets today.

### Features:
- **Predictions**: Get probability estimates for price targets with full reasoning
- **Market Data**: Real-time quotes, technicals, and fundamentals
- **Sentiment Analysis**: News and social media sentiment
- **Transparent Reasoning**: Every prediction shows exactly why

### Example Query:
```
POST /api/v1/predict
{
    "query": "Will NVDA reach $150 by March 2026?"
}
```

### Disclaimer:
This is for informational purposes only and should not be considered financial advice.
""",
    version="1.0.0",
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://192.168.0.9:3000",
        "http://192.168.0.9:3001",
        "http://192.168.0.9:3002",
        "http://192.168.0.9:3003",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
from backend.app.middleware.rate_limit import RateLimitMiddleware

app.add_middleware(RateLimitMiddleware)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to every response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.app_debug else "An unexpected error occurred",
        },
    )


# Include API routes
app.include_router(router, prefix="/api/v1")


# Root redirect to docs
@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to API info."""
    return {
        "name": "Clarividex",
        "tagline": "The Clairvoyant Index",
        "version": "1.0.0",
        "docs": "/docs",
        "api": "/api/v1",
    }


def main():
    """Run the application."""
    logger.info(
        "Starting server",
        host=settings.app_host,
        port=settings.app_port,
    )

    uvicorn.run(
        "backend.app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_development,
        log_level="info" if settings.is_development else "warning",
    )


if __name__ == "__main__":
    main()
