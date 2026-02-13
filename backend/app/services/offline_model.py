"""
Offline Model Service - Smart Local LLM fallback using Ollama.

Provides intelligent fallback when Claude API is unavailable.
Supports multiple models with automatic selection based on quality/speed tradeoffs.

To install Ollama:
1. Download from https://ollama.ai (or run: curl -fsSL https://ollama.ai/install.sh | sh)
2. Run: ollama pull llama3.2:3b (fast) or ollama pull mistral (better quality)
"""

import json
import re
import asyncio
from typing import Optional, Any
from datetime import datetime

import httpx
import structlog

logger = structlog.get_logger()


class OfflineModelService:
    """
    Smart local LLM service via Ollama.

    Features:
    - Automatic model selection based on availability
    - Structured output parsing with multiple fallback strategies
    - Async support for non-blocking inference
    - Intelligent prompt engineering for financial analysis
    """

    OLLAMA_BASE_URL = "http://localhost:11434"

    # Models ranked by quality (best first)
    # The service will use the best available model
    MODEL_PREFERENCES = [
        {"name": "llama3.1:8b", "quality": 0.9, "speed": "slow"},
        {"name": "mistral", "quality": 0.85, "speed": "medium"},
        {"name": "llama3.2:3b", "quality": 0.75, "speed": "fast"},
        {"name": "phi3", "quality": 0.7, "speed": "fast"},
        {"name": "gemma:7b", "quality": 0.8, "speed": "medium"},
        {"name": "gemma:2b", "quality": 0.6, "speed": "fast"},
        {"name": "llama2", "quality": 0.7, "speed": "medium"},
        {"name": "neural-chat", "quality": 0.65, "speed": "medium"},
    ]

    @property
    def system_prompt(self) -> str:
        """Get the active system prompt, falling back to hardcoded version."""
        try:
            from backend.app.prompts.registry import get_active_prompt
            prompt = get_active_prompt("offline_model")
            if prompt:
                return prompt
        except Exception:
            pass
        return self._FALLBACK_SYSTEM_PROMPT

    # Optimized system prompt for financial analysis
    _FALLBACK_SYSTEM_PROMPT = """You are a quantitative financial analyst AI. Analyze the provided market data and generate a probability assessment.

CRITICAL CONSTRAINT: You ONLY analyze financial market queries about:
- Stocks, cryptocurrencies, forex pairs, commodities, indices, ETFs, futures, bonds

If the query is NOT about financial markets, respond ONLY with:
{"error": "non_financial_query", "message": "I can only analyze financial market predictions."}

IMPORTANT INSTRUCTIONS:
1. Base your analysis ONLY on the provided data
2. Calculate probability as a number between 0-100
3. Be specific - cite exact numbers from the data
4. Balance bullish and bearish factors objectively
5. Consider technical indicators, sentiment, and news

You MUST respond with ONLY a JSON object (no other text):
{
    "probability": <0-100>,
    "confidence": "low|medium|high",
    "summary": "<2-3 sentence analysis>",
    "bullish_factors": [{"description": "<specific factor with data>", "weight": <0.1-1.0>, "source": "<data source>"}],
    "bearish_factors": [{"description": "<specific factor with data>", "weight": <0.1-1.0>, "source": "<data source>"}],
    "catalysts": [{"event": "<event>", "date": null, "impact": "positive|negative|uncertain", "importance": "low|medium|high"}],
    "risks": ["<risk1>", "<risk2>"],
    "assumptions": ["<assumption1>"],
    "sentiment_assessment": "bearish|neutral|bullish",
    "technical_assessment": "sell|neutral|buy",
    "fundamental_assessment": "weak|neutral|strong"
}"""

    def __init__(self):
        """Initialize the offline model service."""
        self.logger = logger.bind(service="offline_model")
        self.http_client: Optional[httpx.AsyncClient] = None
        self.sync_client: Optional[httpx.Client] = None
        self.selected_model: Optional[str] = None
        self.model_quality: float = 0.5
        self.is_available: bool = False
        self._checked: bool = False

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                base_url=self.OLLAMA_BASE_URL,
                timeout=httpx.Timeout(180.0, connect=10.0),  # 3 min for inference
            )
        return self.http_client

    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self.sync_client is None:
            self.sync_client = httpx.Client(
                base_url=self.OLLAMA_BASE_URL,
                timeout=httpx.Timeout(180.0, connect=10.0),
            )
        return self.sync_client

    async def check_availability(self) -> bool:
        """
        Check if Ollama is running and select the best available model.

        Returns:
            True if Ollama is available with a compatible model
        """
        if self._checked:
            return self.is_available

        self._checked = True
        self.logger.info("Checking Ollama availability...")

        try:
            client = await self._get_async_client()
            response = await client.get("/api/tags")

            if response.status_code != 200:
                self.logger.warning("Ollama not responding", status=response.status_code)
                return False

            data = response.json()
            installed_models = {m["name"] for m in data.get("models", [])}

            self.logger.info("Ollama models available", models=list(installed_models))

            # Select best available model
            for model_info in self.MODEL_PREFERENCES:
                model_name = model_info["name"]
                # Check exact match or base name match
                if model_name in installed_models:
                    self.selected_model = model_name
                    self.model_quality = model_info["quality"]
                    break
                # Check partial match (e.g., "llama3.2:3b" matches "llama3.2:3b-instruct-q4_0")
                for installed in installed_models:
                    if installed.startswith(model_name.split(":")[0]):
                        self.selected_model = installed
                        self.model_quality = model_info["quality"]
                        break
                if self.selected_model:
                    break

            if self.selected_model:
                self.is_available = True
                self.logger.info(
                    "Ollama ready",
                    model=self.selected_model,
                    quality=self.model_quality,
                )
                return True

            # If no preferred model, use first available
            if installed_models:
                self.selected_model = next(iter(installed_models))
                self.model_quality = 0.5
                self.is_available = True
                self.logger.info("Using fallback model", model=self.selected_model)
                return True

            self.logger.warning("No Ollama models installed")
            return False

        except httpx.ConnectError:
            self.logger.warning("Ollama not running. Start with: ollama serve")
            return False
        except Exception as e:
            self.logger.error("Ollama check failed", error=str(e))
            return False

    async def generate_analysis(
        self,
        query: str,
        context_data: str,
        target_price: Optional[float] = None,
        target_date: Optional[datetime] = None,
    ) -> Optional[dict]:
        """
        Generate financial analysis using the local model.

        Args:
            query: User's prediction query
            context_data: Formatted market data context
            target_price: Target price if applicable
            target_date: Target date if applicable

        Returns:
            Parsed analysis dict or None if failed
        """
        if not self.is_available:
            await self.check_availability()
            if not self.is_available:
                return None

        self.logger.info("Generating analysis with Ollama", model=self.selected_model)

        # Build the prompt
        prompt = self._build_analysis_prompt(query, context_data, target_price, target_date)

        try:
            client = await self._get_async_client()

            request_data = {
                "model": self.selected_model,
                "prompt": prompt,
                "system": self.system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Lower for more deterministic output
                    "num_predict": 2048,
                    "top_p": 0.9,
                    "top_k": 40,
                },
                "format": "json",  # Request JSON format if model supports it
            }

            response = await client.post("/api/generate", json=request_data)

            if response.status_code != 200:
                self.logger.error("Ollama generation failed", status=response.status_code)
                return None

            data = response.json()
            response_text = data.get("response", "")

            # Log generation stats
            total_duration = data.get("total_duration", 0) / 1e9  # Convert to seconds
            self.logger.info(
                "Ollama generation complete",
                duration_sec=f"{total_duration:.1f}",
                model=self.selected_model,
            )

            # Parse the response
            return self._parse_response(response_text)

        except httpx.TimeoutException:
            self.logger.error("Ollama generation timed out")
            return None
        except Exception as e:
            self.logger.error("Ollama generation error", error=str(e))
            return None

    def generate_analysis_sync(
        self,
        query: str,
        context_data: str,
        target_price: Optional[float] = None,
        target_date: Optional[datetime] = None,
    ) -> Optional[dict]:
        """Synchronous version of generate_analysis."""
        if not self.is_available:
            # Run async check in sync context
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.check_availability())
            finally:
                loop.close()
            if not self.is_available:
                return None

        prompt = self._build_analysis_prompt(query, context_data, target_price, target_date)

        try:
            client = self._get_sync_client()

            request_data = {
                "model": self.selected_model,
                "prompt": prompt,
                "system": self.system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 2048,
                },
                "format": "json",
            }

            response = client.post("/api/generate", json=request_data)

            if response.status_code != 200:
                return None

            data = response.json()
            return self._parse_response(data.get("response", ""))

        except Exception as e:
            self.logger.error("Sync generation failed", error=str(e))
            return None

    def _build_analysis_prompt(
        self,
        query: str,
        context_data: str,
        target_price: Optional[float],
        target_date: Optional[datetime],
    ) -> str:
        """Build an optimized prompt for financial analysis."""
        parts = [
            "ANALYSIS REQUEST",
            "=" * 40,
            f"Question: {query}",
        ]

        if target_price:
            parts.append(f"Target Price: ${target_price:.2f}")
        if target_date:
            parts.append(f"Target Date: {target_date.strftime('%Y-%m-%d')}")
            days = (target_date - datetime.now()).days
            parts.append(f"Days Until Target: {days}")

        parts.extend([
            "",
            "MARKET DATA",
            "=" * 40,
            context_data,
            "",
            "INSTRUCTIONS",
            "=" * 40,
            "1. Analyze ALL the data above",
            "2. Calculate probability (0-100%) based on evidence",
            "3. List specific bullish and bearish factors with data",
            "4. Respond with ONLY valid JSON, no other text",
        ])

        return "\n".join(parts)

    def _parse_response(self, response_text: str) -> Optional[dict]:
        """
        Parse model response with multiple fallback strategies.

        Tries:
        1. Direct JSON parse
        2. Extract JSON from markdown code block
        3. Find JSON object in text
        4. Manual extraction as last resort
        """
        if not response_text:
            return None

        response_text = response_text.strip()

        # Strategy 1: Direct parse
        try:
            result = json.loads(response_text)
            if self._validate_analysis(result):
                return self._normalize_analysis(result)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code block
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if code_block_match:
            try:
                result = json.loads(code_block_match.group(1))
                if self._validate_analysis(result):
                    return self._normalize_analysis(result)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON object in text
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                # Handle potential nested braces by finding matching pair
                json_str = self._extract_balanced_json(response_text)
                if json_str:
                    result = json.loads(json_str)
                    if self._validate_analysis(result):
                        return self._normalize_analysis(result)
            except json.JSONDecodeError:
                pass

        # Strategy 4: Manual extraction
        self.logger.warning("JSON parsing failed, using manual extraction")
        return self._extract_manually(response_text)

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """Extract a balanced JSON object from text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        return None

    def _validate_analysis(self, data: dict) -> bool:
        """Check if parsed data has required fields."""
        required = ["probability"]
        return all(key in data for key in required)

    def _normalize_analysis(self, data: dict) -> dict:
        """Normalize and fill in missing fields."""
        # Ensure probability is in correct range
        prob = data.get("probability", 50)
        if isinstance(prob, str):
            prob = float(re.sub(r"[^\d.]", "", prob) or "50")
        data["probability"] = max(0, min(100, prob))

        # Set defaults for missing fields
        defaults = {
            "confidence": "medium",
            "summary": "Analysis generated by local AI model.",
            "bullish_factors": [],
            "bearish_factors": [],
            "catalysts": [],
            "risks": ["Market volatility", "Unexpected events"],
            "assumptions": ["Current conditions persist"],
            "sentiment_assessment": "neutral",
            "technical_assessment": "neutral",
            "fundamental_assessment": "neutral",
        }

        for key, default in defaults.items():
            if key not in data or data[key] is None:
                data[key] = default

        # Normalize factor weights
        for factors in [data.get("bullish_factors", []), data.get("bearish_factors", [])]:
            for factor in factors:
                if isinstance(factor, dict):
                    weight = factor.get("weight", 0.5)
                    if isinstance(weight, str):
                        weight = float(re.sub(r"[^\d.-]", "", weight) or "0.5")
                    factor["weight"] = max(0, min(1, abs(weight)))
                    factor.setdefault("source", "Analysis")
                    factor.setdefault("description", "Factor")

        return data

    def _extract_manually(self, text: str) -> dict:
        """Extract key values from unstructured text as last resort."""
        # Find probability
        prob_patterns = [
            r"probability[:\s]+(\d{1,3})%?",
            r"(\d{1,3})%\s*(?:probability|chance|likelihood)",
            r"(\d{1,3})\s*percent",
        ]
        probability = 50
        for pattern in prob_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                probability = int(match.group(1))
                break

        # Determine sentiment from keywords
        text_lower = text.lower()
        bullish_keywords = ["bullish", "positive", "optimistic", "upside", "growth", "strong"]
        bearish_keywords = ["bearish", "negative", "pessimistic", "downside", "decline", "weak"]

        bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in text_lower)

        if bullish_count > bearish_count:
            sentiment = "bullish"
        elif bearish_count > bullish_count:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "probability": max(0, min(100, probability)),
            "confidence": "low",
            "summary": text[:500] if len(text) > 10 else "Analysis completed with local model.",
            "bullish_factors": [],
            "bearish_factors": [],
            "catalysts": [],
            "risks": ["Analysis parsing incomplete - results may be approximate"],
            "assumptions": ["Based on available data patterns"],
            "sentiment_assessment": sentiment,
            "technical_assessment": "neutral",
            "fundamental_assessment": "neutral",
        }

    async def close(self):
        """Clean up resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        if self.sync_client:
            self.sync_client.close()
            self.sync_client = None


# Singleton instance
offline_model_service = OfflineModelService()
