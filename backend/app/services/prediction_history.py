"""
Prediction History Service

Tracks all predictions made by the system and their outcomes for accuracy analysis.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "predictions.db"


@dataclass
class PredictionRecord:
    """A historical prediction record."""
    id: str
    query: str
    ticker: Optional[str]
    prediction_type: str
    probability: float
    confidence_level: str
    target_price: Optional[float]
    target_date: Optional[str]
    current_price_at_prediction: Optional[float]
    sentiment: str
    data_quality_score: float
    data_points_analyzed: int
    sources_count: int
    created_at: str

    # Outcome tracking
    outcome: Optional[str] = None  # 'correct', 'incorrect', 'pending'
    actual_price_at_resolution: Optional[float] = None
    resolved_at: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class AccuracyStats:
    """Accuracy statistics."""
    total_predictions: int
    resolved_predictions: int
    correct_predictions: int
    incorrect_predictions: int
    pending_predictions: int
    accuracy_rate: float
    average_probability_correct: float
    average_probability_incorrect: float
    by_ticker: Dict[str, Dict[str, Any]]
    by_confidence: Dict[str, Dict[str, Any]]
    by_timeframe: Dict[str, Dict[str, Any]]
    recent_predictions: List[Dict[str, Any]]


class PredictionHistoryService:
    """Service for tracking prediction history and accuracy."""

    def __init__(self):
        self._ensure_db()

    def _ensure_db(self):
        """Ensure database and tables exist."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                ticker TEXT,
                prediction_type TEXT NOT NULL,
                probability REAL NOT NULL,
                confidence_level TEXT NOT NULL,
                target_price REAL,
                target_date TEXT,
                current_price_at_prediction REAL,
                sentiment TEXT NOT NULL,
                data_quality_score REAL NOT NULL,
                data_points_analyzed INTEGER NOT NULL,
                sources_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                outcome TEXT DEFAULT 'pending',
                actual_price_at_resolution REAL,
                resolved_at TEXT,
                notes TEXT,
                reasoning_summary TEXT,
                bullish_factors_count INTEGER,
                bearish_factors_count INTEGER
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON predictions(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON predictions(outcome)")

        conn.commit()
        conn.close()

        logger.info("Prediction history database initialized", path=str(DB_PATH))

    def record_prediction(
        self,
        prediction_id: str,
        query: str,
        ticker: Optional[str],
        prediction_type: str,
        probability: float,
        confidence_level: str,
        target_price: Optional[float],
        target_date: Optional[str],
        current_price: Optional[float],
        sentiment: str,
        data_quality_score: float,
        data_points_analyzed: int,
        sources_count: int,
        reasoning_summary: str = "",
        bullish_factors_count: int = 0,
        bearish_factors_count: int = 0,
    ) -> bool:
        """Record a new prediction."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO predictions (
                    id, query, ticker, prediction_type, probability, confidence_level,
                    target_price, target_date, current_price_at_prediction, sentiment,
                    data_quality_score, data_points_analyzed, sources_count, created_at,
                    reasoning_summary, bullish_factors_count, bearish_factors_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                query,
                ticker,
                prediction_type,
                probability,
                confidence_level,
                target_price,
                target_date,
                current_price,
                sentiment,
                data_quality_score,
                data_points_analyzed,
                sources_count,
                datetime.utcnow().isoformat(),
                reasoning_summary,
                bullish_factors_count,
                bearish_factors_count,
            ))

            conn.commit()
            conn.close()

            logger.info("Prediction recorded", id=prediction_id, ticker=ticker)
            return True

        except Exception as e:
            logger.error("Failed to record prediction", error=str(e))
            return False

    def update_outcome(
        self,
        prediction_id: str,
        outcome: str,
        actual_price: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Update the outcome of a prediction."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE predictions
                SET outcome = ?, actual_price_at_resolution = ?, resolved_at = ?, notes = ?
                WHERE id = ?
            """, (
                outcome,
                actual_price,
                datetime.utcnow().isoformat(),
                notes,
                prediction_id,
            ))

            conn.commit()
            conn.close()

            logger.info("Prediction outcome updated", id=prediction_id, outcome=outcome)
            return True

        except Exception as e:
            logger.error("Failed to update outcome", error=str(e))
            return False

    def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific prediction by ID."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            conn.close()

            if row:
                return dict(row)
            return None

        except Exception as e:
            logger.error("Failed to get prediction", error=str(e))
            return None

    def get_recent_predictions(
        self,
        limit: int = 20,
        ticker: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent predictions with optional filters."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM predictions WHERE 1=1"
            params = []

            if ticker:
                query += " AND ticker = ?"
                params.append(ticker.upper())

            if outcome:
                query += " AND outcome = ?"
                params.append(outcome)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error("Failed to get recent predictions", error=str(e))
            return []

    def get_accuracy_stats(self) -> AccuracyStats:
        """Calculate overall accuracy statistics."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get total counts
            cursor.execute("SELECT COUNT(*) as count FROM predictions")
            total = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE outcome = 'correct'")
            correct = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE outcome = 'incorrect'")
            incorrect = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE outcome = 'pending'")
            pending = cursor.fetchone()["count"]

            resolved = correct + incorrect
            accuracy_rate = (correct / resolved * 100) if resolved > 0 else 0.0

            # Average probability for correct vs incorrect
            cursor.execute("""
                SELECT AVG(probability) as avg_prob
                FROM predictions WHERE outcome = 'correct'
            """)
            avg_prob_correct = cursor.fetchone()["avg_prob"] or 0.0

            cursor.execute("""
                SELECT AVG(probability) as avg_prob
                FROM predictions WHERE outcome = 'incorrect'
            """)
            avg_prob_incorrect = cursor.fetchone()["avg_prob"] or 0.0

            # By ticker
            cursor.execute("""
                SELECT ticker,
                       COUNT(*) as total,
                       SUM(CASE WHEN outcome = 'correct' THEN 1 ELSE 0 END) as correct,
                       SUM(CASE WHEN outcome = 'incorrect' THEN 1 ELSE 0 END) as incorrect
                FROM predictions
                WHERE ticker IS NOT NULL
                GROUP BY ticker
                ORDER BY total DESC
                LIMIT 10
            """)
            by_ticker = {}
            for row in cursor.fetchall():
                resolved_t = row["correct"] + row["incorrect"]
                by_ticker[row["ticker"]] = {
                    "total": row["total"],
                    "correct": row["correct"],
                    "incorrect": row["incorrect"],
                    "accuracy": (row["correct"] / resolved_t * 100) if resolved_t > 0 else 0,
                }

            # By confidence level
            cursor.execute("""
                SELECT confidence_level,
                       COUNT(*) as total,
                       SUM(CASE WHEN outcome = 'correct' THEN 1 ELSE 0 END) as correct,
                       SUM(CASE WHEN outcome = 'incorrect' THEN 1 ELSE 0 END) as incorrect
                FROM predictions
                GROUP BY confidence_level
            """)
            by_confidence = {}
            for row in cursor.fetchall():
                resolved_c = row["correct"] + row["incorrect"]
                by_confidence[row["confidence_level"]] = {
                    "total": row["total"],
                    "correct": row["correct"],
                    "incorrect": row["incorrect"],
                    "accuracy": (row["correct"] / resolved_c * 100) if resolved_c > 0 else 0,
                }

            # Recent predictions
            cursor.execute("""
                SELECT id, ticker, probability, outcome, created_at, query
                FROM predictions
                ORDER BY created_at DESC
                LIMIT 10
            """)
            recent = [dict(row) for row in cursor.fetchall()]

            conn.close()

            return AccuracyStats(
                total_predictions=total,
                resolved_predictions=resolved,
                correct_predictions=correct,
                incorrect_predictions=incorrect,
                pending_predictions=pending,
                accuracy_rate=accuracy_rate,
                average_probability_correct=avg_prob_correct,
                average_probability_incorrect=avg_prob_incorrect,
                by_ticker=by_ticker,
                by_confidence=by_confidence,
                by_timeframe={},  # Can be added later
                recent_predictions=recent,
            )

        except Exception as e:
            logger.error("Failed to get accuracy stats", error=str(e))
            return AccuracyStats(
                total_predictions=0,
                resolved_predictions=0,
                correct_predictions=0,
                incorrect_predictions=0,
                pending_predictions=0,
                accuracy_rate=0.0,
                average_probability_correct=0.0,
                average_probability_incorrect=0.0,
                by_ticker={},
                by_confidence={},
                by_timeframe={},
                recent_predictions=[],
            )

    def auto_resolve_predictions(self) -> int:
        """
        Automatically resolve predictions that have passed their target date.
        Returns the number of predictions resolved.
        """
        try:
            from backend.app.services.market_data import market_data_service

            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get pending predictions with target dates that have passed
            cursor.execute("""
                SELECT id, ticker, target_price, probability, prediction_type
                FROM predictions
                WHERE outcome = 'pending'
                  AND target_date IS NOT NULL
                  AND target_date <= ?
            """, (datetime.utcnow().isoformat(),))

            predictions = cursor.fetchall()
            resolved_count = 0

            for pred in predictions:
                if pred["ticker"]:
                    # Get current price
                    quote = market_data_service.get_quote(pred["ticker"])
                    if quote:
                        current_price = quote.current_price
                        target_price = pred["target_price"]
                        probability = pred["probability"]

                        # Determine if prediction was correct
                        if pred["prediction_type"] == "price_target":
                            # Price target prediction
                            if target_price:
                                reached = current_price >= target_price
                                predicted_bullish = probability >= 0.5
                                is_correct = reached == predicted_bullish
                            else:
                                continue
                        else:
                            # General direction prediction
                            # Check if price moved in predicted direction
                            is_correct = probability >= 0.5  # Simplified

                        outcome = "correct" if is_correct else "incorrect"

                        cursor.execute("""
                            UPDATE predictions
                            SET outcome = ?, actual_price_at_resolution = ?, resolved_at = ?
                            WHERE id = ?
                        """, (
                            outcome,
                            current_price,
                            datetime.utcnow().isoformat(),
                            pred["id"],
                        ))

                        resolved_count += 1

            conn.commit()
            conn.close()

            logger.info("Auto-resolved predictions", count=resolved_count)
            return resolved_count

        except Exception as e:
            logger.error("Failed to auto-resolve predictions", error=str(e))
            return 0


# Singleton instance
prediction_history = PredictionHistoryService()
