"""
Decision Trail Builder - Creates transparent visualization of prediction logic.

This service builds a complete decision trail showing exactly how each
factor contributed to the final prediction probability.
"""

import uuid
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

import structlog

from backend.app.models.schemas import DecisionNode, DecisionTrail

logger = structlog.get_logger()


@dataclass
class FactorContribution:
    """A single factor's contribution to the prediction."""

    category: str  # technical, news, options, market, analyst, social, historical_news, historical_patterns
    source: str  # Specific data source name
    data_point: str  # What was measured
    raw_value: Optional[str]  # The actual value
    signal: str  # bullish, bearish, neutral
    weight: float  # Category weight (0-1)
    score: float  # Signal score (-1 to +1)
    contribution: float  # Actual % contribution to probability
    reasoning: str  # Human-readable explanation


@dataclass
class DecisionTrailData:
    """Complete decision trail data structure."""

    factors: list[FactorContribution] = field(default_factory=list)
    base_probability: float = 50.0
    weighted_signal: float = 0.0
    pre_adjustment_probability: float = 50.0
    adjustments: list[dict] = field(default_factory=list)
    final_probability: float = 50.0
    confidence_factors: list[dict] = field(default_factory=list)


class DecisionTrailBuilder:
    """
    Builds a transparent decision trail for predictions.

    This creates a complete audit trail showing:
    - Each factor analyzed
    - The raw data values
    - How each was interpreted
    - The weight and contribution
    - All adjustments applied
    - The final calculation
    """

    # Category weights for the 8-factor model
    CATEGORY_WEIGHTS = {
        "technical": 0.20,
        "news": 0.15,
        "historical_news": 0.10,
        "options": 0.12,
        "market": 0.12,
        "analyst": 0.13,
        "social": 0.08,
        "historical_patterns": 0.10,
    }

    def __init__(self):
        """Initialize the decision trail builder."""
        self.logger = logger.bind(service="decision_trail_builder")
        self.trail = DecisionTrailData()

    def reset(self):
        """Reset the trail for a new prediction."""
        self.trail = DecisionTrailData()

    def add_factor(
        self,
        category: str,
        source: str,
        data_point: str,
        raw_value: Optional[str],
        signal: str,
        score: float,
        reasoning: str,
    ):
        """
        Add a factor contribution to the trail.

        Args:
            category: Factor category (technical, news, etc.)
            source: Data source name
            data_point: What was measured
            raw_value: The actual value
            signal: bullish/bearish/neutral
            score: Signal score (-1 to +1)
            reasoning: Human-readable explanation
        """
        weight = self.CATEGORY_WEIGHTS.get(category, 0.10)
        contribution = score * weight * 35  # Convert to probability points

        factor = FactorContribution(
            category=category,
            source=source,
            data_point=data_point,
            raw_value=raw_value,
            signal=signal,
            weight=weight,
            score=score,
            contribution=contribution,
            reasoning=reasoning,
        )

        self.trail.factors.append(factor)
        self.logger.debug(
            "Factor added to trail",
            category=category,
            source=source,
            contribution=f"{contribution:+.2f}%",
        )

    def add_adjustment(self, name: str, multiplier: float, reason: str):
        """
        Add a probability adjustment to the trail.

        Args:
            name: Adjustment name (price_gap, timeframe, trend_alignment)
            multiplier: Adjustment multiplier
            reason: Human-readable explanation
        """
        self.trail.adjustments.append({
            "name": name,
            "multiplier": multiplier,
            "reason": reason,
        })

    def add_confidence_factor(self, name: str, value: float, weight: float, description: str):
        """
        Add a confidence calculation factor.

        Args:
            name: Factor name
            value: Factor value (0-1)
            weight: Factor weight
            description: Human-readable description
        """
        self.trail.confidence_factors.append({
            "name": name,
            "value": value,
            "weight": weight,
            "contribution": value * weight,
            "description": description,
        })

    def calculate_weighted_signal(self) -> float:
        """
        Calculate the weighted signal from all factors.

        Returns:
            Weighted signal score (-1 to +1)
        """
        if not self.trail.factors:
            return 0.0

        # Group by category and average within each category
        category_scores = {}
        category_counts = {}

        for factor in self.trail.factors:
            cat = factor.category
            if cat not in category_scores:
                category_scores[cat] = 0.0
                category_counts[cat] = 0
            category_scores[cat] += factor.score
            category_counts[cat] += 1

        # Average within categories
        for cat in category_scores:
            if category_counts[cat] > 0:
                category_scores[cat] /= category_counts[cat]

        # Calculate weighted sum
        total_weight = sum(self.CATEGORY_WEIGHTS.get(cat, 0.10) for cat in category_scores)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            category_scores[cat] * self.CATEGORY_WEIGHTS.get(cat, 0.10)
            for cat in category_scores
        )

        self.trail.weighted_signal = weighted_sum / total_weight
        return self.trail.weighted_signal

    def calculate_probability(self, target_price: Optional[float] = None, current_price: Optional[float] = None, days_until_target: Optional[int] = None) -> float:
        """
        Calculate the final probability with all adjustments.

        Args:
            target_price: Target price if applicable
            current_price: Current price
            days_until_target: Days until target date

        Returns:
            Final probability (0-100)
        """
        # Base calculation
        weighted_signal = self.calculate_weighted_signal()
        base_prob = 50 + (weighted_signal * 35)
        self.trail.pre_adjustment_probability = base_prob

        adjusted_prob = base_prob

        # Price gap adjustment
        if target_price and current_price and current_price > 0:
            gap_pct = abs((target_price - current_price) / current_price) * 100

            if gap_pct > 50:
                multiplier = 0.50
                reason = f"Very large price gap ({gap_pct:.1f}%) significantly reduces probability"
            elif gap_pct > 40:
                multiplier = 0.60
                reason = f"Large price gap ({gap_pct:.1f}%) reduces probability"
            elif gap_pct > 30:
                multiplier = 0.70
                reason = f"Substantial price gap ({gap_pct:.1f}%) moderately reduces probability"
            elif gap_pct > 20:
                multiplier = 0.80
                reason = f"Notable price gap ({gap_pct:.1f}%) slightly reduces probability"
            elif gap_pct > 10:
                multiplier = 0.90
                reason = f"Moderate price gap ({gap_pct:.1f}%) minimally affects probability"
            else:
                multiplier = 1.0
                reason = f"Small price gap ({gap_pct:.1f}%) - no adjustment needed"

            if multiplier != 1.0:
                self.add_adjustment("price_gap", multiplier, reason)
                adjusted_prob *= multiplier

        # Timeframe adjustment
        if days_until_target is not None and days_until_target > 0:
            if days_until_target < 30:
                multiplier = 0.85
                reason = f"Short timeframe ({days_until_target} days) - large moves less likely"
            elif days_until_target < 90:
                multiplier = 0.95
                reason = f"Medium-short timeframe ({days_until_target} days) - slight reduction"
            elif days_until_target > 365:
                multiplier = 1.10
                reason = f"Long timeframe ({days_until_target} days) - more time for target"
            else:
                multiplier = 1.0
                reason = f"Standard timeframe ({days_until_target} days) - no adjustment"

            if multiplier != 1.0:
                self.add_adjustment("timeframe", multiplier, reason)
                adjusted_prob *= multiplier

        # Trend alignment adjustment
        if target_price and current_price:
            is_bullish_target = target_price > current_price
            is_bullish_signal = weighted_signal > 0.15

            if is_bullish_target and weighted_signal < -0.15:
                multiplier = 0.80
                reason = "Bullish target but bearish signals - reduced probability"
                self.add_adjustment("trend_alignment", multiplier, reason)
                adjusted_prob *= multiplier
            elif not is_bullish_target and weighted_signal > 0.15:
                multiplier = 0.80
                reason = "Bearish target but bullish signals - reduced probability"
                self.add_adjustment("trend_alignment", multiplier, reason)
                adjusted_prob *= multiplier

        # Clamp to bounds
        self.trail.final_probability = max(15, min(85, adjusted_prob))
        return self.trail.final_probability

    def build_decision_trail(self) -> DecisionTrail:
        """
        Build the final DecisionTrail object for the response.

        Returns:
            DecisionTrail with all nodes and calculations
        """
        nodes = []

        for factor in self.trail.factors:
            node = DecisionNode(
                id=str(uuid.uuid4())[:8],
                category=factor.category,
                source=factor.source,
                data_point=factor.data_point,
                value=factor.raw_value,
                signal=factor.signal,
                weight=factor.weight,
                score_contribution=round(factor.contribution, 2),
                reasoning=factor.reasoning,
            )
            nodes.append(node)

        # Calculate category scores
        category_scores = {}
        category_counts = {}
        for factor in self.trail.factors:
            cat = factor.category
            if cat not in category_scores:
                category_scores[cat] = 0.0
                category_counts[cat] = 0
            category_scores[cat] += factor.contribution
            category_counts[cat] += 1

        # Build final calculation string
        calc_parts = [f"Base: 50%"]

        for cat, score in sorted(category_scores.items(), key=lambda x: abs(x[1]), reverse=True):
            if score != 0:
                calc_parts.append(f"{cat.replace('_', ' ').title()}: {score:+.1f}%")

        calc_parts.append(f"Pre-adjustment: {self.trail.pre_adjustment_probability:.1f}%")

        for adj in self.trail.adjustments:
            calc_parts.append(f"{adj['name'].replace('_', ' ').title()} (×{adj['multiplier']:.2f})")

        calc_parts.append(f"Final: {self.trail.final_probability:.0f}%")

        return DecisionTrail(
            nodes=nodes,
            category_scores={cat: round(score, 2) for cat, score in category_scores.items()},
            category_weights=self.CATEGORY_WEIGHTS.copy(),
            final_calculation=" → ".join(calc_parts),
        )

    def generate_ascii_tree(self) -> str:
        """
        Generate an ASCII visualization of the decision trail.

        Returns:
            ASCII art decision tree string
        """
        lines = []
        lines.append("┌" + "─" * 60 + "┐")
        lines.append(f"│ DECISION TRAIL: {self.trail.final_probability:.0f}% Probability{' ' * 26}│")
        lines.append("├" + "─" * 60 + "┤")

        # Group factors by category
        by_category = {}
        for factor in self.trail.factors:
            if factor.category not in by_category:
                by_category[factor.category] = []
            by_category[factor.category].append(factor)

        # Sort categories by weight
        sorted_cats = sorted(
            by_category.keys(),
            key=lambda c: self.CATEGORY_WEIGHTS.get(c, 0),
            reverse=True
        )

        lines.append("│ BASE PROBABILITY: 50%{' ' * 37}│")
        lines.append("│{' ' * 60}│")

        for cat in sorted_cats:
            factors = by_category[cat]
            total_contribution = sum(f.contribution for f in factors)
            weight_pct = int(self.CATEGORY_WEIGHTS.get(cat, 0.10) * 100)

            cat_display = cat.upper().replace("_", " ")
            header = f"│ ┌─ {cat_display} ({weight_pct}% weight)"
            contrib = f"{total_contribution:+.1f}%"
            padding = 60 - len(header) - len(contrib) - 2
            lines.append(f"{header}{' ' * padding}{contrib} ─┐│")

            for factor in factors:
                data = f"{factor.data_point}: {factor.raw_value or 'N/A'}"
                if len(data) > 35:
                    data = data[:32] + "..."
                contrib_str = f"{factor.contribution:+.1f}%"
                padding = 55 - len(data) - len(contrib_str)
                lines.append(f"│ │  ├─ {data}{' ' * max(1, padding)}{contrib_str}│ │")

            lines.append("│ └" + "─" * 55 + "┘ │")
            lines.append("│{' ' * 60}│")

        # Adjustments
        if self.trail.adjustments:
            lines.append("│ ADJUSTMENTS:{' ' * 47}│")
            for adj in self.trail.adjustments:
                name = adj['name'].replace('_', ' ').title()
                mult = f"×{adj['multiplier']:.2f}"
                line = f"│  ├─ {name}: {mult}"
                lines.append(f"{line}{' ' * (60 - len(line))}│")
            lines.append("│{' ' * 60}│")

        # Final
        final_line = f"│ FINAL PROBABILITY: {self.trail.final_probability:.0f}%"
        lines.append(f"{final_line}{' ' * (61 - len(final_line))}│")
        lines.append("└" + "─" * 60 + "┘")

        return "\n".join(lines)


# Factory function
def create_decision_trail_builder() -> DecisionTrailBuilder:
    """Create a new decision trail builder instance."""
    return DecisionTrailBuilder()
