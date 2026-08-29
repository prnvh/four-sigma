from __future__ import annotations

from typing import Any

from memory.context_gateway import RiskAnalystContext
from memory.types import Direction, RiskAnalysis, RiskCategory, RiskFactor, jsonable

from .model import ModelClient


RISK_FACTOR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "category": {"type": "string", "enum": [item.value for item in RiskCategory]},
        "probability_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "severity": {"type": "integer", "minimum": 1, "maximum": 5},
        "impact": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "mitigants": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["category", "probability_pct", "severity", "impact", "evidence_refs", "mitigants"],
}

RISK_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_risk_score": {"type": "number", "minimum": 0, "maximum": 100},
        "success_probability_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "neutral_probability_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "failure_probability_pct": {"type": "number", "minimum": 0, "maximum": 100},
        "risk_factors": {"type": "array", "items": RISK_FACTOR_SCHEMA},
        "hidden_assumptions": {"type": "array", "items": {"type": "string"}},
        "second_order_effects": {"type": "array", "items": {"type": "string"}},
        "success_conditions": {"type": "array", "items": {"type": "string"}},
        "failure_conditions": {"type": "array", "items": {"type": "string"}},
        "coverage_gaps": {
            "type": "array",
            "items": {"type": "string", "enum": [item.value for item in RiskCategory]},
        },
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": [
        "overall_risk_score", "success_probability_pct", "neutral_probability_pct",
        "failure_probability_pct", "risk_factors", "hidden_assumptions",
        "second_order_effects", "success_conditions", "failure_conditions",
        "coverage_gaps", "evidence_refs",
    ],
}


class RiskAnalyst:
    """Evidence-bound qualitative risk and scenario-probability analyst."""

    _required = set(RISK_ANALYSIS_SCHEMA["required"])
    _text_lists = {
        "hidden_assumptions", "second_order_effects", "success_conditions",
        "failure_conditions", "coverage_gaps", "evidence_refs",
    }

    def __init__(self, model: ModelClient) -> None:
        self.model = model

    def analyze(self, context: RiskAnalystContext) -> RiskAnalysis:
        if not isinstance(context, RiskAnalystContext):
            raise TypeError("RiskAnalyst requires context from ContextGateway")
        if not context.market_features:
            raise ValueError("risk outcome estimates require point-in-time market features")
        if not (context.company_analyses or context.records or context.promoted_insights):
            raise ValueError("risk analysis requires company evidence or approved analysis")

        allowed_refs = (
            {item.ref for item in context.company_analyses}
            | {item.ref for item in context.records}
            | {item.ref for item in context.promoted_insights}
            | {item.ref for item in context.market_features}
        )
        result = self.model.generate_json(
            instructions=(
                "You are QFIRM's AI Risk Analyst. Identify evidence-backed company and stock "
                "risks, hidden thesis assumptions, invalidation paths, regime sensitivity, and "
                "second-order effects. Assess business, financial, liquidity, market, regulatory, "
                "operational, governance, event, sentiment, and data/model risk. Put any category "
                "that cannot be assessed from supplied evidence in coverage_gaps; never invent a "
                "risk fact. Estimate success, neutral, and failure percentages against the exact "
                "return thresholds and horizon supplied by the caller. They must total 100. These "
                "are uncertain model estimates, not measured frequencies, guarantees, trade "
                "approval, or deterministic risk calculations. Cite only supplied refs."
            ),
            input_data={
                "symbol": context.symbol,
                "simulation_time": context.simulation_time.isoformat(),
                "outcome_definition": jsonable(context.outcome),
                "company_analyses": [jsonable(item) for item in context.company_analyses],
                "company_records": [jsonable(item) for item in context.records],
                "promoted_insights": [jsonable(item) for item in context.promoted_insights],
                "market_features": [jsonable(item) for item in context.market_features],
            },
            schema=RISK_ANALYSIS_SCHEMA,
        )
        self._validate(result, allowed_refs)
        return RiskAnalysis(
            symbol=context.symbol,
            horizon_days=context.outcome.horizon_days,
            overall_risk_score=float(result["overall_risk_score"]),
            success_probability_pct=float(result["success_probability_pct"]),
            neutral_probability_pct=float(result["neutral_probability_pct"]),
            failure_probability_pct=float(result["failure_probability_pct"]),
            risk_factors=tuple(self._factor(item) for item in result["risk_factors"]),
            hidden_assumptions=self._clean(result["hidden_assumptions"]),
            second_order_effects=self._clean(result["second_order_effects"]),
            success_conditions=self._clean(result["success_conditions"]),
            failure_conditions=self._clean(result["failure_conditions"]),
            coverage_gaps=tuple(result["coverage_gaps"]),
            evidence_refs=tuple(dict.fromkeys(result["evidence_refs"])),
        )

    def _validate(self, result: Any, allowed_refs: set[str]) -> None:
        if not isinstance(result, dict) or set(result) != self._required:
            raise ValueError("risk analyst output has missing or unexpected fields")
        numeric = (
            "overall_risk_score", "success_probability_pct", "neutral_probability_pct",
            "failure_probability_pct",
        )
        for field in numeric:
            value = result[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be numeric")
        probabilities = [result[field] for field in numeric[1:]]
        if any(not 0 <= value <= 100 for value in probabilities) or abs(sum(probabilities) - 100) > 0.01:
            raise ValueError("success, neutral and failure probabilities must total 100")
        if not 0 <= result["overall_risk_score"] <= 100:
            raise ValueError("overall_risk_score must be between 0 and 100")
        for field in self._text_lists:
            if not isinstance(result[field], list) or not all(isinstance(x, str) for x in result[field]):
                raise ValueError(f"{field} must be a list of strings")
        if not result["evidence_refs"] or set(result["evidence_refs"]) - allowed_refs:
            raise ValueError("risk analysis contains missing or unknown evidence references")
        if not isinstance(result["risk_factors"], list):
            raise ValueError("risk_factors must be a list")

        assessed: set[str] = set()
        for factor in result["risk_factors"]:
            self._validate_factor(factor, allowed_refs)
            if factor["category"] in assessed:
                raise ValueError("risk categories cannot be duplicated")
            assessed.add(factor["category"])
        gaps = set(result["coverage_gaps"])
        all_categories = {item.value for item in RiskCategory}
        if assessed & gaps or assessed | gaps != all_categories:
            raise ValueError("each risk category must be assessed once or declared a coverage gap")

    @staticmethod
    def _validate_factor(factor: Any, allowed_refs: set[str]) -> None:
        required = {"category", "probability_pct", "severity", "impact", "evidence_refs", "mitigants"}
        if not isinstance(factor, dict) or set(factor) != required:
            raise ValueError("risk factor has missing or unexpected fields")
        if factor["category"] not in {item.value for item in RiskCategory}:
            raise ValueError("risk factor category is invalid")
        probability = factor["probability_pct"]
        severity = factor["severity"]
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 100:
            raise ValueError("risk factor probability must be between 0 and 100")
        if isinstance(severity, bool) or not isinstance(severity, int) or not 1 <= severity <= 5:
            raise ValueError("risk factor severity must be an integer from 1 to 5")
        if not isinstance(factor["impact"], str) or not factor["impact"].strip():
            raise ValueError("risk factor impact must be non-empty")
        refs = factor["evidence_refs"]
        if not isinstance(refs, list) or not refs or not all(isinstance(x, str) for x in refs):
            raise ValueError("risk factor requires evidence references")
        if set(refs) - allowed_refs:
            raise ValueError("risk factor cited unknown evidence")
        if not isinstance(factor["mitigants"], list) or not all(
            isinstance(x, str) for x in factor["mitigants"]
        ):
            raise ValueError("risk factor mitigants must be a list of strings")

    @staticmethod
    def _factor(value: dict[str, Any]) -> RiskFactor:
        return RiskFactor(
            category=RiskCategory(value["category"]),
            probability_pct=float(value["probability_pct"]),
            severity=int(value["severity"]),
            impact=value["impact"].strip(),
            evidence_refs=tuple(dict.fromkeys(value["evidence_refs"])),
            mitigants=RiskAnalyst._clean(value["mitigants"]),
        )

    @staticmethod
    def _clean(values: list[str]) -> tuple[str, ...]:
        return tuple(value.strip() for value in values if value.strip())
