class CoordinatorReasoner:
    """
    CoordinatorReasoner
    -------------------
    Interprets and arbitrates between agent-level risk assessments
    to produce a final system decision with reasoning.
    """

    def __init__(self):
        # Clinically motivated weights
        self.weights = {
            "lab": 0.6,
            "physical": 0.25,
            "demographic": 0.15
        }

    def reason(self, agent_outputs: dict) -> dict:
        """
        Combine agent risks, detect conflicts, and explain decision.
        """

        # Extract risks
        lab_risk = agent_outputs["lab"]["lab_risk_score"]
        phys_risk = agent_outputs.get("physical", {}).get("physical_risk_score", 0.0)
        demo_risk = agent_outputs.get("demographic", {}).get("demographic_risk_score", 0.0)

        # Weighted contributions
        contributions = {
            "lab": self.weights["lab"] * lab_risk,
            "physical": self.weights["physical"] * phys_risk,
            "demographic": self.weights["demographic"] * demo_risk
        }

        final_risk = sum(contributions.values())

        # Dominant agent
        dominant_agent = max(contributions, key=contributions.get)

        # Conflict detection
        conflict = self._detect_conflict(lab_risk, phys_risk, demo_risk)

        # Explanation
        explanation = self._generate_explanation(dominant_agent, conflict)

        return {
            "final_risk": round(final_risk, 3),
            "agent_contributions": {k: round(v, 3) for k, v in contributions.items()},
            "dominant_agent": dominant_agent,
            "conflict_detected": conflict,
            "decision_explanation": explanation
        }

    def _detect_conflict(self, lab, phys, demo) -> bool:
        risks = [lab, phys, demo]
        return max(risks) > 0.6 and min(risks) < 0.3

    def _generate_explanation(self, dominant_agent: str, conflict: bool) -> str:
        if conflict:
            return (
                f"Conflicting signals detected. "
                f"{dominant_agent.capitalize()} indicators dominate despite disagreement."
            )
        else:
            return f"{dominant_agent.capitalize()} indicators primarily drive overall risk."
