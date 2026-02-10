class CounterfactualReasoner:
    """
    CounterfactualReasoner
    ----------------------
    Estimates which agent-level intervention would most
    reduce overall system risk.
    """

    def __init__(self, step=0.1):
        self.step = step  # simulated improvement

    def rank_interventions(self, agent_outputs, coordinator):
        base_decision = coordinator.reason(agent_outputs)
        base_risk = base_decision["final_risk"]

        impacts = {}

        for agent_name, agent_data in agent_outputs.items():
            modified_outputs = {}

            # Deep copy all agents
            for k, v in agent_outputs.items():
                modified_outputs[k] = v.copy()

            # Reduce primary risk score
            risk_key = [k for k in agent_data.keys() if "risk_score" in k][0]
            modified_outputs[agent_name][risk_key] = max(
                0.0, agent_data[risk_key] - self.step
            )

            new_risk = coordinator.reason(modified_outputs)["final_risk"]
            impacts[agent_name] = round(base_risk - new_risk, 3)

        ranked = sorted(impacts.items(), key=lambda x: x[1], reverse=True)

        return {
            "base_risk": base_risk,
            "ranked_interventions": ranked
        }
