class RiskEngine:
    """
    Converts predicted diabetes probability into
    risk categories and recommended actions.
    """

    def __init__(self, low=0.3, high=0.6):
        self.low_threshold = low
        self.high_threshold = high

    def assign_risk(self, probability: float) -> dict:
        if probability < self.low_threshold:
            return {
                "risk_level": "Low Risk",
                "recommendation": "Maintain healthy lifestyle and periodic monitoring."
            }

        elif probability < self.high_threshold:
            return {
                "risk_level": "Moderate Risk",
                "recommendation": "Recommend lifestyle intervention and follow-up screening."
            }

        else:
            return {
                "risk_level": "High Risk",
                "recommendation": "Recommend immediate clinical evaluation and diagnostic tests."
            }
