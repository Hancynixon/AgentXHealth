import numpy as np
import pandas as pd



class FairnessEvaluator:

    def evaluate(self, df):

        report = {}

        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[0, 30, 45, 60, 100],
            labels=["Young", "Adult", "MidAge", "Senior"]
        )

        for group in df["AgeGroup"].unique():

            subset = df[df["AgeGroup"] == group]

            if len(subset) == 0:
                continue

            mean_risk = subset["final_risk"].mean()

            accuracy = (
                (subset["final_risk"] >= 0.5).astype(int)
                == subset["Outcome"]
            ).mean()

            report[str(group)] = {
                "mean_risk": round(float(mean_risk), 3),
                "accuracy": round(float(accuracy), 3)
            }

        return report
