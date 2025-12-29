import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class DemographicAgentMetadata:
    features_processed: list
    original_zero_counts: dict
    imputation_medians: dict
    imputed_counts: dict


class DemographicAgent:
    """
    Demographic Agent responsible for preprocessing
    age and pregnancy-related features AND recording metadata.
    """

    def __init__(self):
        self.metadata = None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        demo_cols = ["Age", "Pregnancies"]
        demo_df = df[demo_cols].copy()

        original_zero_counts = {}
        imputation_medians = {}
        imputed_counts = {}

        # AGE: 0 is invalid → treat as missing
        age_zero_count = (demo_df["Age"] == 0).sum()
        original_zero_counts["Age"] = int(age_zero_count)

        demo_df["Age"] = demo_df["Age"].replace(0, np.nan)

        age_median = demo_df["Age"].median()
        imputation_medians["Age"] = float(age_median)

        missing_before = demo_df["Age"].isna().sum()
        demo_df["Age"] = demo_df["Age"].fillna(age_median)
        imputed_counts["Age"] = int(missing_before)

        # PREGNANCIES: 0 is valid → no imputation
        original_zero_counts["Pregnancies"] = int((demo_df["Pregnancies"] == 0).sum())
        imputation_medians["Pregnancies"] = None
        imputed_counts["Pregnancies"] = 0

        # Save metadata
        self.metadata = DemographicAgentMetadata(
            features_processed=demo_cols,
            original_zero_counts=original_zero_counts,
            imputation_medians=imputation_medians,
            imputed_counts=imputed_counts
        )

        return demo_df

    def get_metadata(self) -> DemographicAgentMetadata:
        return self.metadata
