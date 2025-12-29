import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class LabAgentMetadata:
    features_processed: list
    original_zero_counts: dict
    imputation_medians: dict
    imputed_counts: dict


class LabAgent:
    """
    Lab Agent responsible for preprocessing
    laboratory features AND recording metadata.
    """

    def __init__(self):
        self.metadata = None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        lab_cols = ["Glucose", "Insulin"]
        lab_df = df[lab_cols].copy()

        original_zero_counts = {}
        imputation_medians = {}
        imputed_counts = {}

        # Replace 0 -> NaN and record zero counts
        for col in lab_cols:
            zero_count = (lab_df[col] == 0).sum()
            original_zero_counts[col] = int(zero_count)
            lab_df[col] = lab_df[col].replace(0, np.nan)

        # Create insulin missingness indicator
        lab_df["Insulin_missing"] = lab_df["Insulin"].isna().astype(int)

        # Median imputation
        for col in lab_cols:
            median_value = lab_df[col].median()
            imputation_medians[col] = float(median_value)

            missing_before = lab_df[col].isna().sum()
            lab_df[col] = lab_df[col].fillna(median_value)
            imputed_counts[col] = int(missing_before)

        # Save metadata
        self.metadata = LabAgentMetadata(
            features_processed=lab_cols,
            original_zero_counts=original_zero_counts,
            imputation_medians=imputation_medians,
            imputed_counts=imputed_counts
        )

        return lab_df

    def get_metadata(self) -> LabAgentMetadata:
        return self.metadata
