import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class PhysicalAgentMetadata:
    features_processed: list
    original_zero_counts: dict
    imputation_medians: dict
    imputed_counts: dict


class PhysicalAgent:
    """
    Physical Agent responsible for preprocessing
    body measurement features AND recording metadata.
    """

    def __init__(self):
        self.metadata = None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        physical_cols = ["BloodPressure", "BMI", "SkinThickness"]
        physical_df = df[physical_cols].copy()

        original_zero_counts = {}
        imputation_medians = {}
        imputed_counts = {}

        # Replace 0 -> NaN and record zero counts
        for col in physical_cols:
            zero_count = (physical_df[col] == 0).sum()
            original_zero_counts[col] = int(zero_count)
            physical_df[col] = physical_df[col].replace(0, np.nan)

        # Create missingness indicators
        for col in physical_cols:
            physical_df[f"{col}_missing"] = physical_df[col].isna().astype(int)

        # Median imputation + record metadata
        for col in physical_cols:
            median_value = physical_df[col].median()
            imputation_medians[col] = float(median_value)

            missing_before = physical_df[col].isna().sum()
            physical_df[col] = physical_df[col].fillna(median_value)
            imputed_counts[col] = int(missing_before)

        # Save metadata
        self.metadata = PhysicalAgentMetadata(
            features_processed=physical_cols,
            original_zero_counts=original_zero_counts,
            imputation_medians=imputation_medians,
            imputed_counts=imputed_counts
        )

        return physical_df

    def get_metadata(self) -> PhysicalAgentMetadata:
        return self.metadata