import pandas as pd
import numpy as np

class PhysicalAgent:
    """
    Physical Agent responsible for preprocessing
    body measurement features.
    """

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        physical_cols = ["BloodPressure", "BMI", "SkinThickness"]
        physical_df = df[physical_cols].copy()

        # Step 1: Replace invalid zeros with NaN
        for col in physical_cols:
            physical_df[col] = physical_df[col].replace(0, np.nan)

        # Step 2: Missingness indicators
        for col in physical_cols:
            physical_df[f"{col}_missing"] = physical_df[col].isna().astype(int)

        # Step 3: Median imputation
        for col in physical_cols:
            median_value = physical_df[col].median()
            physical_df[col] = physical_df[col].fillna(median_value)

        return physical_df