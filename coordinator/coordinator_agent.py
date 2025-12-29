import pandas as pd
from dataclasses import dataclass

@dataclass
class CoordinatorMetadata:
    lab_metadata: object
    physical_metadata: object
    demographic_metadata: object
    total_features: int


class CoordinatorAgent:
    """
    Coordinator Agent responsible for:
    - Executing all domain agents
    - Fusing their outputs
    - Preserving metadata
    """

    def __init__(self, lab_agent, physical_agent, demographic_agent):
        self.lab_agent = lab_agent
        self.physical_agent = physical_agent
        self.demographic_agent = demographic_agent
        self.metadata = None

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # Run individual agents
        lab_df = self.lab_agent.preprocess(df)
        physical_df = self.physical_agent.preprocess(df)
        demographic_df = self.demographic_agent.preprocess(df)

        # Merge all agent outputs
        final_df = pd.concat(
            [lab_df, physical_df, demographic_df],
            axis=1
        )

        # Save metadata
        self.metadata = CoordinatorMetadata(
            lab_metadata=self.lab_agent.get_metadata(),
            physical_metadata=self.physical_agent.get_metadata(),
            demographic_metadata=self.demographic_agent.get_metadata(),
            total_features=final_df.shape[1]
        )

        return final_df

    def get_metadata(self) -> CoordinatorMetadata:
        return self.metadata