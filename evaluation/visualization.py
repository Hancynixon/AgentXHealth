import matplotlib.pyplot as plt


class RiskVisualizer:

    def generate_all(self, df):

        # Plot risk distribution
        if "final_risk" not in df.columns:
            print("No final_risk column found.")
            return

        plt.figure(figsize=(8, 5))
        plt.hist(df["final_risk"], bins=20)
        plt.title("Final Risk Distribution")
        plt.xlabel("Risk Score")
        plt.ylabel("Frequency")
        plt.show()

        # If agent risks exist
        agent_cols = [col for col in df.columns if "risk" in col.lower()]

        if len(agent_cols) >= 3:
            plt.figure(figsize=(8, 5))
            df[agent_cols].mean().plot(kind="bar")
            plt.title("Average Agent Risk Scores")
            plt.ylim(0, 1)
            plt.show()