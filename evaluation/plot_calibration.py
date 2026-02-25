import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve

# ===============================
# Load Results
# ===============================
df = pd.read_csv("results_with_risk.csv")

y_true = df["Outcome"]
y_prob = df["final_risk"]

# ===============================
# Compute Calibration
# ===============================
prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

# ===============================
# Plot Settings (Journal Style)
# ===============================
plt.style.use("default")
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(6, 6))

plt.plot(prob_pred, prob_true, marker='o', linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Proportion")
plt.title("Calibration Curve – AgentXHealth")

plt.tight_layout()
plt.savefig("Figure_3_Calibration_Curve.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure_3_Calibration_Curve.png saved successfully.")