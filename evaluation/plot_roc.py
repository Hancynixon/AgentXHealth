import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

# ===============================
# Load Results
# ===============================
df = pd.read_csv("results_with_risk.csv")

y_true = df["Outcome"]
y_score = df["final_risk"]

# ===============================
# Compute ROC
# ===============================
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# ===============================
# Plot Settings (Journal Style)
# ===============================
plt.style.use("default")
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(6, 6))

plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – AgentXHealth")
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig("Figure_2_ROC_Curve.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure_2_ROC_Curve.png saved successfully.")