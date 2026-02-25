import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


class ModelEvaluator:

    def evaluate(self, df):

        if "Outcome" not in df.columns:
            print("Outcome column missing.")
            return

        if "final_prediction" not in df.columns:
            print("final_prediction column missing.")
            return

        if "final_risk" not in df.columns:
            print("final_risk column missing.")
            return

        y_true = df["Outcome"]
        y_pred = df["final_prediction"]
        y_prob = df["final_risk"]

        print("\n==============================")
        print("FINAL MODEL EVALUATION")
        print("==============================")

        print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
        print("Precision:", round(precision_score(y_true, y_pred), 4))
        print("Recall   :", round(recall_score(y_true, y_pred), 4))
        print("F1 Score :", round(f1_score(y_true, y_pred), 4))
        print("ROC AUC  :", round(roc_auc_score(y_true, y_prob), 4))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))