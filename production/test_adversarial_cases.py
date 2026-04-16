"""
Adversarial Edge Case Test Suite for AgentXHealth
Run with: python -m production.test_adversarial_cases
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from production.model_pipeline import run_model_for_input


# =========================================================
# TEST CASE DEFINITIONS
# =========================================================
TEST_CASES = [
    {
        "id":             "Case 1",
        "name":           "False Negative Trap — Lean Diabetic (Type 1 Phenotype)",
        "trap":           "FALSE NEGATIVE",
        "input": {
            "Email":         "test1@agentxhealth.com",
            "Gender":        "Male",
            "Glucose":       350,
            "Insulin":       5,
            "BMI":           18.0,
            "BloodPressure": 110,
            "Age":           15,
            "Pregnancies":   0,
        },
        "expect_label":   "HIGH",
        "expect_min_risk": 0.70,
        "expect_max_risk": None,
        "note": "Critical hyperglycemia + very low insulin must override demographic low-risk profile.",
    },
    {
        "id":             "Case 2",
        "name":           "False Positive Trap — Healthy Obese",
        "trap":           "FALSE POSITIVE",
        "input": {
            "Email":         "test2@agentxhealth.com",
            "Gender":        "Female",
            "Glucose":       85,
            "Insulin":       15,
            "BMI":           55.0,
            "BloodPressure": 180,
            "Age":           85,
            "Pregnancies":   6,
        },
        "expect_label":   None,              # flexible — model may score moderate due to boosts
        "expect_min_risk": None,
        "expect_max_risk": 0.75,             # should NOT hit HIGH purely from BMI/Age/BP
        "note": "Glucose and insulin are healthy. BMI/BP/Age boosts may push moderate but not HIGH.",
    },
    {
        "id":             "Case 3",
        "name":           "Hidden Risk — Severe Insulin Resistance",
        "trap":           "HIDDEN RISK",
        "input": {
            "Email":         "test3@agentxhealth.com",
            "Gender":        "Female",
            "Glucose":       95,
            "Insulin":       380,
            "BMI":           35.0,
            "BloodPressure": 130,
            "Age":           45,
            "Pregnancies":   2,
        },
        "expect_label":   "HIGH",
        "expect_min_risk": 0.65,
        "expect_max_risk": None,
        "note": "Extreme hyperinsulinemia must trigger insulin override and clinical floor escalation.",
    },
    {
        "id":             "Case 4",
        "name":           "Maximum Overload Point",
        "trap":           "MAX OVERLOAD",
        "input": {
            "Email":         "test4@agentxhealth.com",
            "Gender":        "Female",
            "Glucose":       400,
            "Insulin":       400,
            "BMI":           60.0,
            "BloodPressure": 220,
            "Age":           100,
            "Pregnancies":   20,
        },
        "expect_label":   "HIGH",
        "expect_min_risk": 0.95,
        "expect_max_risk": None,
        "note": "Every metric is extreme. Model must not behave erratically — should confidently hit 0.99.",
    },
    {
        "id":             "Case 5",
        "name":           "Minimum Baseline — Severe Hypoglycemia",
        "trap":           "MIN BASELINE",
        "input": {
            "Email":         "test5@agentxhealth.com",
            "Gender":        "Male",
            "Glucose":       40,
            "Insulin":       2,
            "BMI":           12.0,
            "BloodPressure": 60,
            "Age":           10,
            "Pregnancies":   0,
        },
        "expect_label":   "HIGH",
        "expect_min_risk": 0.75,
        "expect_max_risk": None,
        "note": "Hypoglycemia override (floor 0.85) + hypotension override (floor 0.75) must both fire.",
    },
    {
        "id":             "Case 6A",
        "name":           "Gender Bias Check — Female",
        "trap":           "GENDER BIAS",
        "input": {
            "Email":         "test6a@agentxhealth.com",
            "Gender":        "Female",
            "Glucose":       140,
            "Insulin":       150,
            "BMI":           30.0,
            "BloodPressure": 120,
            "Age":           40,
            "Pregnancies":   0,
        },
        "expect_label":   "HIGH",
        "expect_min_risk": 0.65,
        "expect_max_risk": None,
        "note": "Identical vitals to Case 6B. Compare scores — gap must be < 0.05.",
    },
    {
        "id":             "Case 6B",
        "name":           "Gender Bias Check — Male (identical vitals)",
        "trap":           "GENDER BIAS",
        "input": {
            "Email":         "test6b@agentxhealth.com",
            "Gender":        "Male",
            "Glucose":       140,
            "Insulin":       150,
            "BMI":           30.0,
            "BloodPressure": 120,
            "Age":           40,
            "Pregnancies":   0,
        },
        "expect_label":   "HIGH",
        "expect_min_risk": 0.65,
        "expect_max_risk": None,
        "note": "Identical vitals to Case 6A. Compare scores — gap must be < 0.05.",
    },
]


# =========================================================
# EVALUATOR
# =========================================================
def evaluate(case_id, result, tc):
    """Returns (passed: bool, reason: str)"""
    if result.get("error_state"):
        return False, f"Pipeline error: {result.get('explanation','unknown')}"

    risk  = result.get("final_risk", -1.0)
    label = result.get("risk_label", "N/A")
    reasons = []

    if tc.get("expect_label") and label != tc["expect_label"]:
        reasons.append(f"Label={label} but expected {tc['expect_label']}")

    if tc.get("expect_min_risk") and risk < tc["expect_min_risk"]:
        reasons.append(f"Risk={risk:.4f} below minimum {tc['expect_min_risk']}")

    if tc.get("expect_max_risk") and risk > tc["expect_max_risk"]:
        reasons.append(f"Risk={risk:.4f} above maximum {tc['expect_max_risk']}")

    if reasons:
        return False, " | ".join(reasons)
    return True, "All checks passed"


# =========================================================
# MAIN TEST RUNNER
# =========================================================
def run_tests():
    print("=" * 65)
    print("  AgentXHealth — Adversarial Edge Case Test Suite")
    print("=" * 65)

    results_log = []
    risk_scores = {}

    for tc in TEST_CASES:
        print(f"\n{'─'*65}")
        print(f"  {tc['id']} | {tc['trap']}")
        print(f"  {tc['name']}")
        print(f"  Input: Glucose={tc['input']['Glucose']} | "
              f"Insulin={tc['input']['Insulin']} | "
              f"BMI={tc['input']['BMI']} | "
              f"BP={tc['input']['BloodPressure']} | "
              f"Age={tc['input']['Age']} | "
              f"Gender={tc['input']['Gender']}")

        try:
            result = run_model_for_input(tc["input"])
        except Exception as e:
            result = {
                "error_state":    True,
                "explanation":    str(e),
                "final_risk":     -1.0,
                "risk_label":     "ERROR",
                "dominant_agent": "Exception",
            }

        risk   = result.get("final_risk", -1.0)
        label  = result.get("risk_label", "ERROR")
        agent  = result.get("dominant_agent", "N/A")
        passed, reason = evaluate(tc["id"], result, tc)

        risk_scores[tc["id"]] = risk

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n  Result  : {label} RISK | Final Risk = {risk:.4f}")
        print(f"  Dominant: {agent}")
        print(f"  Test    : {status} — {reason}")
        print(f"  Note    : {tc['note']}")

        if result.get("clinical_warnings"):
            for w in result["clinical_warnings"]:
                print(f"  ⚠ {w}")

        results_log.append({
            "id":     tc["id"],
            "name":   tc["name"],
            "trap":   tc["trap"],
            "risk":   risk,
            "label":  label,
            "passed": passed,
            "reason": reason,
        })

    # ── Gender Bias Delta Check ─────────────────────────────
    print(f"\n{'─'*65}")
    print("  GENDER BIAS ANALYSIS (Case 6A vs 6B)")
    score_6a = risk_scores.get("Case 6A", -1)
    score_6b = risk_scores.get("Case 6B", -1)
    if score_6a > 0 and score_6b > 0:
        delta = abs(score_6a - score_6b)
        bias_ok = delta < 0.05
        bias_status = "✅ NO BIAS" if bias_ok else "❌ GENDER BIAS DETECTED"
        print(f"  Female Risk : {score_6a:.4f}")
        print(f"  Male Risk   : {score_6b:.4f}")
        print(f"  Delta       : {delta:.4f} (threshold < 0.05)")
        print(f"  Result      : {bias_status}")

        # Update pass/fail for 6A and 6B based on delta
        for r in results_log:
            if r["id"] in ("Case 6A", "Case 6B"):
                if not bias_ok:
                    r["passed"] = False
                    r["reason"] += f" | Gender delta={delta:.4f} exceeds 0.05"

    # ── Final Summary ───────────────────────────────────────
    total  = len(results_log)
    passed = sum(1 for r in results_log if r["passed"])
    failed = total - passed

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS: {passed}/{total} passed | {failed} failed")
    print(f"{'='*65}")
    print(f"  {'ID':<10} {'TRAP':<18} {'RISK':>8} {'LABEL':<12} {'STATUS'}")
    print(f"  {'─'*60}")
    for r in results_log:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"  {r['id']:<10} {r['trap']:<18} {r['risk']:>8.4f} {r['label']:<12} {status}")
        if not r["passed"]:
            print(f"  {'':10} Reason: {r['reason']}")
    print(f"{'='*65}")

    if failed == 0:
        print("  🎉 All adversarial cases passed! Pipeline is robust.")
    else:
        print(f"  ⚠  {failed} case(s) need attention — review clinical override rules.")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    run_tests()
